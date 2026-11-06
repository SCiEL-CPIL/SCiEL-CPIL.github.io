"""Morse cluster global minima: Non-Equilibrium Molecular Dynamics (NEMD) with shocks."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from scipy.optimize import minimize

_START = time.time()
_HARD_DEADLINE = _START + 295.0  # safety margin before 300s limit / 330s kill


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(pos, rho):
    """Exact evaluator Morse energy: sum_{i<j} exp(rho(1-r))(exp(rho(1-r))-2)."""
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    iu = np.triu_indices(n, 1)
    d = d[iu]
    x = np.exp(rho * (1.0 - d))
    return float(np.sum(x * (x - 2.0)))


def _min_dist(pos):
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    if n < 2:
        return 1e10
    diff = pos[:, None, :] - pos[None, :, :]
    d = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
    iu = np.triu_indices(n, 1)
    return float(d[iu].min())


def _energy_grad(pos, rho):
    res = morse_energy_gradient(np.asarray(pos, dtype=float), rho)
    a, b = res
    if np.ndim(a) == 0:
        return float(a), np.asarray(b, dtype=float)
    else:
        return float(b), np.asarray(a, dtype=float)


def _fcc_points(n, a=np.sqrt(2.0)):
    pts = []
    L = int(np.ceil(n ** (1.0 / 3.0))) + 2
    offs = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]), np.array([0.0, 0.5, 0.5])]
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                base = np.array([i, j, k]) * a
                for off in offs:
                    pts.append(base + off)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _bcc_points(n, a=2.0 / np.sqrt(3.0)):
    pts = []
    L = int(np.ceil(n ** (1.0 / 3.0))) + 2
    offs = [np.array([0, 0, 0]), np.array([0.5, 0.5, 0.5])]
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                base = np.array([i, j, k]) * a
                for off in offs:
                    pts.append(base + off)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    r = np.linalg.norm(pts, axis=1)
    idx = np.argsort(r)[:n]
    return pts[idx]


def _random_compact(n, rng, spacing=1.0):
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    pts = []
    attempts = 0
    min_d = 0.7 * spacing
    while len(pts) < n and attempts < 5000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < min_d:
                    ok = False
                    break
            if ok:
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        pts.append(rng.uniform(-radius, radius, 3))
    return np.array(pts[:n])


def _enforce_min_dist(pos, min_d=0.5, iters=50):
    pos = np.array(pos, dtype=float, copy=True)
    n = len(pos)
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    if dist < 1e-9:
                        d = np.array([1e-6, 0, 0])
                        dist = 1e-6
                    push = d / dist * (min_d - dist) * 0.5
                    pos[i] += push
                    pos[j] -= push
                    moved = True
        if not moved:
            break
    return pos


def _nemd_trajectory(start_pos, rho, rng, time_budget, T_init=5.0, T_final=0.05):
    """
    Non-Equilibrium Molecular Dynamics with adaptive thermostatting and shocks.
    Uses Velocity Verlet integration.
    """
    pos = np.array(start_pos, dtype=float, copy=True)
    # Initial velocities
    v = rng.normal(0.0, 1.0, pos.shape)
    
    # Initial relaxation to ensure we start from a reasonable spot
    try:
        e0, pos_relaxed = relax(pos, 'morse', rho)
        pos = np.asarray(pos_relaxed, dtype=float)
        e0 = _morse_energy(pos, rho)
    except Exception:
        e0 = _morse_energy(pos, rho)
        
    best_pos = pos.copy()
    best_e = e0
    
    t0 = time.time()
    dt = 0.01  # Time step
    n_steps = 0
    shock_interval = 20  # Apply shock every N steps
    shock_magnitude = 2.0
    
    # Get initial gradient
    try:
        e_curr, g_curr = _energy_grad(pos, rho)
    except Exception:
        e_curr = _morse_energy(pos, rho)
        g_curr = np.zeros_like(pos)

    while (time.time() - t0) < time_budget and _time_left() > 3.0:
        # Adaptive Temperature: Cool down over time
        frac = min(1.0, (time.time() - t0) / max(time_budget, 1e-9))
        T = T_init + (T_final - T_init) * frac
        
        # Velocity Verlet Step 1: Update position
        pos_new = pos + v * dt + 0.5 * g_curr * (dt * dt) # Assuming mass=1, F=-g, a=F/m
        
        # Calculate new energy and gradient
        try:
            e_new, g_new = _energy_grad(pos_new, rho)
        except Exception:
            break
            
        # Check for min distance violation
        if _min_dist(pos_new) < 0.45: # Slightly looser during dynamics to allow passage, enforce later
            # If too close, reject step and dampen velocity
            v *= 0.5
            continue
            
        # Velocity Verlet Step 2: Update velocity
        # Average acceleration
        a_avg = -0.5 * (g_curr + g_new) # F = -g, m=1
        v_new = v + a_avg * dt
        
        # Thermostatting: Rescale velocity to match target kinetic energy
        # KE = 0.5 * m * v^2. Target KE = 1.5 * N * k_B * T (3D)
        ke_curr = 0.5 * np.sum(v_new ** 2)
        ke_target = 1.5 * len(pos) * T
        if ke_curr > 1e-6:
            scale = np.sqrt(ke_target / ke_curr)
            # Smoothing factor to avoid abrupt changes
            alpha = 0.1 
            scale = 1.0 + alpha * (scale - 1.0)
            v_new *= scale
            
        pos, v = pos_new, v_new
        e_curr, g_curr = e_new, g_new
        n_steps += 1
        
        # Update best
        if e_curr < best_e:
            best_e = e_curr
            best_pos = pos.copy()
            
        # Shock perturbation
        if n_steps % shock_interval == 0:
            # Random kick to break symmetry or escape local minima
            kick = rng.normal(0.0, shock_magnitude, pos.shape)
            v += kick
            # Also occasionally perturb position slightly
            if rng.random() < 0.1:
                pos += rng.normal(0.0, 0.1, pos.shape)
                # Recalculate e, g after position perturbation
                try:
                    e_curr, g_curr = _energy_grad(pos, rho)
                except Exception:
                    pass

    return best_pos, best_e


def _polish(pos, rho, time_budget):
    pos = np.array(pos, dtype=float, copy=True)
    try:
        e0, _ = relax(pos, 'morse', rho)
        e0 = _morse_energy(pos, rho)
    except Exception:
        e0 = _morse_energy(pos, rho)
    best_pos = pos.copy()
    best_e = e0
    t0 = time.time()
    # Gradient-based L-BFGS-B polish
    try:
        def fun(x):
            e, g = _energy_grad(x.reshape(-1, 3), rho)
            return e, g
        res = minimize(fun, pos.ravel(), method='L-BFGS-B', jac=True,
                       options={'maxiter': 300, 'maxfun': 3000,
                                'gtol': 1e-8, 'ftol': 1e-12})
        new_pos = res.x.reshape(-1, 3)
        if _min_dist(new_pos) >= 0.5:
            new_e = _morse_energy(new_pos, rho)
            if new_e < best_e:
                best_e = new_e
                best_pos = new_pos
    except Exception:
        pass
    # Final relax for safety
    if _time_left() - (time.time() - t0) > 0.5 and (time.time() - t0) < time_budget:
        try:
            _, p2 = relax(best_pos, 'morse', rho)
            p2 = np.asarray(p2, dtype=float)
            if _min_dist(p2) >= 0.5:
                e2 = _morse_energy(p2, rho)
                if e2 < best_e:
                    best_e = e2
                    best_pos = p2
        except Exception:
            pass
    return best_pos, best_e


def construct(instance):
    n = int(instance["n"])
    rho = instance.get("rho", instance.get("kind"))
    rho = float(rho)

    inst_start = time.time()
    inst_budget = 48.0

    def inst_left():
        return min(inst_budget, _time_left() - 3.0) - (time.time() - inst_start)

    rng = np.random.default_rng(n * 7919 + int(round(rho * 10)))

    # --- Seeds ---
    seeds = []
    try:
        seeds.append(_fcc_points(n))
        seeds.append(_bcc_points(n))
        seeds.append(_random_compact(n, rng))
        seeds.append(_fcc_points(n) + rng.normal(0, 0.15, (n, 3)))
        seeds.append(_bcc_points(n) + rng.normal(0, 0.15, (n, 3)))
    except Exception:
        seeds.append(_random_compact(n, rng))

    best_pos = None
    best_e = np.inf
    seed_results = []
    for seed in seeds:
        if inst_left() < 2.0:
            break
        try:
            e, p = relax(seed, 'morse', rho)
            p = np.asarray(p, dtype=float)
            e = _morse_energy(p, rho)
        except Exception:
            continue
        if e < best_e:
            best_e = e
            best_pos = p
        seed_results.append((e, p))

    if best_pos is None:
        # Fallback
        best_pos = _random_compact(n, rng)
        try:
            _, best_pos = relax(best_pos, 'morse', rho)
        except Exception:
            pass

    # Sort seeds by energy for chain starts
    seed_results.sort(key=lambda x: x[0])

    # --- NEMD Trajectories ---
    nemd_time = max(5.0, inst_left() - 4.0)
    n_chains = 3
    chain_budget = nemd_time / n_chains
    # Different temperature schedules for diversity
    schedules = [
        (10.0, 0.1),  # High start, low end
        (5.0, 0.05),  # Moderate
        (20.0, 0.5),  # Very high start, stays warm
    ]
    
    for ci in range(n_chains):
        if inst_left() < 3.0:
            break
        # Start chain from best available (rotate through top seeds for diversity)
        start = seed_results[ci % len(seed_results)][1] if seed_results else best_pos
        T_init, T_final = schedules[ci % len(schedules)]
        cb = min(chain_budget, inst_left() - 2.0)
        if cb < 1.0:
            break
        try:
            cpos, ce = _nemd_trajectory(start, rho, rng, cb, T_init, T_final)
        except Exception:
            continue
        if ce < best_e:
            best_e = ce
            best_pos = cpos

    # --- Final polish ---
    if inst_left() > 2.0 and best_pos is not None:
        try:
            ppol, epol = _polish(best_pos, rho, max(1.0, inst_left() - 1.0))
            if epol < best_e:
                best_e = epol
                best_pos = ppol
        except Exception:
            pass

    # --- Safety: enforce min distance ---
    best_pos = np.asarray(best_pos, dtype=float)
    if _min_dist(best_pos) < 0.5:
        best_pos = _enforce_min_dist(best_pos, 0.5)
        # re-relax after enforcement
        try:
            _, best_pos = relax(best_pos, 'morse', rho)
        except Exception:
            pass
        if _min_dist(best_pos) < 0.5:
            # last resort: use best seed
            if seed_results:
                best_pos = seed_results[0][1]

    return best_pos
# EVOLVE-BLOCK-END
