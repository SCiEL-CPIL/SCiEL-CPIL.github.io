"""Parallel basin-hopping with MD-enhanced exploration for Morse clusters.

Strategy:
  1. Generate diverse structural seeds (FCC, BCC, random compact).
  2. Run multiple independent basin-hopping chains in parallel.
  3. Each chain uses adaptive perturbation scales, MD phases with adaptive
     thermostatting for barrier crossing, and periodic re-seeding.
  4. Final high-precision polish on the best configuration.
  5. Return the best configuration found within the time budget.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
import concurrent.futures as _cf
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 285.0
    return _T0


def _morse_E(P, rho):
    """Exact energy E = sum_{i<j} exp(rho(1-r))(exp(rho(1-r))-2)."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    r = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    iu = np.triu_indices(n, k=1)
    r = r[iu]
    t = np.exp(rho * (1.0 - r))
    return float((t * (t - 2.0)).sum())


def _morse_grad(P, rho):
    """Gradient of exact Morse energy w.r.t. coordinates (n,3)."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    D = P[:, None, :] - P[None, :, :]
    r = np.linalg.norm(D, axis=2)
    np.fill_diagonal(r, np.inf)
    t = np.exp(rho * (1.0 - r))
    c = rho * t * (2.0 * t - 2.0)
    inv_r = np.where(r > 1e-12, 1.0 / r, 0.0)
    W = c * inv_r
    rowsum = W.sum(axis=1)
    grad = -(rowsum[:, None] * P - W @ P)
    return grad


def _enforce_min_dist(P, min_d=0.5):
    """Robustly nudge so every pair is >= min_d apart."""
    P = np.array(P, dtype=float, copy=True)
    n = P.shape[0]
    for _ in range(500):
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        bad = (d < min_d)
        np.fill_diagonal(bad, False)
        if not bad.any():
            break
        idx = np.argwhere(bad)
        if len(idx) == 0:
            break
        i, j = idx[0]
        gap = min_d - d[i, j] + 0.01
        v = P[i] - P[j]
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            v = np.array([1.0, 0.0, 0.0]); nv = 1.0
        v = v / nv
        P[i] += 0.5 * gap * v
        P[j] -= 0.5 * gap * v
    return P


def _local_opt(P0, rho, max_iter=300, ftol=1e-10):
    """L-BFGS-B on exact Morse energy; returns (E, P)."""
    P0 = np.asarray(P0, dtype=float)
    n = P0.shape[0]

    def fun(x):
        P = x.reshape(n, 3)
        return _morse_E(P, rho)

    def gfun(x):
        P = x.reshape(n, 3)
        return _morse_grad(P, rho).ravel()

    try:
        res = minimize(fun, P0.ravel(), jac=gfun, method="L-BFGS-B",
                       options={"maxiter": max_iter, "maxfun": 10000,
                                "ftol": ftol, "gtol": 1e-7, "iprint": 0})
        P = res.x.reshape(n, 3)
        E = _morse_E(P, rho)
        return E, P
    except Exception:
        return None, None


def _md_phase(P, rho, n_steps=200, dt=0.005, T_target=0.5, deadline=None):
    """
    Non-equilibrium MD with adaptive thermostatting (Nosé-Hoover-like).
    Returns (E, P) after MD phase.
    
    Uses velocity Verlet integration with adaptive temperature control.
    """
    P = np.array(P, dtype=float, copy=True)
    V = np.zeros_like(P)
    n = P.shape[0]
    
    # Initialize velocities from thermal distribution
    rng = np.random.RandomState(int(time.time() * 1000) % 2**31)
    V = rng.normal(0, np.sqrt(T_target / n), size=P.shape)
    
    # Compute initial energy
    E_kin = 0.5 * np.sum(V**2)
    E_pot = _morse_E(P, rho)
    
    # Adaptive thermostat parameters
    T_current = T_target
    gamma = 0.1  # thermostat coupling
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline - 1.0:
            break
            
        # Velocity Verlet step 1: half-step velocity update
        F = -_morse_grad(P, rho)
        V_half = V + 0.5 * dt * F
        
        # Full-step position update
        P_new = P + dt * V_half
        
        # Compute new forces
        F_new = -_morse_grad(P_new, rho)
        
        # Full-step velocity update
        V_new = V_half + 0.5 * dt * F_new
        
        # Compute kinetic energy
        E_kin_new = 0.5 * np.sum(V_new**2)
        
        # Adaptive thermostat: adjust temperature to maintain target
        # Target kinetic energy for ideal gas: (3n/2) * T
        target_KE = 1.5 * n * T_target
        KE_ratio = E_kin_new / max(target_KE, 1e-10)
        
        # Scale velocities to maintain target temperature
        if KE_ratio > 1.5:
            scale = np.sqrt(1.0 / KE_ratio) * (1.0 + gamma * (KE_ratio - 1.0))
            V_new *= min(scale, 1.0)
        elif KE_ratio < 0.5:
            scale = np.sqrt(1.0 / max(KE_ratio, 1e-6)) * (1.0 + gamma * (1.0 - KE_ratio))
            V_new *= min(scale, 2.0)
        
        P = P_new
        V = V_new
        E_pot = _morse_E(P, rho)
        
        # Enforce minimum distance constraint
        P = _enforce_min_dist(P, min_d=0.5)
    
    E = _morse_E(P, rho)
    return E, P


def _fcc_seeds(n, rho):
    """Generate FCC lattice seeds with varying lattice constants."""
    seeds = []
    R = n ** (1/3.0) * 1.2
    for a in [0.9, 1.0, 1.1, 1.2]:
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0]
        ])
        pts = []
        extent = int(np.ceil(R / a)) + 2
        for i in range(-extent, extent + 1):
            for j in range(-extent, extent + 1):
                for k in range(-extent, extent + 1):
                    for b in basis:
                        p = np.array([i, j, k]) * a + b * a
                        if np.linalg.norm(p) <= R:
                            pts.append(p)
        if len(pts) >= n:
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            order = np.argsort(dists)
            pts = pts[order][:n]
            seeds.append(pts)
    return seeds


def _bcc_seeds(n, rho):
    """Generate BCC lattice seeds."""
    seeds = []
    R = n ** (1/3.0) * 1.2
    for a in [0.9, 1.0, 1.1]:
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5]
        ])
        pts = []
        extent = int(np.ceil(R / a)) + 2
        for i in range(-extent, extent + 1):
            for j in range(-extent, extent + 1):
                for k in range(-extent, extent + 1):
                    for b in basis:
                        p = np.array([i, j, k]) * a + b * a
                        if np.linalg.norm(p) <= R:
                            pts.append(p)
        if len(pts) >= n:
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            order = np.argsort(dists)
            pts = pts[order][:n]
            seeds.append(pts)
    return seeds


def _random_compact(n, seed=None):
    """Random compact cluster."""
    rng = np.random.RandomState(seed)
    R = n ** (1/3.0) * 1.5
    P = rng.uniform(-R, R, size=(n, 3))
    return P


def _basin_hopping_chain(P0, rho, n, deadline, rng_seed, global_best_tracker):
    """
    Run a basin-hopping chain with MD-enhanced exploration, adaptive step size,
    and periodic re-seeding.
    
    Args:
        P0: Initial positions
        rho: Morse parameter
        n: Number of atoms
        deadline: Time limit
        rng_seed: Random seed for reproducibility
        global_best_tracker: Shared dict with 'best_E' and 'best_P' for re-seeding
    
    Returns:
        (best_E, best_P) for this chain
    """
    rng = np.random.RandomState(rng_seed)
    
    # Initial local optimization
    P0 = _enforce_min_dist(P0)
    E, P = _local_opt(P0, rho, max_iter=200)
    if E is None:
        E, P = 1e10, P0
    
    best_E = E
    best_P = P.copy()
    
    # Adaptive perturbation scale
    step = 0.4
    stagnant_iters = 0
    max_stagnant = 25
    
    it = 0
    max_iters = 250
    
    while it < max_iters:
        if time.time() > deadline - 2.0:
            break
        
        # Periodic re-seeding: if stagnant for too long, jump from global best
        if stagnant_iters > max_stagnant and global_best_tracker.get('best_P') is not None:
            gp = global_best_tracker['best_P']
            # Add moderate noise to the global best
            noise = rng.normal(0, 0.3, size=gp.shape)
            P = _enforce_min_dist(gp + noise)
            E, P = _local_opt(P, rho, max_iter=150)
            if E is not None:
                if E < best_E:
                    best_E = E
                    best_P = P.copy()
                stagnant_iters = 0
            else:
                E, P = 1e10, P
        
        # Every 5 iterations, use MD phase instead of random perturbation
        if it % 5 == 0 and it > 0:
            # MD phase with adaptive temperature
            T_md = 0.3 + 0.2 * (it / max(1, max_iters))
            E_new, P_new = _md_phase(P, rho, n_steps=100, dt=0.005, 
                                     T_target=T_md, deadline=deadline)
            if E_new is None:
                E_new, P_new = 1e10, P
            
            # Follow up with local optimization
            E_new, P_new = _local_opt(P_new, rho, max_iter=100)
            if E_new is None:
                E_new, P_new = 1e10, P
        else:
            # Standard perturbation
            noise = rng.normal(0, step, size=P.shape)
            P_new = P + noise
            P_new = _enforce_min_dist(P_new)
            
            # Local optimization
            E_new, P_new = _local_opt(P_new, rho, max_iter=150)
            if E_new is None:
                E_new, P_new = 1e10, P
        
        # Track best
        if E_new < best_E:
            best_E = E_new
            best_P = P_new.copy()
            stagnant_iters = 0
        else:
            stagnant_iters += 1
        
        # Update global best tracker
        if E_new < global_best_tracker.get('best_E', float('inf')):
            global_best_tracker['best_E'] = E_new
            global_best_tracker['best_P'] = P_new.copy()
        
        # Metropolis acceptance with adaptive temperature
        T = 0.3 * (0.005 / 0.3) ** (it / max(1, max_iters - 1))
        if E_new <= E or rng.random() < np.exp(-(E_new - E) / max(T, 1e-12)):
            E, P = E_new, P_new
            # Increase step on acceptance
            step = min(step * 1.02, 0.8)
        else:
            # Decrease step on rejection
            step = max(step * 0.95, 0.1)
        
        it += 1
    
    return best_E, best_P


def _run_chain_wrapper(args):
    """Wrapper for parallel execution."""
    P0, rho, n, deadline, rng_seed, global_best_tracker = args
    try:
        return _basin_hopping_chain(P0, rho, n, deadline, rng_seed, global_best_tracker)
    except Exception:
        return None, None


def construct(instance):
    _init_timer()
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    call_start = time.time()
    deadline = min(_HARD_DEADLINE, call_start + 42.0)
    
    best_E = float('inf')
    best_P = None
    
    # Generate diverse seeds
    seeds = []
    seeds.extend(_fcc_seeds(n, rho))
    seeds.extend(_bcc_seeds(n, rho))
    for s in range(3):
        seeds.append(_random_compact(n, seed=100 + s))
    
    # Ensure we have at least some seeds
    if not seeds:
        seeds = [_random_compact(n, seed=0)]
    
    # Quick local optimization on all seeds to get baseline
    seed_results = []
    for i, s in enumerate(seeds):
        if time.time() > deadline - 10.0:
            break
        E, P = _local_opt(_enforce_min_dist(s), rho, max_iter=100)
        if E is not None:
            seed_results.append((E, P))
            if E < best_E:
                best_E = E
                best_P = P
    
    # Sort seeds by energy, use best ones for basin hopping
    seed_results.sort(key=lambda x: x[0])
    top_seeds = seed_results[:6] if len(seed_results) >= 6 else seed_results
    
    # Shared tracker for global best across chains
    global_best_tracker = {'best_E': best_E, 'best_P': best_P.copy() if best_P is not None else None}
    
    # Run basin hopping chains in parallel (up to 4 threads)
    n_chains = min(4, len(top_seeds))
    if n_chains > 0 and time.time() < deadline - 8.0:
        # Prepare arguments for parallel chains
        chain_args = []
        for i in range(n_chains):
            P0 = top_seeds[i][1]
            chain_args.append((P0, rho, n, deadline, 1000 + i, global_best_tracker))
        
        with _cf.ThreadPoolExecutor(max_workers=n_chains) as pool:
            futures = [pool.submit(_run_chain_wrapper, args) for args in chain_args]
            
            for f in futures:
                try:
                    E, P = f.result(timeout=1.0)
                    if E is not None and E < best_E:
                        best_E = E
                        best_P = P
                except Exception:
                    pass
        
        # Update from global tracker
        if global_best_tracker['best_E'] < best_E:
            best_E = global_best_tracker['best_E']
            best_P = global_best_tracker['best_P']
    
    # If we still have time, do a final high-precision polish on the best configuration
    if best_P is not None and time.time() < deadline - 3.0:
        # First, a short MD phase to escape any shallow local minimum
        E_md, P_md = _md_phase(_enforce_min_dist(best_P), rho, n_steps=50, 
                               dt=0.003, T_target=0.2, deadline=deadline)
        # Then, high-precision local optimization
        E, P = _local_opt(_enforce_min_dist(P_md), rho, max_iter=1000, ftol=1e-14)
        if E is not None and E < best_E:
            best_E = E
            best_P = P
        elif E is not None:
            # Even if not better, use the polished version
            best_P = P
    
    # Fallback if nothing worked
    if best_P is None:
        best_P = _random_compact(n, seed=42)
        best_E = _morse_E(best_P, rho)
    
    # Final validity check
    best_P = _enforce_min_dist(best_P)
    
    # One last quick relaxation to ensure validity
    if time.time() < deadline - 1.0:
        E, P = _local_opt(best_P, rho, max_iter=100)
        if E is not None:
            best_P = P
    
    best_P = _enforce_min_dist(best_P)
    
    return best_P

# EVOLVE-BLOCK-END
