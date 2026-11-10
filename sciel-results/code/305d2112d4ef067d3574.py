"""Morse cluster global minima: basin hopping with structured crystalline seeds and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_DEADLINE = 325.0  # seconds; evaluator kills at 330s, keep safety margin

# Precompute indices for upper triangle
_MAX_N = 80
_iu_indices = [np.triu_indices(n, k=1) for n in range(2, _MAX_N + 1)]


def _morse_energy(pos, rho):
    """E = sum_{i<j} exp(rho(1-r_ij)) (exp(rho(1-r_ij)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    ii, jj = _iu_indices[n - 2]
    diff = pos[ii] - pos[jj]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    return float(np.sum(e))


def _morse_energy_and_grad(pos, rho):
    """Energy and gradient of the Morse potential."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    ii, jj = _iu_indices[n - 2]
    diff = pos[ii] - pos[jj]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    x = rho * (1.0 - r)
    ex = np.exp(x)
    e = ex * (ex - 2.0)
    energy = float(np.sum(e))
    de_dr = rho * (1.0 - 2.0 * ex) * ex
    grad = np.zeros_like(pos)
    inv_r = 1.0 / r
    f = (de_dr * inv_r)[:, None]
    np.add.at(grad, ii, f * diff)
    np.add.at(grad, jj, -f * diff)
    return energy, grad


def _is_valid(pos, min_dist=0.5):
    n = len(pos)
    if n < 2:
        return True
    ii, jj = _iu_indices[n - 2]
    diff = pos[ii] - pos[jj]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    return float(np.min(r)) >= min_dist


def _local_minimize(pos, rho, deadline):
    """Local minimization using L-BFGS-B on the gradient; fall back to relax."""
    pos = pos.astype(np.float64)

    def fun_g(x):
        if time.time() > deadline:
            return 1e30, np.zeros_like(x)
        e, g = _morse_energy_and_grad(x.reshape(-1, 3), rho)
        return e, g

    try:
        res = minimize(fun_g, pos.ravel(), method="L-BFGS-B",
                       options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-8,
                                "maxfun": 4000})
        if res.success or res.fun < _morse_energy(pos, rho):
            newpos = res.x.reshape(-1, 3)
            if _is_valid(newpos):
                return newpos
    except Exception:
        pass
    # fallback to provided relax
    try:
        e, newpos = relax(pos, "morse", rho)
        if _is_valid(newpos):
            return newpos
    except Exception:
        pass
    return pos


def _fcc_cluster(n, a=1.0):
    """Generate n points on an FCC lattice, centered."""
    basis = np.array([
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]
    ])
    R = int(np.ceil((n / 4.0) ** (1.0 / 3.0))) + 1
    i, j, k = np.meshgrid(range(-R, R + 1), range(-R, R + 1), range(-R, R + 1), indexing='ij')
    coords = np.stack([i, j, k], axis=-1).reshape(-1, 3)
    all_coords = []
    for b in basis:
        all_coords.append(coords + b)
    coords = np.vstack(all_coords)
    coords *= a
    d = np.linalg.norm(coords, axis=1)
    order = np.argsort(d)
    pts = coords[order[:n]]
    pts = pts - pts.mean(axis=0)
    return pts


def _bcc_cluster(n, a=1.0):
    """Generate n points on a BCC lattice, centered."""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    R = int(np.ceil((n / 2.0) ** (1.0 / 3.0))) + 1
    i, j, k = np.meshgrid(range(-R, R + 1), range(-R, R + 1), range(-R, R + 1), indexing='ij')
    coords = np.stack([i, j, k], axis=-1).reshape(-1, 3)
    all_coords = []
    for b in basis:
        all_coords.append(coords + b)
    coords = np.vstack(all_coords)
    coords *= a
    d = np.linalg.norm(coords, axis=1)
    order = np.argsort(d)
    pts = coords[order[:n]]
    pts = pts - pts.mean(axis=0)
    return pts


def _icosahedral(n):
    """Icosahedral shell construction (Mackay-like), centered."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    pts = [np.zeros(3)]
    shells = [1.0, np.sqrt(2.0 + 2.0 / phi), 2.0]
    for v in verts:
        pts.append(shells[0] * v)
    if len(pts) >= n:
        return np.array(pts[:n])
    
    shell2 = []
    perms = []
    for p in [(0, phi, 1), (0, -phi, 1), (0, phi, -1), (0, -phi, -1),
              (phi, 1, 0), (-phi, 1, 0), (phi, -1, 0), (-phi, -1, 0),
              (1, 0, phi), (-1, 0, phi), (1, 0, -phi), (-1, 0, -phi),
              (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)]:
        perms.append(p)
    # Add more permutations to get closer to 30
    for p in [(1, phi, 0), (1, -phi, 0), (-1, phi, 0), (-1, -phi, 0),
              (0, 1, phi), (0, 1, -phi), (0, -1, phi), (0, -1, -phi),
              (phi, 1, 0), (phi, -1, 0), (-phi, 1, 0), (-phi, -1, 0)]:
        perms.append(p)
        
    for perm in perms:
        p = np.array(perm, dtype=float)
        p = p / np.linalg.norm(p)
        shell2.append(shells[1] * p)
    pts.extend(shell2)
    if len(pts) >= n:
        return np.array(pts[:n])
        
    shell3 = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                p = verts[i] + verts[j] + verts[k]
                if np.linalg.norm(p) > 1e-6:
                    p = p / np.linalg.norm(p)
                    shell3.append(shells[2] * p)
    pts.extend(shell3)
    pts = np.array(pts)
    if len(pts) < n:
        rng = np.random.default_rng(0)
        while len(pts) < n:
            d = rng.normal(size=3)
            d = d / np.linalg.norm(d)
            pts = np.vstack([pts, shells[2] * d * rng.uniform(0.9, 1.1)])
    return pts[:n]


def _spherical_shell(n):
    """Fibonacci sphere, centered, radius chosen for packing."""
    if n == 1:
        return np.zeros((1, 3))
    radius = 0.5 * n ** (1.0 / 3.0) + 0.5
    pts = [np.zeros(3)]
    m = n - 1
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(m):
        z = 1.0 - 2.0 * (i + 0.5) / m
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * np.pi * i / golden
        pts.append(radius * np.array([r * np.cos(theta), r * np.sin(theta), z]))
    pts = np.array(pts)
    return pts - pts.mean(axis=0)


def _random_compact(n, spacing=1.0, seed=0):
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    # Simple random placement, validity checked later
    pts = rng.uniform(-radius, radius, size=(n, 3))
    return pts


def construct(instance):
    """Basin hopping over structured seeds with deadline guard."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3.0))
    start = time.time()
    deadline = start + _DEADLINE

    # Build candidate seed generators
    seeds = []
    try:
        seeds.append(_fcc_cluster(n))
    except Exception:
        pass
    try:
        seeds.append(_bcc_cluster(n))
    except Exception:
        pass
    try:
        seeds.append(_icosahedral(n))
    except Exception:
        pass
    try:
        seeds.append(_spherical_shell(n))
    except Exception:
        pass
    for s in range(8):
        try:
            seeds.append(_random_compact(n, seed=s))
        except Exception:
            pass

    best_pos = None
    best_energy = np.inf
    candidates = []

    def consider(pos):
        nonlocal best_pos, best_energy
        if not _is_valid(pos):
            return
        e = _morse_energy(pos, rho)
        if e < best_energy:
            best_energy = e
            best_pos = pos.copy()
        candidates.append((e, pos.copy()))

    # Phase 1: relax each seed
    for idx, seed in enumerate(seeds):
        if time.time() > deadline - 30.0:
            break
        try:
            pos = _local_minimize(seed, rho, deadline)
            consider(pos)
        except Exception:
            pass
            
    # Sort candidates to keep the best ones for BH
    candidates.sort(key=lambda x: x[0])
    # Keep only top 10 to save memory and speed up selection
    if len(candidates) > 10:
        candidates = candidates[:10]

    # Phase 2: Basin Hopping
    perturb_scale = 0.5
    rng = np.random.default_rng(42)
    bh_iter = 0
    max_bh = 300
    
    while time.time() < deadline - 15.0 and bh_iter < max_bh:
        bh_iter += 1
        # Pick starting point
        if best_pos is not None and rng.random() < 0.6:
            cur = best_pos.copy()
        elif candidates:
            cur = candidates[rng.integers(len(candidates))][1].copy()
        else:
            break
            
        # Perturb
        # Adaptive step size: start moderate, shrink if improving, grow if stuck
        noise = rng.normal(0.0, perturb_scale, size=cur.shape)
        perturbed = cur + noise
        
        # Check validity of perturbed state (cheap)
        if not _is_valid(perturbed):
            # Try smaller step
            noise = rng.normal(0.0, perturb_scale * 0.5, size=cur.shape)
            perturbed = cur + noise
            if not _is_valid(perturbed):
                perturb_scale = max(0.1, perturb_scale * 0.9)
                continue

        # Relax
        try:
            newpos = _local_minimize(perturbed, rho, deadline)
        except Exception:
            continue
            
        e = _morse_energy(newpos, rho)
        if _is_valid(newpos):
            if e < best_energy:
                best_energy = e
                best_pos = newpos.copy()
                perturb_scale = max(0.1, perturb_scale * 0.95)
            else:
                # Metropolis-like acceptance or just occasional growth
                if rng.random() < 0.1:
                    perturb_scale = min(1.5, perturb_scale * 1.1)
                else:
                    perturb_scale = max(0.1, perturb_scale * 0.99)

    # Phase 3: Final Polish
    if best_pos is not None and time.time() < deadline - 5.0:
        try:
            polished = _local_minimize(best_pos, rho, deadline)
            e = _morse_energy(polished, rho)
            if _is_valid(polished) and e < best_energy:
                best_energy = e
                best_pos = polished.copy()
        except Exception:
            pass

    if best_pos is None:
        best_pos = _random_compact(n, seed=99)
    return best_pos
# EVOLVE-BLOCK-END
