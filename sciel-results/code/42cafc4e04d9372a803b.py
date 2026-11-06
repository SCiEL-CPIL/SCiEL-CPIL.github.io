"""Search-portfolio solver for Morse cluster global minima (rewritten with Dual Annealing and Basin Hopping)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import concurrent.futures as _cf
from scipy.optimize import minimize, dual_annealing
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_POOL = None
_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 285.0  # Leave 15s buffer for 300s limit
    return _T0


def _morse_E(P, rho):
    """Exact energy."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if n < 2:
        return 0.0
    D = P[:, None, :] - P[None, :, :]
    r = np.sqrt((D * D).sum(axis=2))
    np.fill_diagonal(r, np.inf)
    t = np.exp(rho * (1.0 - r))
    return float((t * (t - 2.0)).sum() / 2.0)


def _morse_grad(P, rho):
    """Gradient of Morse energy."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if n < 2:
        return np.zeros_like(P)
    D = P[:, None, :] - P[None, :, :]
    r = np.sqrt((D * D).sum(axis=2))
    np.fill_diagonal(r, np.inf)
    t = np.exp(rho * (1.0 - r))
    c = rho * t * (2.0 * t - 2.0)
    inv_r = np.where(r > 1e-12, 1.0 / r, 0.0)
    W = c * inv_r
    rowsum = W.sum(axis=1)
    grad = -(rowsum[:, None] * P - W @ P)
    return grad


def _enforce_min_dist(P, min_d=0.5):
    """Simple enforcement of minimum distance."""
    P = np.array(P, dtype=float, copy=True)
    n = P.shape[0]
    for _ in range(100):
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        bad = (d < min_d)
        np.fill_diagonal(bad, False)
        if not bad.any():
            break
        idx = np.argwhere(bad)
        for i, j in idx[:10]:
            gap = min_d - d[i, j] + 0.01
            v = P[i] - P[j]
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                v = np.array([1.0, 0.0, 0.0]); nv = 1.0
            v /= nv
            P[i] += 0.5 * gap * v
            P[j] -= 0.5 * gap * v
    return P


def _local_opt(P0, rho, max_iter=1000, tol=1e-10):
    """Local optimization using L-BFGS-B."""
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
                       options={"maxiter": max_iter, "maxfun": 50000,
                                "ftol": tol, "gtol": 1e-8, "iprint": 0})
        P = res.x.reshape(n, 3)
        E = _morse_E(P, rho)
        return E, P
    except Exception:
        return None, None


def _relax_with_lib(P, rho):
    """Try using the provided relax function first."""
    try:
        P2 = relax(P, "morse", rho)
        P2 = np.asarray(P2, dtype=float).reshape(len(P), 3)
        E2 = _morse_E(P2, rho)
        return E2, P2
    except Exception:
        return None, None


def _hybrid_relax(P, rho):
    """Combine library relax and L-BFGS-B."""
    best_E, best_P = None, None
    
    # Try library relax
    E1, P1 = _relax_with_lib(P, rho)
    if E1 is not None:
        best_E, best_P = E1, P1
    
    # Try L-BFGS-B from original and from library result
    E2, P2 = _local_opt(P, rho)
    if E2 is not None and (best_E is None or E2 < best_E):
        best_E, best_P = E2, P2
        
    if P1 is not None:
        E3, P3 = _local_opt(P1, rho)
        if E3 is not None and (best_E is None or E3 < best_E):
            best_E, best_P = E3, P3
            
    if best_E is None:
        # Fallback
        E4, P4 = _local_opt(P, rho, max_iter=200)
        if E4 is not None:
            best_E, best_P = E4, P4
            
    return best_E, best_P


def _get_pool():
    global _POOL
    if _POOL is None:
        _POOL = _cf.ThreadPoolExecutor(max_workers=4)
    return _POOL


def _fcc(n, rng, jitter=0.0):
    a = 1.0
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                pts.append([i * a, j * a, k * a])
                pts.append([(i + 0.5) * a, (j + 0.5) * a, k * a])
                pts.append([(i + 0.5) * a, j * a, (k + 0.5) * a])
                pts.append([i * a, (j + 0.5) * a, (k + 0.5) * a])
    pts = np.array(pts)
    order = np.argsort(np.linalg.norm(pts, axis=1))
    pts = pts[order][:n]
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, pts.shape)
    return pts

def _bcc(n, rng, jitter=0.0):
    a = 1.0
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                pts.append([i * a, j * a, k * a])
                pts.append([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
    pts = np.array(pts)
    if len(pts) > 1:
        keep = [0]
        for idx in range(1, len(pts)):
            if np.all(np.linalg.norm(pts[idx] - pts[keep], axis=1) > 1e-6):
                keep.append(idx)
        pts = pts[keep]
    order = np.argsort(np.linalg.norm(pts, axis=1))
    pts = pts[order][:n]
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, pts.shape)
    return pts

def _hcp(n, rng, jitter=0.0):
    a = 1.0
    c = np.sqrt(8.0 / 3.0) * a
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = a * (i + 0.5 * (j % 2))
            y = c / 2 * j
            z = 0.0
            pts.append([x, y, z])
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = a * (i + 0.5 * (j % 2)) + a / 2
            y = c / 2 * j
            z = c / 2
            pts.append([x, y, z])
    for i in range(-5, 6):
        for j in range(-5, 6):
            x = a * (i + 0.5 * (j % 2))
            y = c / 2 * j
            z = c
            pts.append([x, y, z])
    pts = np.array(pts)
    order = np.argsort(np.linalg.norm(pts, axis=1))
    pts = pts[order][:n]
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, pts.shape)
    return pts

def _sunflower(n, R, jitter, rng):
    i = np.arange(n)
    t = (i + 0.5) / n
    r = R * t ** (1.0 / 3.0)
    z = 1.0 - 2.0 * t
    th = 2.0 * np.pi * i * 1.6180339887498949
    xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    P = np.column_stack([r * xy * np.cos(th), r * xy * np.sin(th), r * z])
    if jitter > 0:
        P = P + rng.normal(0.0, jitter, P.shape)
    return P

def _random_pack(n, R, rng, min_d):
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 2000:
        tries += 1
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) > R:
            continue
        if all(np.linalg.norm(p - q) >= min_d for q in pts):
            pts.append(p)
    while len(pts) < n:
        pts.append(rng.uniform(-R, R, 3))
    return np.array(pts)

def _make_seeds(n, rng):
    base = n ** (1.0 / 3.0)
    seeds = []
    seeds.append(_fcc(n, rng, 0.0))
    seeds.append(_bcc(n, rng, 0.0))
    seeds.append(_hcp(n, rng, 0.0))
    seeds.append(_fcc(n, rng, 0.1))
    seeds.append(_bcc(n, rng, 0.1))
    seeds.append(_hcp(n, rng, 0.1))
    seeds.append(_sunflower(n, base * 1.0, 0.1, rng))
    seeds.append(_random_pack(n, base * 1.2, rng, 0.6))
    seeds.append(rng.uniform(-base, base, size=(n, 3)))
    return seeds

def _dual_anneal_worker(n, rho, bounds, seed, deadline):
    """Run dual_annealing with a fixed seed."""
    if time.time() > deadline:
        return None, None
    try:
        res = dual_annealing(
            lambda x: _morse_E(x.reshape(n, 3), rho),
            bounds,
            seed=seed,
            maxiter=100,
            nmax=10,
            workers=1,
            initial_temp=10.0,
            restart_temp_ratio=1e-4,
            visit=2.62,
            clip=2.0,
            disp=False
        )
        if time.time() > deadline:
            return None, None
        P = res.x.reshape(n, 3)
        E = _morse_E(P, rho)
        # Polish
        E, P = _hybrid_relax(P, rho)
        return E, P
    except Exception:
        return None, None

def _basin_hopping_worker(n, rho, start_P, deadline, rng_seed):
    """Run a basin hopping chain."""
    if time.time() > deadline:
        return None, None
    rng = np.random.default_rng(rng_seed)
    P = start_P.copy()
    best_E, best_P = _hybrid_relax(P, rho)
    kick_scale = 0.5
    for i in range(50):
        if time.time() > deadline:
            break
        kick = rng.normal(0.0, kick_scale, P.shape)
        P_new = P + kick
        E_new, P_new_relaxed = _hybrid_relax(P_new, rho)
        if E_new is not None:
            if best_E is None or E_new < best_E:
                best_E, best_P = E_new, P_new_relaxed
                P = P_new_relaxed.copy()
                kick_scale *= 0.9 # Cool down
            else:
                kick_scale = min(kick_scale * 1.1, 1.5) # Heat up slightly
    return best_E, best_P

def construct(instance):
    _init_timer()
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(size * 1000003 + 17)

    call_start = time.time()
    # Allocate 45s per instance, leaving 15s total buffer for 6 instances
    deadline = min(_HARD_DEADLINE, call_start + 45.0)
    
    best_E, best_P = None, None
    pool = _get_pool()
    
    # Phase 1: Initial seeds relaxation
    seeds = _make_seeds(size, rng)
    futs = [pool.submit(_hybrid_relax, s, rho) for s in seeds]
    for f in _cf.as_completed(futs):
        try:
            E, P = f.result()
            if E is not None and (best_E is None or E < best_E):
                best_E, best_P = E, P
        except Exception:
            pass

    if best_P is None:
        P = _fcc(size, rng, 0.0)
        best_E, best_P = _hybrid_relax(P, rho)

    # Phase 2: Global Search (Dual Annealing + Basin Hopping)
    # Bounds for dual annealing
    R = size ** (1.0/3.0) * 1.5
    bounds = [(-R, R)] * (size * 3)
    
    # Submit Dual Annealing tasks
    da_futs = []
    for i in range(4):
        seed = rng.integers(0, 100000)
        da_futs.append(pool.submit(_dual_anneal_worker, size, rho, bounds, seed, deadline))
        
    # Submit Basin Hopping tasks from best seeds
    bh_futs = []
    for i in range(4):
        seed = rng.integers(0, 100000)
        start = best_P.copy() if best_P is not None else _fcc(size, rng, 0.1)
        # Perturb start slightly for diversity
        start = start + rng.normal(0, 0.2, start.shape)
        bh_futs.append(pool.submit(_basin_hopping_worker, size, rho, start, deadline, seed))
        
    all_futs = da_futs + bh_futs
    
    for f in _cf.as_completed(all_futs):
        if time.time() > deadline - 1.0:
            break
        try:
            E, P = f.result()
            if E is not None and (best_E is None or E < best_E):
                best_E, best_P = E, P
        except Exception:
            pass

    # Phase 3: Final Polish
    if time.time() < deadline - 0.5 and best_P is not None:
        E, P = _local_opt(best_P, rho, max_iter=2000, tol=1e-12)
        if E is not None and E < best_E:
            best_E, best_P = E, P
            
    P = _enforce_min_dist(best_P, 0.5)
    return P
# EVOLVE-BLOCK-END
