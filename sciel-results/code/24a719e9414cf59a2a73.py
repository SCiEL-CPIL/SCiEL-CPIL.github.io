"""
Strategy: Multi-start L-BFGS with CEC acceptance + explicit error_repair.
- Parallel (sequential-but-budgeted) L-BFGS starts from diverse seeds.
- CEC (concurrent evolution control): accept a new local min if its energy
  beats the running best OR if it is "different enough" (diversity pressure).
- error_repair: when a configuration has a pair < 0.5 or a near-zero distance,
  run a short projected gradient descent that hard-enforces the min-distance
  constraint via a penalty + projection step, then re-polish with L-BFGS.
- Vectorized energy/gradient; tight inner loop; explicit wall-clock guard.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START = time.time()
_LIMIT = 322.0  # 330s kill, 8s safety margin

def _energy_grad(pos, rho):
    """Vectorized Morse energy and gradient. Returns (E, g) where g is (n,3)."""
    n = pos.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(pos)
    # pair differences
    diff = pos[:, None, :] - pos[None, :, :]          # (n,n,3)
    r = np.linalg.norm(diff, axis=2)                   # (n,n)
    np.fill_diagonal(r, 1e300)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    E = 0.5 * np.sum(ex * (ex - 2.0))
    # dV/dr = -rho * 2 * ex * (ex - 1)
    dVdr = -2.0 * rho * ex * (ex - 1.0)
    inv_r = np.zeros_like(r)
    mask = r > 1e-12
    inv_r[mask] = 1.0 / r[mask]
    F = dVdr * inv_r                                    # (n,n) force-magnitude factor
    # grad_i = -sum_j F_ij * (pos_i - pos_j) / r_ij  ... use standard form
    # dE/dpos_i = sum_j  (dV/dr_ij) * (pos_i - pos_j)/r_ij
    # = sum_j dVdr_ij * diff_ij / r_ij
    # diff_ij = pos_i - pos_j
    grad = np.sum(F[:, :, None] * diff, axis=1)         # (n,3)
    return float(E), grad

def _check_min_dist(pos, min_d=0.5):
    """Return (ok, min_pair_dist)."""
    n = pos.shape[0]
    if n < 2:
        return True, 1e30
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, 1e30)
    m = r.min()
    return (m >= min_d - 1e-9), float(m)

def _local_min(pos, rho, maxiter=400, ftol=1e-13, gtol=1e-10):
    """L-BFGS polish. Returns (pos, E)."""
    def fun(x):
        p = x.reshape(-1, 3)
        e, g = _energy_grad(p, rho)
        return e, g.ravel()
    try:
        res = minimize(fun, pos.ravel(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                                'maxls': 20})
        return res.x.reshape(-1, 3), float(res.fun)
    except Exception:
        e, _ = _energy_grad(pos, rho)
        return pos, e

def _error_repair(pos, rho, min_d=0.5, max_iter=300, step=0.05):
    """
    error_repair: projected gradient descent with a hard min-distance penalty.
    Used when a configuration is invalid (pair < min_d) or has a near-singular
    geometry. Returns (pos_repaired, E).
    """
    pos = pos.copy()
    n = pos.shape[0]
    # penalty weight ramping up
    for it in range(max_iter):
        ok, md = _check_min_dist(pos, min_d)
        if ok and md > min_d + 1e-3:
            break
        e, g = _energy_grad(pos, rho)
        # penalty gradient: push apart any pair < min_d + margin
        margin = 0.05
        diff = pos[:, None, :] - pos[None, :, :]
        r = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(r, 1e30)
        bad = r < (min_d + margin)
        if np.any(bad):
            inv_r = np.zeros_like(r)
            m2 = r > 1e-12
            inv_r[m2] = 1.0 / r[m2]
            # force magnitude for penalty: w * (min_d+margin - r) for bad pairs
            w = 50.0
            pen_mag = np.where(bad, w * np.maximum(0.0, (min_d + margin) - r), 0.0)
            np.fill_diagonal(pen_mag, 0.0)
            pen_grad = np.sum(pen_mag[:, :, None] * diff, axis=1)
            g = g + pen_grad
        # gradient step with backtracking
        step_size = step
        for _ in range(10):
            new_pos = pos - step_size * g
            new_ok, new_md = _check_min_dist(new_pos, min_d - 1e-6)
            new_e, _ = _energy_grad(new_pos, rho)
            if new_ok and new_e < e + 1e-12:
                pos = new_pos
                e = new_e
                break
            step_size *= 0.5
        else:
            break
    # final polish
    pos, e = _local_min(pos, rho, maxiter=200, ftol=1e-12, gtol=1e-9)
    return pos, e

def _seed_fcc(n, rng):
    a = np.sqrt(2.0)
    pts = []
    M = int(np.ceil(n ** (1/3) * 1.6)) + 2
    for i in range(-M, M+1):
        for j in range(-M, M+1):
            for k in range(-M, M+1):
                base = np.array([i, j, k]) * a
                for dx,dy,dz in [(0,0,0),(0,0.5*a,0.5*a),(0.5*a,0,0.5*a),(0.5*a,0.5*a,0)]:
                    pts.append(base + np.array([dx,dy,dz]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts, axis=1)
    idx = np.argsort(d)
    return pts[idx[:n]]

def _seed_bcc(n, rng):
    a = 2.0 / np.sqrt(3.0)
    pts = []
    M = int(np.ceil(n ** (1/3) * 1.6)) + 2
    for i in range(-M, M+1):
        for j in range(-M, M+1):
            for k in range(-M, M+1):
                base = np.array([i, j, k]) * a
                for dx,dy,dz in [(0,0,0),(0.5*a,0.5*a,0.5*a)]:
                    pts.append(base + np.array([dx,dy,dz]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts, axis=1)
    idx = np.argsort(d)
    return pts[idx[:n]]

def _seed_hcp(n, rng):
    a = 2.0
    c = a * np.sqrt(8.0/3.0)
    pts = []
    M = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-M, M+1):
        for j in range(-M, M+1):
            for k in range(-M, M+1):
                # AB stacking
                if (i + j + k) % 2 == 0:
                    base = np.array([i, j, 0]) * a + np.array([0,0,k*c])
                    pts.append(base)
                    pts.append(base + np.array([a/2, a*np.sqrt(3)/2, 0]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    d = np.linalg.norm(pts, axis=1)
    idx = np.argsort(d)
    return pts[idx[:n]]

def _seed_icosahedral(n, rng):
    pts = [np.zeros(3)]
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [1,phi,0],[-1,phi,0],[1,-phi,0],[-1,-phi,0],
        [0,1,phi],[0,-1,phi],[0,1,-phi],[0,-1,-phi],
        [phi,0,1],[-phi,0,1],[phi,0,-1],[-phi,0,-1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    shells = [1.0, 1.7, 2.4, 3.1, 3.8, 4.5, 5.2]
    for R in shells:
        if len(pts) >= n: break
        for v in ico:
            if len(pts) >= n: break
            p = v * R
            if all(np.linalg.norm(p - q) >= 0.5 for q in pts):
                pts.append(p)
        # fill shell randomly
        while len(pts) < n:
            th = rng.uniform(0, 2*np.pi)
            ct = rng.uniform(-1, 1)
            st = np.sqrt(1-ct*ct)
            p = np.array([R*st*np.cos(th), R*st*np.sin(th), R*ct])
            if all(np.linalg.norm(p - q) >= 0.5 for q in pts):
                pts.append(p)
            else:
                break
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts

def _seed_random(n, rng):
    R = 1.2 * n ** (1/3) * 1.6
    pts = []
    for _ in range(n * 800):
        if len(pts) >= n: break
        p = rng.uniform(-R, R, 3)
        if all(np.linalg.norm(p - q) >= 0.55 for q in pts):
            pts.append(p)
    while len(pts) < n:
        pts.append(rng.uniform(-R, R, 3))
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts

def construct(instance):
    global _START, _LIMIT
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    rng = np.random.default_rng(1234)

    # seed pool
    seeds = [
        _seed_fcc(n, rng),
        _seed_bcc(n, rng),
        _seed_hcp(n, rng),
        _seed_icosahedral(n, rng),
        _seed_random(n, rng),
        _seed_random(n, rng),
    ]

    # CEC state
    best_pos = None
    best_E = float('inf')
    # diversity pool: keep a few distinct local minima
    pool = []  # list of (E, pos)

    # time budget per start: ~45s total for 6 instances -> ~7.5s each
    # but we get 322s / 6 = ~53s per instance. Use ~48s.
    t_start = time.time()
    per_instance_budget = 48.0

    # Phase A: polish each seed, then error_repair if invalid, then add to pool
    for s_idx, s in enumerate(seeds):
        if time.time() - t_start > per_instance_budget:
            break
        # polish
        pos, E = _local_min(s, rho, maxiter=300)
        # check validity; if invalid, error_repair
        ok, md = _check_min_dist(pos, 0.5)
        if not ok:
            pos, E = _error_repair(pos, rho)
            ok, md = _check_min_dist(pos, 0.5)
        if not ok:
            # still invalid: try a fresh random seed and repair
            s2 = _seed_random(n, rng)
            pos2, E2 = _local_min(s2, rho, maxiter=200)
            ok2, _ = _check_min_dist(pos2, 0.5)
            if not ok2:
                pos2, E2 = _error_repair(pos2, rho)
            pos, E = pos2, E2
        if E < best_E:
            best_E = E
            best_pos = pos.copy()
        # add to pool if distinct (simple: energy differs by > 1e-6)
        distinct = True
        for (pE, pP) in pool:
            if abs(pE - E) < 1e-6:
                distinct = False
                break
        if distinct and len(pool) < 12:
            pool.append((E, pos.copy()))

    # Phase B: CEC-style iterative improvement
    # For a fixed number of iterations (or until time), pick a parent from pool,
    # perturb, polish, error_repair if needed, accept if better or diverse.
    n_iter = 60
    for it in range(n_iter):
        if time.time() - t_start > per_instance_budget - 5.0:
            break
        # parent selection: 70% best, 30% random from pool
        if pool:
            if rng.random() < 0.7 and best_pos is not None:
                parent = best_pos
            else:
                parent = pool[rng.integers(len(pool))][1]
        else:
            parent = _seed_random(n, rng)

        # perturbation: Gaussian with scale depending on n
        scale = 0.3 + 0.2 * (n / 80.0)
        child = parent + rng.normal(0, scale, parent.shape)

        # polish
        child, cE = _local_min(child, rho, maxiter=250)
        ok, md = _check_min_dist(child, 0.5)
        if not ok:
            # error_repair
            child, cE = _error_repair(child, rho)
            ok, md = _check_min_dist(child, 0.5)
        if not ok:
            continue

        # acceptance: better, or diverse (energy gap > threshold and not in pool)
        if cE < best_E:
            best_E = cE
            best_pos = child.copy()
            pool.append((cE, child.copy()))
        else:
            # diversity pressure: accept if not too similar to existing pool
            diverse = True
            for (pE, pP) in pool:
                if abs(pE - cE) < 1e-4:
                    diverse = False
                    break
            if diverse and len(pool) < 15:
                pool.append((cE, child.copy()))

    # Phase C: final high-precision polish on best
    if best_pos is not None and time.time() - t_start < per_instance_budget - 1.0:
        final, fE = _local_min(best_pos, rho, maxiter=1500, ftol=1e-15, gtol=1e-12)
        ok, _ = _check_min_dist(final, 0.5)
        if ok and fE < best_E:
            best_E = fE
            best_pos = final

    # safety: if best_pos is None or invalid, return a repaired random
    if best_pos is None:
        best_pos = _seed_random(n, rng)
        best_pos, best_E = _error_repair(best_pos, rho)

    ok, _ = _check_min_dist(best_pos, 0.5)
    if not ok:
        # last-resort repair
        best_pos, best_E = _error_repair(best_pos, rho)

    return best_pos
# EVOLVE-BLOCK-END
