"""Simulated-annealing portfolio solver for Morse cluster global minima."""
# EVOLVE-BLOCK-START
import time
import threading
import numpy as np
import concurrent.futures as _cf
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax

_T0 = None
_HARD_DEADLINE = None
_POOL = None
_POOL_LOCK = threading.Lock()
_DEADLOCK = 250.0          # absolute wall-clock cap shared across all construct calls
_PER_CALL = 40.0           # nominal budget per construct call
_N_THREADS = 4


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + _DEADLOCK
    return _T0


def _morse_E(P, rho):
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    iu = np.triu_indices(n, k=1)
    r = d[iu]
    t = np.exp(rho * (1.0 - r))
    return float((t * (t - 2.0)).sum())


def _morse_grad(P, rho):
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
    return -(rowsum[:, None] * P - W @ P)


def _local_opt(P0, rho, max_iter=300, ftol=1e-11, gtol=1e-8):
    P0 = np.asarray(P0, dtype=float)
    n = P0.shape[0]

    def fun(x):
        return _morse_E(x.reshape(n, 3), rho)

    def gfun(x):
        return _morse_grad(x.reshape(n, 3), rho).ravel()

    try:
        res = minimize(fun, P0.ravel(), jac=gfun, method="L-BFGS-B",
                       options={"maxiter": max_iter, "maxfun": 20000,
                                "ftol": ftol, "gtol": gtol, "iprint": 0})
        P = res.x.reshape(n, 3)
        return _morse_E(P, rho), P
    except Exception:
        return None, None


def _relax(pos, rho):
    """Library relax then BFGS polish; keep the lower-energy exact result."""
    P = np.asarray(pos, dtype=float).reshape(len(pos), 3)
    best_E, best_P = None, None
    try:
        _, P2 = relax(P, "morse", rho)
        P2 = np.asarray(P2, dtype=float).reshape(len(pos), 3)
        best_E, best_P = _morse_E(P2, rho), P2
    except Exception:
        pass
    start = best_P if best_P is not None else P
    E3, P3 = _local_opt(start, rho)
    if E3 is not None and (best_E is None or E3 < best_E):
        best_E, best_P = E3, P3
    if best_E is None:
        E3, P3 = _local_opt(P, rho)
        if E3 is not None:
            best_E, best_P = E3, P3
    return best_E, best_P


def _get_pool():
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = _cf.ThreadPoolExecutor(max_workers=_N_THREADS)
    return _POOL


# ---------- structured seed generators ----------
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


def _shells(n, rng, jitter=0.05):
    phi = (1 + 5 ** 0.5) / 2
    base = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                     [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    pts = [np.zeros(3)]
    for rad in (1.0, 1.6, 2.2, 2.9, 3.6, 4.4, 5.2):
        if len(pts) >= n:
            break
        for _ in range(12):
            if len(pts) >= n:
                break
            pts.append(base[rng.integers(12)] * rad + rng.normal(0, jitter, 3))
        for _ in range(6):
            if len(pts) >= n:
                break
            pts.append(octa[rng.integers(6)] * rad + rng.normal(0, jitter, 3))
    P = np.array(pts[:n])
    while len(P) < n:
        P = np.vstack([P, rng.uniform(-1, 1, 3)])
    return P


def _fcc(n, rng, jitter=0.08):
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


def _bcc(n, rng, jitter=0.08):
    a = 1.0
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                pts.append([i * a, j * a, k * a])
                pts.append([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a])
    pts = np.array(pts)
    order = np.argsort(np.linalg.norm(pts, axis=1))
    pts = pts[order][:n]
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, pts.shape)
    return pts


def _hcp(n, rng, jitter=0.08):
    a = 1.0
    c = np.sqrt(8.0 / 3.0) * a
    pts = []
    for layer in range(3):
        for i in range(-5, 6):
            for j in range(-5, 6):
                x = a * (i + 0.5 * (j % 2)) + (a / 2 if layer == 1 else 0.0)
                y = c / 2 * j
                z = c / 2 * layer
                pts.append([x, y, z])
    pts = np.array(pts)
    order = np.argsort(np.linalg.norm(pts, axis=1))
    pts = pts[order][:n]
    if jitter > 0:
        pts = pts + rng.normal(0, jitter, pts.shape)
    return pts


def _random_pack(n, R, rng, min_d):
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 1200:
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
    for c in (0.85, 1.0, 1.15):
        R = base * c
        seeds.append(_sunflower(n, R, 0.0, rng))
        seeds.append(_sunflower(n, R, 0.12, rng))
    seeds.append(_fcc(n, rng, 0.08))
    seeds.append(_bcc(n, rng, 0.08))
    seeds.append(_hcp(n, rng, 0.08))
    seeds.append(_shells(n, rng, 0.05))
    seeds.append(_random_pack(n, base * 1.15, rng, 0.6))
    seeds.append(_random_pack(n, base * 0.95, rng, 0.55))
    seeds.append(rng.uniform(-base, base, size=(n, 3)))
    seeds.append(_sunflower(n, base * 0.7, 0.15, rng))
    seeds.append(_fcc(n, rng, 0.15))
    seeds.append(_hcp(n, rng, 0.15))
    return seeds


# ---------- simulated annealing (structural search change) ----------
def _kick(P, rng, ks):
    """Random displacement of a random subset of atoms; ks is the scale."""
    P = P.copy()
    m = P.shape[0]
    frac = 0.25 + 0.45 * rng.random()
    k = max(1, int(round(frac * m)))
    idx = rng.choice(m, size=k, replace=False)
    P[idx] += ks * (0.6 + 0.8 * rng.random()) * rng.normal(0.0, 1.0, (k, 3))
    return P


def _anneal(P0, rho, T0, T1, n_iter, deadline, best):
    """Simulated annealing over the cluster; updates shared best on new minima."""
    P = np.asarray(P0, dtype=float).copy()
    E = _morse_E(P, rho)
    best_E = E
    best_P = P.copy()
    for it in range(n_iter):
        if time.time() > deadline:
            break
        T = T0 * (T1 / T0) ** (it / max(1, n_iter - 1))
        Pn = _kick(P, _rng_global, 0.2 + 0.4 * (T / T0))
        En = _morse_E(Pn, rho)
        if En < E or (T > 1e-6 and rng_accept(En - E, T)):
            P, E = Pn, En
            if E < best_E - 1e-12:
                best_E, best_P = E, P.copy()
                # polish the new minimum and update the shared best
                Ep, Pp = _local_opt(P, rho, max_iter=120, ftol=1e-10, gtol=1e-7)
                if Ep is not None and Ep < best_E:
                    best_E, best_P = Ep, Pp
                _update_best(best, best_E, best_P)
                P, E = best_P.copy(), best_E
    return best_E, best_P


def rng_accept(delta, T):
    return np.random.random() < np.exp(-delta / T)


# shared RNG for kicks (annealing is CPU-bound; single shared rng is fine)
_rng_global = np.random.default_rng(20240517)


def _update_best(best, E, P):
    if E is None:
        return
    if best[0] is None or E < best[0] - 1e-12:
        best[0] = E
        best[1] = P.copy()


def _enforce_min_dist(P, min_d=0.5):
    P = np.array(P, dtype=float, copy=True)
    n = P.shape[0]
    for _ in range(200):
        d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
        bad = (d < min_d)
        np.fill_diagonal(bad, False)
        if not bad.any():
            break
        i, j = np.argwhere(bad)[0]
        gap = min_d - d[i, j] + 0.005
        v = P[i] - P[j]
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            v = np.array([1.0, 0.0, 0.0]); nv = 1.0
        v = v / nv
        P[i] += 0.5 * gap * v
        P[j] -= 0.5 * gap * v
    return P


def _min_dist_violation(P, min_d=0.5):
    d = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return float(min_d - d.min()) if d.min() < min_d else 0.0


def construct(instance):
    _init_timer()
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(size * 1000003 + 17)

    call_start = time.time()
    deadline = min(_HARD_DEADLINE - 1.0, call_start + _PER_CALL)
    best = [None, None]
    pool = _get_pool()

    # Phase 1: relax diverse structured seeds in parallel
    seeds = _make_seeds(size, rng)
    futs = [pool.submit(_relax, s, rho) for s in seeds]
    for f in futs:
        try:
            E, P = f.result()
        except Exception:
            E, P = None, None
        _update_best(best, E, P)

    if best[1] is None:
        P = _sunflower(size, size ** (1.0 / 3.0), 0.0, rng)
        best[0], best[1] = _morse_E(P, rho), P

    # Phase 2: parallel simulated annealing from the best + a few alternates
    T0, T1, n_iter = 3.0, 0.02, 2000
    start_points = [best[1].copy()]
    for s in seeds[:3]:
        start_points.append(s)
    # anneal the current best in parallel with alternate starts
    anneal_futs = [pool.submit(_anneal, sp, rho, T0, T1, n_iter, deadline, best)
                   for sp in start_points[:_N_THREADS]]
    for f in anneal_futs:
        try:
            f.result()
        except Exception:
            pass

    # Phase 3: a few more anneal rounds from the current best if time remains
    while time.time() < deadline - 0.5:
        T0, T1, n_iter = 1.5, 0.02, 1500
        sp = best[1].copy()
        f = pool.submit(_anneal, sp, rho, T0, T1, n_iter, deadline, best)
        try:
            f.result()
        except Exception:
            pass

    # Phase 4: final polish of the best minimum
    if best[1] is not None and time.time() < _HARD_DEADLINE - 1.0:
        E, P = _local_opt(best[1], rho, max_iter=600, ftol=1e-11, gtol=1e-8)
        if E is not None and E < best[0] - 1e-10:
            best[0], best[1] = E, P

    # Final: enforce min distance only if violated
    if best[1] is not None and _min_dist_violation(best[1], 0.5) > 0:
        P = _enforce_min_dist(best[1], 0.5)
        E_new = _morse_E(P, rho)
        if E_new < best[0] + 0.01:
            best[1] = P
    return best[1]
# EVOLVE-BLOCK-END
