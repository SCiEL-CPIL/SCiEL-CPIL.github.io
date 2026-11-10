"""Rewritten packing optimizer: rings+hex seed, soft-min L-BFGS polish,
directed-repulsion multi-restart SA, explicit deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize, check_grad

N = 50

# EVOLVE-BLOCK-START

def _min_radius(c):
    """Exact objective (maximize): min(pairwise/2, boundary)."""
    c = np.asarray(c, dtype=np.float64)
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair = dist.min() / 2.0
    bound = 1.0 - np.linalg.norm(c, axis=1).max()
    return min(pair, bound)


def _softmin_penalty(c, lam):
    """Smooth surrogate: positive when constraints violated (minimize)."""
    c = np.asarray(c, dtype=np.float64)
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_v = lam - dist            # >0 if too close
    norms = np.linalg.norm(c, axis=1)
    bound_v = norms + lam - 1.0    # >0 if too close to wall
    M = np.concatenate([pair_v.ravel(), bound_v.ravel()])
    mx = M.max()
    if mx <= 0:
        return 0.0
    s = M - mx
    return mx + np.log(np.sum(np.exp(s)))


def _softmin_grad(c, lam):
    """Gradient of the soft-min penalty (for L-BFGS-B)."""
    c = np.asarray(c, dtype=np.float64)
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    pair_v = lam - dist
    norms = np.linalg.norm(c, axis=1)
    bound_v = norms + lam - 1.0
    M = np.concatenate([pair_v.ravel(), bound_v.ravel()])
    mx = M.max()
    if mx <= 0:
        return np.zeros(2 * N)
    s = M - mx
    w = np.exp(s)
    Z = np.sum(w)
    w = w / Z
    pair_w = w[:N * N].reshape(N, N)
    bound_w = w[N * N:]
    # d pair_v[i,j] / d c[i] = -(c[i]-c[j])/dist[i,j]
    inv = np.zeros((N, N))
    mask = dist > 1e-12
    inv[mask] = 1.0 / dist[mask]
    grad = np.zeros(2 * N)
    for i in range(N):
        g = np.zeros(2)
        for j in range(N):
            if i == j:
                continue
            wv = pair_w[i, j]
            if wv > 0:
                g += -wv * (c[i] - c[j]) * inv[i, j]
        # boundary
        if bound_w[i] > 0:
            g += bound_w[i] * (c[i] / (norms[i] + 1e-300))
        grad[2 * i:2 * i + 2] = g
    return grad


def _lbfgs_polish(c0, lam, maxit=250, tol=1e-9):
    """Gradient-based polish using the smooth soft-min surrogate."""
    c0 = np.asarray(c0, dtype=np.float64)
    x0 = c0.ravel()
    try:
        res = minimize(_softmin_penalty, x0,
                       jac=_softmin_grad,
                       args=(lam,),
                       method='L-BFGS-B',
                       options={'maxiter': maxit, 'ftol': 1e-14,
                                'gtol': 1e-8, 'maxls': 40})
        return res.x.reshape(N, 2)
    except Exception:
        return c0


def _seed_rings():
    """Concentric ring seed: 1 centre + rings of ~2*pi*r."""
    pts = [np.array([0.0, 0.0])]
    r = 0.15
    step = 0.15
    while len(pts) < N:
        n = max(1, int(round(2 * np.pi * r / step)))
        for k in range(n):
            if len(pts) >= N:
                break
            a = 2 * np.pi * k / n
            pts.append(r * np.array([np.cos(a), np.sin(a)]))
        r += step
    return np.array(pts[:N])


def _seed_hex():
    """Hex lattice seed, keep most central N points."""
    spacing = 0.22
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    pts = np.array(rows)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    mask = rad <= 1.0 - spacing / 2.0
    pts = pts[mask]
    rad = rad[mask]
    idx = np.argsort(rad)
    return pts[idx[:N]]


def _seed_random(rmax=0.85):
    pts = np.random.uniform(-rmax, rmax, (N, 2))
    norms = np.linalg.norm(pts, axis=1)
    bad = norms > rmax
    if np.any(bad):
        pts[bad] *= (rmax / norms[bad])[:, None]
    return pts


def _directed_move(c, r):
    """Return a directed move: index + displacement that improves the
    tightest constraint (pairwise or boundary)."""
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    norms = np.linalg.norm(c, axis=1)
    # pairwise tightness: dist - 2r ; boundary tightness: (1 - norm) - r
    # find the point whose worst constraint is smallest (most violated)
    worst_pair = dist.min(axis=1) - 2 * r
    worst_bound = (1.0 - norms) - r
    worst = np.minimum(worst_pair, worst_bound)
    i = int(np.argmin(worst))
    # decide direction
    if worst_bound[i] <= worst_pair[i]:
        # boundary: push inward
        nrm = norms[i] + 1e-300
        disp = -(c[i] / nrm) * 0.5 * r
    else:
        # pairwise: find nearest neighbour
        j = int(np.argmin(dist[i]))
        away = c[i] - c[j]
        an = np.linalg.norm(away) + 1e-300
        disp = (away / an) * 0.5 * r
    return i, disp


def _sa(c0, t_start, budget, r_init):
    """Directed-repulsion simulated annealing within a time budget."""
    c = np.array(c0, dtype=np.float64)
    r = _min_radius(c)
    best = c.copy()
    best_r = r
    T = max(0.02, 0.5 * r)
    alpha = 0.99995
    T_min = 1e-5
    # reheating schedule
    reheat_every = 2000
    it = 0
    while True:
        if time.time() - t_start > budget:
            break
        i, disp = _directed_move(c, r)
        # add noise scaled by T
        noise = np.random.normal(0, T, 2)
        new_c = c.copy()
        new_c[i] = c[i] + disp + noise
        new_r = _min_radius(new_c)
        if new_r >= r:
            c = new_c
            r = new_r
            if r > best_r:
                best_r = r
                best = c.copy()
        else:
            delta = r - new_r
            if np.random.rand() < np.exp(-delta / (T + 1e-12)):
                c = new_c
                r = new_r
        T *= alpha
        if T < T_min:
            T = T_min
        it += 1
        if it % reheat_every == 0:
            # occasional reheat to escape local minima
            T = max(T, 0.5 * r)
    return best, best_r


def construct():
    start = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def remaining():
        return DEADLINE - (time.time() - start)

    # ---- Phase 0: candidate seeds ----
    seeds = [
        _seed_rings(),
        _seed_hex(),
        _seed_random(0.85),
        _seed_random(0.7),
    ]
    # polish each seed a little to get a baseline
    polished = []
    for s in seeds:
        if remaining() < 20.0:
            break
        p = _lbfgs_polish(s, lam=0.05, maxit=80)
        polished.append(p)

    # pick the best polished seed
    best = polished[0]
    best_r = _min_radius(best)
    for p in polished[1:]:
        r = _min_radius(p)
        if r > best_r:
            best_r = r
            best = p

    # ---- Phase 1: multi-restart SA + polish ----
    # Allocate budget: keep >= 12s for final polish.
    n_restarts = 0
    max_restarts = 6
    while remaining() > 25.0 and n_restarts < max_restarts:
        # SA from best (or a fresh seed for diversity)
        if n_restarts % 2 == 0:
            start_c = best
        else:
            start_c = polished[n_restarts % len(polished)]
        sa_budget = min(remaining() - 20.0, 25.0)
        if sa_budget < 3.0:
            break
        cand, cand_r = _sa(start_c, time.time(), sa_budget, best_r)
        # polish the SA result
        if remaining() > 12.0:
            cand = _lbfgs_polish(cand, lam=0.03, maxit=150)
            cand_r = _min_radius(cand)
        if cand_r > best_r:
            best_r = cand_r
            best = cand
        n_restarts += 1

    # ---- Phase 2: final high-precision polish ----
    # Decreasing-lam soft-min polish for tightness.
    for lam in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
        if remaining() < 3.0:
            break
        best = _lbfgs_polish(best, lam=lam, maxit=200, tol=1e-12)
        r = _min_radius(best)
        if r > best_r:
            best_r = r

    # Final directed micro-move sweep (cheap, exact)
    for _ in range(200):
        if remaining() < 2.0:
            break
        i, disp = _directed_move(best, best_r)
        new_c = best.copy()
        new_c[i] = best[i] + disp * 0.5
        new_r = _min_radius(new_c)
        if new_r > best_r:
            best = new_c
            best_r = new_r
        else:
            break

    # ---- Final validation / safety ----
    # Ensure finite and inside (clip if a numeric glitch pushed outside).
    best = np.asarray(best, dtype=np.float64)
    if not np.all(np.isfinite(best)):
        best = _seed_hex()
        best_r = _min_radius(best)
    norms = np.linalg.norm(best, axis=1)
    # hard-clip to radius 1 - tiny to guarantee validity
    tiny = 1e-9
    bad = norms > (1.0 - tiny)
    if np.any(bad):
        best[bad] *= ((1.0 - tiny) / norms[bad])[:, None]

    return best

# EVOLVE-BLOCK-END
