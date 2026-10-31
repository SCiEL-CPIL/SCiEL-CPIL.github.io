"""Improved program: start from good symmetric configs and refine with smooth optimization."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

DEADLINE = 110.0  # safety margin before 120s limit


def _pairwise_dists_sq(P):
    """Return matrix of squared pairwise distances, shape (n, n)."""
    diff = P[:, None, :] - P[None, :, :]
    return np.sum(diff * diff, axis=-1)


def _score(P):
    """(max pairwise dist / min pairwise dist)^2 over distinct pairs."""
    D2 = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, k=1)
    vals = D2[iu]
    vmin = vals.min()
    vmax = vals.max()
    if vmin <= 1e-15:
        return 1e12
    return (vmax / vmin)


def _smooth_objective(P, beta=50.0):
    """
    Smooth surrogate: encourage all pairwise distances to be close to 1.
    Uses a log-sum-exp style soft max/min on squared distances.
    We minimize (soft_max(D2) - soft_min(D2)) / soft_min(D2) style,
    but simpler: minimize variance-like term via log-sum-exp of D2 and log-sum-exp of -D2.
    """
    D2 = _pairwise_dists_sq(P)
    iu = np.triu_indices(N, k=1)
    vals = D2[iu]
    # soft max of vals
    m = vals.max()
    logsumexp_max = m + np.log(np.sum(np.exp(vals - m)))
    # soft min of vals = -soft max of -vals
    mn = vals.min()
    logsumexp_min_neg = -mn + np.log(np.sum(np.exp(-vals + mn)))
    soft_min = -logsumexp_min_neg
    # ratio surrogate
    if soft_min <= 1e-8:
        return 1e6
    return (logsumexp_max - soft_min) / soft_min


def _initial_grid():
    """4x4 regular grid, centered."""
    pts = []
    for i in range(4):
        for j in range(4):
            pts.append([i - 1.5, j - 1.5])
    return np.array(pts, dtype=float)


def _initial_circle():
    """16 points: 8 on outer circle radius 2, 8 on inner circle radius 1, alternating angles."""
    pts = []
    for k in range(8):
        ang = 2 * np.pi * k / 8
        pts.append([2 * np.cos(ang), 2 * np.sin(ang)])
    for k in range(8):
        ang = 2 * np.pi * (k + 0.5) / 8
        pts.append([1 * np.cos(ang), 1 * np.sin(ang)])
    return np.array(pts, dtype=float)


def _initial_hexagonal():
    """Approximate hexagonal packing for 16 points."""
    pts = []
    # 4 rows: 4,4,4,4 shifted
    for row in range(4):
        y = row - 1.5
        xoff = 0.0 if row % 2 == 0 else 0.5
        for col in range(4):
            x = col - 1.5 + xoff
            pts.append([x, y])
    return np.array(pts, dtype=float)


def _refine(P0, time_budget=30.0):
    """Run L-BFGS-B on smooth objective from P0."""
    start = time.time()
    try:
        res = minimize(
            _smooth_objective,
            P0.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50},
        )
        P_opt = res.x.reshape(N, D)
    except Exception:
        P_opt = P0
    elapsed = time.time() - start
    return P_opt, elapsed


def _polish(P0, time_budget=20.0):
    """Final high-precision polish: minimize the actual (nonsmooth) ratio via many small steps."""
    start = time.time()
    P = P0.copy()
    lr = 0.01
    for it in range(2000):
        if time.time() - start > time_budget:
            break
        D2 = _pairwise_dists_sq(P)
        iu = np.triu_indices(N, k=1)
        vals = D2[iu]
        vmin = vals.min()
        vmax = vals.max()
        if vmin <= 1e-12:
            break
        ratio = vmax / vmin
        # Gradient of ratio w.r.t. P: use subgradient of max and min
        # For max: gradient points to increase the largest distance
        # For min: gradient points to decrease the smallest distance
        # Objective f = vmax/vmin
        # df/dP = (1/vmin) * d(vmax)/dP - (vmax/vmin^2) * d(vmin)/dP
        # d(vmax)/dP = 2*(Pi - Pj) for the pair achieving max
        # d(vmin)/dP = 2*(Pi - Pj) for the pair achieving min
        imax = np.unravel_index(vals.argmax(), (N, N))
        imin = np.unravel_index(vals.argmin(), (N, N))
        # Ensure we get the correct pair (triu indices)
        if imax[0] > imax[1]:
            imax = (imax[1], imax[0])
        if imin[0] > imin[1]:
            imin = (imin[1], imin[0])
        dmax = np.zeros_like(P)
        dmax[imax[0]] += 2 * (P[imax[0]] - P[imax[1]])
        dmax[imax[1]] += 2 * (P[imax[1]] - P[imax[0]])
        dmin = np.zeros_like(P)
        dmin[imin[0]] += 2 * (P[imin[0]] - P[imin[1]])
        dmin[imin[1]] += 2 * (P[imin[1]] - P[imin[0]])
        grad = dmax / vmin - (vmax / vmin**2) * dmin
        # Normalize gradient
        gn = np.linalg.norm(grad)
        if gn > 1e-15:
            grad = grad / gn
        P -= lr * grad
        # Adaptive learning rate
        if ratio < 1.001:
            lr *= 0.95
    return P


def construct():
    t_start = time.time()
    best_P = _initial_grid()
    best_score = _score(best_P)

    candidates = [_initial_grid(), _initial_circle(), _initial_hexagonal()]

    for i, P0 in enumerate(candidates):
        if time.time() - t_start > DEADLINE - 15:
            break
        time_left = DEADLINE - 15 - (time.time() - t_start)
        budget = max(5.0, min(25.0, time_left / 2))
        P_refined, elapsed = _refine(P0, time_budget=budget)
        s = _score(P_refined)
        if s < best_score:
            best_score = s
            best_P = P_refined.copy()

    # Final polish
    if time.time() - t_start < DEADLINE - 5:
        polish_budget = min(20.0, DEADLINE - 5 - (time.time() - t_start))
        P_polished = _polish(best_P, time_budget=polish_budget)
        s_polished = _score(P_polished)
        if s_polished < best_score:
            best_P = P_polished
            best_score = s_polished

    # Ensure all points are distinct
    D2 = _pairwise_dists_sq(best_P)
    iu = np.triu_indices(N, k=1)
    if D2[iu].min() < 1e-10:
        # Perturb slightly
        np.random.seed(42)
        best_P += np.random.randn(N, D) * 1e-5

    return best_P
# EVOLVE-BLOCK-END
