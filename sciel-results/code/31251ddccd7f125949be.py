"""Heilbronn-11: Simulated annealing global search + L-BFGS-B polish."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4

DEADLINE = 142.0

# Precompute all triplets
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])


def _min_area(pts):
    """Min triangle area for N points in Cartesian coords."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return np.min(np.abs(cross)) * 0.5


def _all_areas(pts):
    """All triangle areas."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return np.abs(cross) * 0.5


def _project(pts):
    """Project points into the triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1.0 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _is_valid(pts, tol=1e-9):
    for i in range(len(pts)):
        x, y = pts[i]
        if y < -tol or y > S3H + tol:
            return False
        x_min = y / S3
        x_max = 1.0 - y / S3
        if x < x_min - tol or x > x_max + tol:
            return False
    return True


def _soft_min(pts, beta):
    """Soft minimum of all triangle areas."""
    areas = _all_areas(pts)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return max_a - log_sum / beta


def _symmetric_starts():
    """Generate multiple symmetric initial configurations."""
    starts = []

    # Configuration 1: corners + midpoints + centroid + edge points
    c1 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [1/6, 0.0],
        [5/6, 0.0],
        [0.5, S3H/3],
        [1/3, S3H/6],
    ])
    starts.append(_project(c1))

    # Configuration 2: hexagonal-like packing
    # 3 corners, 3 edge midpoints, 3 inner ring, 1 center
    r = 1/6
    c2 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [0.5, S3H/3],
        [0.5 + r*S3, S3H/3],
        [0.5 - r*S3, S3H/3],
        [1/3, S3H/3 + r*S3H],
        [2/3, S3H/3 + r*S3H],
    ])
    starts.append(_project(c2))

    # Configuration 3: based on known best 11-point arrangement
    # Use a grid-like structure
    c3 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.167, S3H/6],
        [0.833, S3H/6],
        [0.5, 2*S3H/3],
        [0.333, S3H/3],
        [0.667, S3H/3],
    ])
    starts.append(_project(c3))

    # Configuration 4: more spread out
    c4 = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.25, 0.0],
        [0.75, 0.0],
        [0.5, S3H/2],
        [0.125, S3H/4],
        [0.875, S3H/4],
        [0.5, 3*S3H/4],
        [0.375, S3H/4],
        [0.625, S3H/4],
    ])
    starts.append(_project(c4))

    return starts


def _sa_step(pts, rng, step_size, beta):
    """One simulated annealing step: perturb and accept/reject."""
    # Randomly perturb a random subset of points
    n_perturb = rng.integers(1, 4)
    idx = rng.choice(N, size=n_perturb, replace=False)
    new_pts = pts.copy()
    for i in idx:
        dx = rng.normal(0, step_size)
        dy = rng.normal(0, step_size)
        new_pts[i, 0] += dx
        new_pts[i, 1] += dy
    new_pts = _project(new_pts)

    # Compute objective
    old_area = _min_area(pts)
    new_area = _min_area(new_pts)

    # Accept if better or with Boltzmann probability
    if new_area >= old_area:
        return new_pts
    delta = new_area - old_area  # negative
    prob = np.exp(delta / beta)
    if rng.random() < prob:
        return new_pts
    return pts


def _sa_search(pts_init, rng, t0, t1, n_iter, deadline):
    """Simulated annealing search."""
    pts = pts_init.copy()
    best_pts = pts.copy()
    best_area = _min_area(pts)

    for i in range(n_iter):
        if time.time() > deadline:
            break

        # Linear cooling
        t = t0 + (t1 - t0) * (i / max(1, n_iter - 1))
        step_size = max(0.001, 0.05 * t / t0)

        pts = _sa_step(pts, rng, step_size, t)

        area = _min_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    return best_pts, best_area


def _lbfgs_polish(pts, deadline, maxiter=100):
    """L-BFGS-B polish using soft-min objective."""
    def objective(x):
        p = x.reshape(N, 2)
        p = _project(p)
        areas = _all_areas(p)
        # Use a penalty for being outside
        penalty = 0.0
        for j in range(N):
            xj, yj = p[j]
            if yj < 0 or yj > S3H:
                penalty += 100 * abs(yj - max(0, min(S3H, yj)))
            xmin = yj / S3
            xmax = 1 - yj / S3
            if xj < xmin or xj > xmax:
                penalty += 100 * abs(xj - max(xmin, min(xmax, xj)))
        # Maximize min area => minimize -min area
        # Use soft min for smoothness
        beta = 500.0
        max_a = np.max(areas)
        shifted = -beta * (areas - max_a)
        log_sum = np.log(np.sum(np.exp(shifted)))
        soft_min = max_a - log_sum / beta
        return -soft_min + penalty

    x0 = pts.flatten()
    try:
        result = minimize(objective, x0, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12})
        polished = _project(result.x.reshape(N, 2))
        return polished
    except Exception:
        return pts


def construct():
    start_time = time.time()
    rng = np.random.default_rng(2024)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin=5):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid(pts):
            return
        area = _min_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Phase 1: SA from multiple symmetric starts
    starts = _symmetric_starts()

    # Add a few random starts
    for _ in range(3):
        rpts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            x = v + 0.5 * (1 - u - v)
            y = (1 - u - v) * S3H
            rpts[i] = [x, y]
        starts.append(_project(rpts))

    # SA parameters
    t0 = 0.1
    t1 = 0.001
    n_sa_iter = 2000

    for si, spts in enumerate(starts):
        if not _elapsed_ok(15):
            break

        _update_best(spts)

        # Run SA
        sa_best, sa_area = _sa_search(spts, rng, t0, t1, n_sa_iter, DEADLINE - 30)
        _update_best(sa_best)

        # L-BFGS polish
        if _elapsed_ok(10):
            polished = _lbfgs_polish(sa_best, DEADLINE - 10)
            _update_best(polished)

        # If improved significantly, do another SA round from the polished point
        if best_area > 0.01 and _elapsed_ok(20):
            sa_best2, _ = _sa_search(polished, rng, t0 * 0.5, t1, n_sa_iter // 2, DEADLINE - 20)
            _update_best(sa_best2)
            if _elapsed_ok(10):
                polished2 = _lbfgs_polish(sa_best2, DEADLINE - 10)
                _update_best(polished2)

    # Phase 2: Final refinement with higher beta
    if best_pts is not None and _elapsed_ok(15):
        # Try polishing with different betas
        for beta_val in [300, 1000]:
            if not _elapsed_ok(5):
                break
            def objective2(x, bv=beta_val):
                p = x.reshape(N, 2)
                p = _project(p)
                areas = _all_areas(p)
                max_a = np.max(areas)
                shifted = -bv * (areas - max_a)
                log_sum = np.log(np.sum(np.exp(shifted)))
                soft_min = max_a - log_sum / bv
                return -soft_min

            x0 = best_pts.flatten()
            try:
                result = minimize(objective2, x0, method='L-BFGS-B',
                                  options={'maxiter': 50, 'ftol': 1e-15, 'gtol': 1e-13})
                final_pts = _project(result.x.reshape(N, 2))
                _update_best(final_pts)
            except Exception:
                pass

    if best_pts is None:
        # Fallback: return the first symmetric start
        return _project(starts[0])

    return _project(best_pts)
