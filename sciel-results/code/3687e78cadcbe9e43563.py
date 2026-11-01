"""Minimize squared max/min pairwise distance ratio for 14 points in 3D."""
import numpy as np
import time

N = 14
D = 3


def _tetrahedron_points():
    """Return 14 points: 4 tetrahedron vertices + 6 edge midpoints, centered at origin."""
    # Regular tetrahedron vertices (alternating cube vertices)
    v = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float)
    # Edge midpoints: 6 edges of tetrahedron
    edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    mids = np.array([(v[i]+v[j])/2.0 for i,j in edges], dtype=float)
    pts = np.vstack([v, mids])
    # Center at origin
    pts -= pts.mean(axis=0)
    return pts


def _pairwise_dists(pts):
    """Return array of all pairwise squared distances."""
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return d2[iu]


def _score(pts):
    """Squared ratio of max to min pairwise distance."""
    d2 = _pairwise_dists(pts)
    mn = np.min(d2)
    mx = np.max(d2)
    if mn <= 1e-15:
        return np.inf
    return (mx / mn) ** 2


def _objective(pts_flat):
    pts = pts_flat.reshape(N, D)
    return _score(pts)


def _local_optimize(pts, time_budget):
    """Local refinement using Nelder-Mead on the flat vector."""
    from scipy.optimize import minimize
    start = time.time()
    best = pts.copy()
    best_score = _score(best)
    # Flatten
    x0 = best.reshape(-1)
    try:
        res = minimize(
            _objective, x0, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if res.fun < best_score:
            best = res.x.reshape(N, D)
            best_score = res.fun
    except Exception:
        pass
    return best, best_score


def construct():
    deadline = time.time() + 110.0  # safety margin under 120s
    start = time.time()

    # Base configuration: tetrahedron + edge midpoints
    base = _tetrahedron_points()

    best = base.copy()
    best_score = _score(base)

    # Try a few random perturbations + local opt to see if we can improve
    rng = np.random.default_rng(12345)
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        # Small perturbation
        scale = 0.05 * (0.5 ** min(attempts, 6))
        perturbed = base + rng.normal(0, scale, size=base.shape)
        # Center
        perturbed -= perturbed.mean(axis=0)
        # Local optimize
        cand, cs = _local_optimize(perturbed, 5.0)
        if cs < best_score:
            best = cand
            best_score = cs

    return best


if __name__ == "__main__":
    pts = construct()
    print("score:", _score(pts))
