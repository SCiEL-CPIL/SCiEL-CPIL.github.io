"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial.distance import pdist

N = 200


def _score(pts):
    """Compute the radius: min of half pairwise distances and min distance to border."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # min distance to each wall
    d_to_border = np.min(np.stack([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ]), axis=1)
    border_r = np.min(d_to_border)
    # pairwise
    d = pdist(pts)
    pair_r = np.min(d) / 2.0
    return min(border_r, pair_r)


# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 145  # 5s safety margin before 150s kill

    # ---- Initialisation: hexagonal packing in square ----
    # Use a hex lattice and crop to the square
    # Start with a dense grid then jitter
    np.random.seed(42)

    # Square grid as starting point (15x14 = 210, take 200)
    m = 15
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    # Take first 200
    grid = grid[:N]
    pts = grid + np.random.normal(0, 0.002, (N, 2))
    # Clip inside
    pts = np.clip(pts, -0.5 + 0.001, 0.5 - 0.001)

    best_pts = pts.copy()
    best_r = _score(best_pts)

    # ---- Force-directed relaxation ----
    # We want to maximize min pair distance and min distance to border.
    # Use a repulsive force model: for each pair, repulsion ~ 1/d^2 if d < threshold
    # Plus attraction to keep points inside.

    def _force(pts):
        """Compute forces on all points. Returns array of shape (N,2)."""
        forces = np.zeros_like(pts)
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                d_vec = pts[i] - pts[j]
                d = np.linalg.norm(d_vec)
                if d < 1e-12:
                    d = 1e-12
                    d_vec = np.array([1e-12, 0.0])
                # Repulsion strength: we want to push apart until d >= 2r
                # Use a soft repulsion
                f = 1.0 / (d * d)
                forces[i] += d_vec / d * f
                forces[j] -= d_vec / d * f
        # Border repulsion: push away from walls
        margin = 0.0
        for i in range(N):
            for axis in range(2):
                if pts[i, axis] < -0.5 + margin:
                    forces[i, axis] += (-(0.5 - margin) - pts[i, axis]) * 10.0
                elif pts[i, axis] > 0.5 - margin:
                    forces[i, axis] -= (pts[i, axis] - (0.5 - margin)) * 10.0
        return forces

    # Simulated annealing with force-directed layout
    T = 0.01  # initial "temperature"
    step = 0.01
    n_iter = 3000
    best_r_so_far = best_r
    best_pts_so_far = best_pts.copy()

    for it in range(n_iter):
        if time.time() > deadline:
            break
        # Perturb: random small displacement
        noise = np.random.normal(0, step, (N, 2))
        new_pts = pts + noise

        # Also apply force-based movement
        f = _force(new_pts)
        new_pts = new_pts + 0.01 * f

        # Clip
        new_pts = np.clip(new_pts, -0.5 + 0.001, 0.5 - 0.001)

        new_r = _score(new_pts)
        if new_r >= best_r_so_far:
            best_r_so_far = new_r
            best_pts_so_far = new_pts.copy()
            pts = new_pts
        else:
            # Metropolis acceptance
            if np.random.random() < np.exp((new_r - best_r_so_far) / (T + 1e-12)):
                pts = new_pts
        # Cool down
        T *= 0.999

    best_pts = best_pts_so_far
    best_r = best_r_so_far

    # ---- High-precision polish: gradient ascent on the min objective ----
    # Use a soft-min to get gradients, then do a few gradient steps
    def _soft_min(arr, beta=1000.0):
        """Approximate min via log-sum-exp."""
        a = -arr * beta
        a_max = np.max(a)
        s = np.sum(np.exp(a - a_max))
        return -a_max / beta - np.log(s) / beta

    def _soft_score(pts, beta=1000.0):
        """Soft version of the score for gradient computation."""
        pts = np.asarray(pts, dtype=np.float64)
        # Border distances
        d_border = np.stack([
            pts[:, 0] + 0.5,
            0.5 - pts[:, 0],
            pts[:, 1] + 0.5,
            0.5 - pts[:, 1]
        ])
        border_soft = _soft_min(d_border, beta)
        # Pairwise distances / 2
        d_mat = pdist(pts)
        pair_soft = _soft_min(d_mat, beta) / 2.0
        return min(border_soft, pair_soft)

    # Gradient-based polish using finite differences on the soft objective
    # But that's expensive. Instead, do a local force-directed refinement.
    for polish_iter in range(200):
        if time.time() > deadline:
            break
        f = _force(best_pts)
        # Move in direction of force (increases min distance)
        move = 0.005 * f
        new_pts = best_pts + move
        new_pts = np.clip(new_pts, -0.5 + 0.001, 0.5 - 0.001)
        new_r = _score(new_pts)
        if new_r >= best_r:
            best_r = new_r
            best_pts = new_pts
        else:
            # Try smaller step
            move = 0.001 * f
            new_pts = best_pts + move
            new_pts = np.clip(new_pts, -0.5 + 0.001, 0.5 - 0.001)
            new_r = _score(new_pts)
            if new_r >= best_r:
                best_r = new_r
                best_pts = new_pts
            else:
                break  # Stagnated

    # Final: ensure all points are strictly inside
    best_pts = np.clip(best_pts, -0.5 + 1e-10, 0.5 - 1e-10)
    return best_pts
# EVOLVE-BLOCK-END
