# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd

def _score(points):
    """Compute R = (d_min / d_max)^2."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    return (d_min / d_max) ** 2

def _local_search(points, time_budget, rng):
    """
    Local search with adaptive step sizes.
    Returns (best_points, best_score).
    """
    n, d = points.shape
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score

    start = time.time()
    step_size = 0.1
    no_improve_count = 0
    max_no_improve = 500

    while time.time() - start < time_budget:
        # Pick a random point and a random direction
        i = rng.integers(n)
        direction = rng.standard_normal(d)
        direction /= np.linalg.norm(direction) + 1e-15

        # Try a few step sizes around current
        for s in [step_size, step_size * 0.5, step_size * 2.0]:
            candidate = current.copy()
            candidate[i] += s * direction
            cand_score = _score(candidate)

            if cand_score > current_score:
                current = candidate
                current_score = cand_score
                no_improve_count = 0

                if cand_score > best_score:
                    best = current.copy()
                    best_score = cand_score

                break
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                step_size *= 0.7
                no_improve_count = 0
                if step_size < 1e-6:
                    break

    return best, best_score

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2

    # Hard deadline: 600s wall-clock, leave 30s margin
    deadline = time.time() + 570

    best_points = None
    best_score = -1.0

    # --- Phase 1: Good initial configurations ---
    # 1. Regular hexagon + inner points
    # 2. Two concentric regular octagons
    # 3. Random with seed
    # 4. Grid-like with jitter

    initial_configs = []

    # Config 1: Two concentric regular octagons
    angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
    r_outer = 1.0
    r_inner = 0.5
    pts1 = np.vstack([
        np.column_stack([r_outer * np.cos(angles_outer), r_outer * np.sin(angles_outer)]),
        np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    ])
    initial_configs.append(pts1)

    # Config 2: Regular hexagon (6) + inner hexagon (6) + 2 center-ish
    angles_h = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    r_h_outer = 1.0
    r_h_inner = 0.4
    pts2 = np.vstack([
        np.column_stack([r_h_outer * np.cos(angles_h), r_h_outer * np.sin(angles_h)]),
        np.column_stack([r_h_inner * np.cos(angles_h), r_h_inner * np.sin(angles_h)]),
        np.array([[0.1, 0.1], [-0.1, -0.1]])
    ])
    initial_configs.append(pts2)

    # Config 3: Random
    rng = np.random.default_rng(42)
    initial_configs.append(rng.standard_normal((n, d)))

    # Config 4: Square grid 4x4 with jitter
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts4 = np.column_stack([gx.ravel(), gy.ravel()])
    pts4 += rng.standard_normal((n, d)) * 0.05
    initial_configs.append(pts4)

    # Config 5: Circle with 16 points
    angles_c = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts5 = np.column_stack([np.cos(angles_c), np.sin(angles_c)])
    initial_configs.append(pts5)

    # Config 6: Two squares rotated
    angles_sq1 = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
    angles_sq2 = angles_sq1 + np.pi/4
    r1, r2 = 1.0, 0.6
    pts6 = np.vstack([
        np.column_stack([r1 * np.cos(angles_sq1), r1 * np.sin(angles_sq1)]),
        np.column_stack([r2 * np.cos(angles_sq2), r2 * np.sin(angles_sq2)]),
        np.array([[0.3, 0.3], [-0.3, 0.3], [0.3, -0.3], [-0.3, -0.3]])
    ])
    initial_configs.append(pts6)

    # Evaluate and improve each initial config
    for idx, init in enumerate(initial_configs):
        if time.time() > deadline - 60:
            break

        # Normalize to unit scale
        init = init - init.mean(axis=0)
        scale = np.max(np.linalg.norm(init, axis=1))
        if scale > 1e-15:
            init = init / scale

        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()

        # Local search
        remaining = deadline - time.time()
        if remaining < 5:
            break
        time_budget = min(remaining - 5, 120)  # Cap per-config time
        improved_pts, improved_score = _local_search(init, time_budget, rng)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_pts.copy()

    # --- Phase 2: Additional random restarts if time permits ---
    while time.time() < deadline - 30:
        init = rng.standard_normal((n, d))
        init = init - init.mean(axis=0)
        scale = np.max(np.linalg.norm(init, axis=1))
        if scale > 1e-15:
            init = init / scale

        s = _score(init)
        if s > best_score:
            best_score = s
            best_points = init.copy()

        remaining = deadline - time.time()
        if remaining < 10:
            break
        time_budget = min(remaining - 10, 60)
        improved_pts, improved_score = _local_search(init, time_budget, rng)
        if improved_score > best_score:
            best_score = improved_score
            best_points = improved_pts.copy()

    if best_points is None:
        return np.random.randn(n, d)

    return best_points

# EVOLVE-BLOCK-END
