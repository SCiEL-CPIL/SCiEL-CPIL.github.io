# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # Safety margin of 10 seconds

    def compute_score(points):
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def objective(points_flat):
        points = points_flat.reshape(16, 2)
        score = compute_score(points)
        # We want to maximize score, so minimize negative score
        if not np.isfinite(score):
            return 1e10
        return -score

    from scipy.optimize import minimize

    best_points = None
    best_score = -np.inf

    n = 16
    d = 2

    # Generate diverse initial candidates
    candidates = []

    # 1. Regular 4x4 grid
    grid = np.array([np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))]).reshape(2, -1).T
    candidates.append(grid)

    # 2. Triangular lattice subset
    # Generate a triangular lattice and pick 16 points
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * s + (j % 2) * s / 2
            y = j * h
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    # Select 16 points closest to the centroid
    centroid = np.mean(lattice_points, axis=0)
    dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
    idx = np.argsort(dists_to_centroid)[:16]
    candidates.append(lattice_points[idx])

    # 3. Points on a circle (regular 16-gon)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    circle_points = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(circle_points)

    # 4. Random points in a square
    np.random.seed(42)
    for seed in range(5):
        np.random.seed(seed + 100)
        random_points = np.random.randn(n, d)
        candidates.append(random_points)

    # 5. Two concentric circles (8 on inner, 8 on outer)
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = inner_angles + np.pi / 8
    inner_radius = 0.5
    outer_radius = 1.0
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    two_circle = np.vstack([inner_points, outer_points])
    candidates.append(two_circle)

    # 6. Random points with different seeds (from inspiration)
    np.random.seed(42)
    random_seed_points = np.random.randn(n, d)
    candidates.append(random_seed_points)

    # Evaluate initial scores
    for i, cand in enumerate(candidates):
        score = compute_score(cand)
        if score > best_score:
            best_score = score
            best_points = cand.copy()

    # Try local optimization on the best few candidates
    # Sort candidates by score
    scored_candidates = [(compute_score(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Optimize the top 5 candidates
    for score, cand in scored_candidates[:5]:
        if time.time() > deadline:
            break
        try:
            result = minimize(objective, cand.flatten(), method='Nelder-Mead',
                             options={'maxiter': 1000, 'fatol': 1e-9, 'xatol': 1e-9})
            if time.time() > deadline:
                break
            opt_points = result.x.reshape(16, 2)
            opt_score = compute_score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception as e:
            pass

    # If we still have time, try a few more random restarts with optimization
    if time.time() < deadline - 30:
        for attempt in range(10):
            if time.time() > deadline:
                break
            np.random.seed(attempt * 7 + 13)
            random_points = np.random.randn(n, d)
            # Scale to have similar range
            random_points = random_points / np.std(random_points)
            try:
                result = minimize(objective, random_points.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 500, 'fatol': 1e-9, 'xatol': 1e-9})
                if time.time() > deadline:
                    break
                opt_points = result.x.reshape(16, 2)
                opt_score = compute_score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
            except Exception as e:
                pass

    # Final check
    if best_points is None:
        # Fallback to grid
        best_points = candidates[0]

    return best_points

# EVOLVE-BLOCK-END
