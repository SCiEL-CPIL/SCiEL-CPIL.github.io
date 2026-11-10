# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def _compute_score(points):
    """Compute R = (d_min/d_max)^2 for a set of points."""
    d = pdist(points)
    d_min = np.min(d)
    d_max = np.max(d)
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2


def _local_refine(points, max_iter=200):
    """
    Local refinement using a smooth surrogate: maximize min distance by
    minimizing -log(min_dist) subject to max_dist <= 1 (normalized).
    We use a penalty approach to keep it simple and fast.
    """
    n = points.shape[0]
    d = 2

    def _surrogate(flat):
        pts = flat.reshape(n, 2)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 1e10
        # Normalize: scale so max dist = 1
        pts_norm = pts / d_max
        dists_norm = pdist(pts_norm)
        d_min_norm = np.min(dists_norm)
        # Objective: minimize -log(d_min_norm) with a penalty if any dist < 0
        if d_min_norm <= 0:
            return 1e10
        return -np.log(d_min_norm)

    flat = points.flatten()
    try:
        result = minimize(
            _surrogate,
            flat,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        opt_points = result.x.reshape(n, 2)
        return opt_points
    except Exception:
        return points


def _simulated_annealing(points, n_iter=5000, t_start=0.5, t_end=0.001, rng=None):
    """
    Simulated Annealing for the min/max distance ratio problem.
    Moves are random perturbations; acceptance based on score improvement.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = points.shape[0]
    current = points.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score

    # Precompute pair indices for efficiency
    pair_idx = np.array([(i, j) for i in range(n) for j in range(i + 1, n)])
    n_pairs = len(pair_idx)

    for it in range(n_iter):
        # Temperature schedule (geometric cooling)
        t = t_start * (t_end / t_start) ** (it / n_iter)

        # Generate a random perturbation
        # Scale perturbation by current temperature
        scale = t * 0.1
        noise = rng.standard_normal((n, 2)) * scale
        candidate = current + noise

        candidate_score = _compute_score(candidate)

        # Acceptance criterion
        if candidate_score >= current_score:
            current = candidate
            current_score = candidate_score
            if current_score > best_score:
                best = candidate.copy()
                best_score = current_score
        else:
            # Accept with probability exp((candidate_score - current_score) / t)
            # Since scores are in [0, 1], the difference is small
            delta = current_score - candidate_score
            if delta > 0:
                prob = np.exp(-delta / (t + 1e-10))
                if rng.random() < prob:
                    current = candidate
                    current_score = candidate_score

    return best, best_score


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit

    # Generate diverse initial configurations
    rng = np.random.default_rng(42)
    initial_configs = []

    # 1. 4x4 grid
    grid = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    initial_configs.append(grid)

    # 2. Centered 4x4 grid
    grid2 = np.array([[i - 1.5, j - 1.5] for i in range(4) for j in range(4)], dtype=float)
    initial_configs.append(grid2)

    # 3. Hexagonal packing
    hex_pts = []
    for row in range(4):
        for col in range(4):
            x = col + 0.5 * (row % 2)
            y = row * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts, dtype=float)
    initial_configs.append(hex_pts)

    # 4. Random configurations
    for _ in range(10):
        pts = rng.standard_normal((n, d))
        initial_configs.append(pts)

    # 5. Circular configurations
    for radius in [1.0, 2.0]:
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
        initial_configs.append(pts)

    # 6. Two concentric circles
    for inner_r, outer_r in [(0.5, 1.5), (0.7, 2.0)]:
        angles_inner = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        angles_outer = np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8
        pts_inner = np.column_stack([inner_r * np.cos(angles_inner), inner_r * np.sin(angles_inner)])
        pts_outer = np.column_stack([outer_r * np.cos(angles_outer), outer_r * np.sin(angles_outer)])
        pts = np.vstack([pts_inner, pts_outer])
        initial_configs.append(pts)

    best_points = initial_configs[0].copy()
    best_score = _compute_score(best_points)

    # Phase 1: Local refinement of each initial config
    for config in initial_configs:
        if time.time() > deadline:
            break

        refined = _local_refine(config, max_iter=100)
        score = _compute_score(refined)
        if score > best_score:
            best_score = score
            best_points = refined.copy()

    # Phase 2: Simulated Annealing from best and several good configs
    # Sort configs by score
    scored_configs = []
    for config in initial_configs:
        refined = _local_refine(config, max_iter=50)
        score = _compute_score(refined)
        scored_configs.append((score, refined))
    scored_configs.sort(key=lambda x: -x[0])

    # Run SA from top configs
    sa_configs = [c[1] for c in scored_configs[:5]]
    sa_configs.append(best_points)

    for config in sa_configs:
        if time.time() > deadline:
            break

        sa_points, sa_score = _simulated_annealing(
            config,
            n_iter=3000,
            t_start=0.3,
            t_end=0.0005,
            rng=np.random.default_rng()
        )
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()

        # Local refine after SA
        refined = _local_refine(sa_points, max_iter=100)
        score = _compute_score(refined)
        if score > best_score:
            best_score = score
            best_points = refined.copy()

    # Phase 3: More random restarts with SA if time permits
    while time.time() < deadline - 30:
        pts = rng.standard_normal((n, d))
        refined = _local_refine(pts, max_iter=50)
        sa_points, sa_score = _simulated_annealing(
            refined,
            n_iter=2000,
            t_start=0.2,
            t_end=0.0005,
            rng=np.random.default_rng()
        )
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()

        final_refined = _local_refine(sa_points, max_iter=100)
        score = _compute_score(final_refined)
        if score > best_score:
            best_score = score
            best_points = final_refined.copy()

    # Ensure valid output
    if best_points.shape != (n, d):
        best_points = initial_configs[0]

    return best_points


# EVOLVE-BLOCK-END
