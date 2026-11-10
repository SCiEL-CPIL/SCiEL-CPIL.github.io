# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin before 630s limit

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

    def compute_score_smooth(points, beta=50.0):
        """
        Smooth surrogate for the min/max ratio using log-sum-exp.
        We maximize (d_min/d_max)^2.
        Using soft-min and soft-max:
        soft_min(d, beta) = -1/beta * log(sum(exp(-beta*d_i)))
        soft_max(d, beta) = 1/beta * log(sum(exp(beta*d_i)))
        Then score = (soft_min / soft_max)^2
        """
        if points.shape != (16, 2):
            return -np.inf
        d = pdist(points)
        if d.size == 0:
            return 0.0
        
        # Soft min
        neg_d = -d
        max_neg_d = np.max(neg_d)
        soft_min = -1.0 / beta * (max_neg_d + np.log(np.sum(np.exp(neg_d - max_neg_d))))
        
        # Soft max
        max_d = np.max(d)
        soft_max = 1.0 / beta * (max_d + np.log(np.sum(np.exp(d - max_d))))
        
        if soft_max <= 0:
            return 0.0
        
        ratio = soft_min / soft_max
        return ratio * ratio

    # Configuration 1: 4x4 grid
    grid_4x4 = np.array([[i, j] for i in range(4) for j in range(4)], dtype=float)
    grid_4x4 = (grid_4x4 - 1.5) / 1.5  # Center and normalize

    # Configuration 2: Two concentric squares (4 inner, 12 outer)
    inner_r = 0.5
    outer_r = 1.0
    inner_pts = np.array([[inner_r, inner_r], [-inner_r, inner_r], 
                          [-inner_r, -inner_r], [inner_r, -inner_r]])
    outer_pts = []
    # 4 corners
    outer_pts.append([outer_r, outer_r])
    outer_pts.append([-outer_r, outer_r])
    outer_pts.append([-outer_r, -outer_r])
    outer_pts.append([outer_r, -outer_r])
    # 8 edge midpoints
    outer_pts.append([0, outer_r])
    outer_pts.append([outer_r, 0])
    outer_pts.append([0, -outer_r])
    outer_pts.append([-outer_r, 0])
    outer_pts = np.array(outer_pts)
    config2 = np.vstack([inner_pts, outer_pts])

    # Configuration 3: Triangular lattice selection
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice = []
    for row in range(-2, 3):
        for col in range(-2, 3):
            x = col * s + (row % 2) * s / 2
            y = row * h
            lattice.append([x, y])
    lattice = np.array(lattice)
    centroid = np.mean(lattice, axis=0)
    dists = np.sum((lattice - centroid) ** 2, axis=1)
    idx = np.argsort(dists)[:16]
    config3 = lattice[idx]
    config3 = (config3 - np.mean(config3, axis=0)) / np.max(np.max(np.abs(config3), axis=0))

    # Configuration 4: 16-gon (regular polygon)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    config4 = np.column_stack([np.cos(angles), np.sin(angles)])

    # Configuration 5: Two concentric circles (8+8)
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = inner_angles + np.pi / 8
    inner_r5 = 0.5
    outer_r5 = 1.0
    inner_pts5 = np.column_stack([inner_r5 * np.cos(inner_angles), inner_r5 * np.sin(inner_angles)])
    outer_pts5 = np.column_stack([outer_r5 * np.cos(outer_angles), outer_r5 * np.sin(outer_angles)])
    config5 = np.vstack([inner_pts5, outer_pts5])

    # Configuration 6: 3+6+7 arrangement (approximate)
    center = np.array([[0.0, 0.0]])
    ring6_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    ring6_r = 0.6
    ring6 = np.column_stack([ring6_r * np.cos(ring6_angles), ring6_r * np.sin(ring6_angles)])
    ring9_angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    ring9_r = 1.0
    ring9 = np.column_stack([ring9_r * np.cos(ring9_angles), ring9_r * np.sin(ring9_angles)])
    config6 = np.vstack([center, ring6, ring9])

    # Configuration 7: Hexagonal lattice based
    hex_s = 1.0
    hex_h = hex_s * np.sqrt(3) / 2
    hex_pts = []
    for row in range(-2, 3):
        for col in range(-2, 3):
            x = col * hex_s + (row % 2) * hex_s / 2
            y = row * hex_h
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    # Select 16 closest to centroid
    hex_centroid = np.mean(hex_pts, axis=0)
    hex_dists = np.sum((hex_pts - hex_centroid) ** 2, axis=1)
    hex_idx = np.argsort(hex_dists)[:16]
    config7 = hex_pts[hex_idx]
    config7 = (config7 - np.mean(config7, axis=0)) / np.max(np.max(np.abs(config7), axis=0))

    # Configuration 8: Two concentric hexagons (6+6+4)
    hex_angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    inner_hex_r = 0.5
    outer_hex_r = 1.0
    inner_hex = np.column_stack([inner_hex_r * np.cos(hex_angles), inner_hex_r * np.sin(hex_angles)])
    outer_hex = np.column_stack([outer_hex_r * np.cos(hex_angles + np.pi/6), outer_hex_r * np.sin(hex_angles + np.pi/6)])
    # Add 4 more points at cardinal directions on outer ring
    extra_pts = np.array([[0, 1.0], [1.0, 0], [0, -1.0], [-1.0, 0.0]])
    config8 = np.vstack([inner_hex, outer_hex, extra_pts])

    configs = [grid_4x4, config2, config3, config4, config5, config6, config7, config8]

    best_score = -np.inf
    best_points = None

    for cfg in configs:
        score = compute_score(cfg)
        if score > best_score:
            best_score = score
            best_points = cfg.copy()

    # Phase 1: Smooth objective optimization with Nelder-Mead
    if best_points is not None and time.time() < deadline - 120:
        from scipy.optimize import minimize

        def objective_smooth(points_flat, beta=50.0):
            points = points_flat.reshape(16, 2)
            score = compute_score_smooth(points, beta)
            if not np.isfinite(score):
                return 1e10
            return -score

        try:
            result = minimize(objective_smooth, best_points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 3000, 'fatol': 1e-12, 'xatol': 1e-12})
            opt_points = result.x.reshape(16, 2)
            opt_score = compute_score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass

    # Phase 2: Fine-tune with raw objective
    if best_points is not None and time.time() < deadline - 90:
        from scipy.optimize import minimize

        def objective_raw(points_flat):
            points = points_flat.reshape(16, 2)
            score = compute_score(points)
            if not np.isfinite(score):
                return 1e10
            return -score

        try:
            result = minimize(objective_raw, best_points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 2000, 'fatol': 1e-14, 'xatol': 1e-14})
            opt_points = result.x.reshape(16, 2)
            opt_score = compute_score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass

    # Phase 3: Multiple random restarts with smooth objective
    if time.time() < deadline - 60:
        from scipy.optimize import minimize

        for seed in range(10):
            if time.time() > deadline - 30:
                break
            np.random.seed(seed * 123 + 45)
            rand_pts = np.random.randn(16, 2)
            rand_pts = (rand_pts - np.mean(rand_pts, axis=0))
            std = np.std(rand_pts)
            if std > 0:
                rand_pts = rand_pts / std

            def objective_smooth_restart(points_flat, beta=50.0):
                points = points_flat.reshape(16, 2)
                score = compute_score_smooth(points, beta)
                if not np.isfinite(score):
                    return 1e10
                return -score

            try:
                result = minimize(objective_smooth_restart, rand_pts.flatten(), method='Nelder-Mead',
                                 options={'maxiter': 1500, 'fatol': 1e-12, 'xatol': 1e-12})
                opt_points = result.x.reshape(16, 2)
                opt_score = compute_score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
            except Exception:
                pass

    # Phase 4: High-precision polish pass
    if best_points is not None and time.time() < deadline - 15:
        from scipy.optimize import minimize

        def objective_polish(points_flat):
            points = points_flat.reshape(16, 2)
            score = compute_score(points)
            if not np.isfinite(score):
                return 1e10
            return -score

        try:
            result = minimize(objective_polish, best_points.flatten(), method='Nelder-Mead',
                             options={'maxiter': 5000, 'fatol': 1e-16, 'xatol': 1e-16})
            opt_points = result.x.reshape(16, 2)
            opt_score = compute_score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception:
            pass

    if best_points is None:
        best_points = grid_4x4

    return best_points

# EVOLVE-BLOCK-END
