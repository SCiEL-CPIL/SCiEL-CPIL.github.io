# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time
from scipy.optimize import minimize

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
        # Center and normalize to handle invariances for robustness, though score is invariant
        # Actually, score is invariant to scale and translation.
        # We just compute pdist.
        d = pdist(points)
        if d.size == 0:
            return 0.0
        d_min = np.min(d)
        d_max = np.max(d)
        if d_max == 0:
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio

    def objective_exact(points_flat):
        """
        Exact objective for Nelder-Mead. Minimize -R.
        """
        points = points_flat.reshape(16, 2)
        score = compute_score(points)
        if not np.isfinite(score):
            return 1e10
        return -score

    def objective_smooth(points_flat, beta=100.0):
        """
        Smooth surrogate using log-sum-exp.
        We normalize the points inside to keep them on the unit sphere (in terms of sum of squares)
        to help the optimizer converge faster and avoid scale drift.
        """
        points = points_flat.reshape(16, 2)
        
        # Center
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Scale: normalize so that sum of squared distances to origin is 1
        # This fixes the scale gauge.
        norm = np.linalg.norm(points_centered, axis=1)
        sum_sq = np.sum(norm**2)
        if sum_sq < 1e-12:
            return 1e10
        points_normalized = points_centered / np.sqrt(sum_sq)
        
        d = pdist(points_normalized)
        if d.size == 0 or np.any(d < 1e-12):
            return 1e10
        
        # Soft max
        # To avoid overflow in exp, subtract max
        d_max_val = np.max(d)
        log_sum_exp_max = d_max_val + np.log(np.sum(np.exp(beta * (d - d_max_val))))
        d_max_soft = log_sum_exp_max / beta
        
        # Soft min
        d_min_val = np.min(d)
        # -log-sum-exp(-beta * d)
        # log(sum(exp(-beta * d))) = -beta * d_min_val + log(sum(exp(-beta * (d - d_min_val))))
        log_sum_exp_min = -beta * d_min_val + np.log(np.sum(np.exp(-beta * (d - d_min_val))))
        d_min_soft = -log_sum_exp_min / beta
        
        if d_min_soft <= 1e-12:
            return 1e10
            
        # Minimize log(d_max) - log(d_min)
        return np.log(d_max_soft) - np.log(d_min_soft)

    n = 16
    d = 2

    # Generate diverse initial candidates
    candidates = []

    # 1. Regular 4x4 grid
    grid = np.array([np.meshgrid(np.linspace(-1, 1, 4), np.linspace(-1, 1, 4))]).reshape(2, -1).T
    candidates.append(grid)

    # 2. Triangular lattice subset
    s = 1.0
    h = s * np.sqrt(3) / 2
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * s + (j % 2) * s / 2
            y = j * h
            lattice_points.append([x, y])
    lattice_points = np.array(lattice_points)
    centroid = np.mean(lattice_points, axis=0)
    dists_to_centroid = np.sum((lattice_points - centroid) ** 2, axis=1)
    idx = np.argsort(dists_to_centroid)[:16]
    candidates.append(lattice_points[idx])

    # 3. Points on a circle (regular 16-gon)
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    circle_points = np.column_stack([np.cos(angles), np.sin(angles)])
    candidates.append(circle_points)

    # 4. Two concentric circles (8 inner, 8 outer)
    inner_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_angles = inner_angles + np.pi / 8
    inner_radius = 0.5
    outer_radius = 1.0
    inner_points = np.column_stack([inner_radius * np.cos(inner_angles), inner_radius * np.sin(inner_angles)])
    outer_points = np.column_stack([outer_radius * np.cos(outer_angles), outer_radius * np.sin(outer_angles)])
    two_circle = np.vstack([inner_points, outer_points])
    candidates.append(two_circle)

    # 5. Jittered Grid
    np.random.seed(123)
    jittered_grid = grid + np.random.normal(0, 0.1, grid.shape)
    candidates.append(jittered_grid)

    # 6. Jittered Lattice
    jittered_lattice = candidates[1].copy()
    jittered_lattice += np.random.normal(0, 0.1, jittered_lattice.shape)
    candidates.append(jittered_lattice)

    # 7. Random points
    for seed in [42, 100, 200]:
        np.random.seed(seed)
        random_points = np.random.randn(n, d)
        candidates.append(random_points)

    # Evaluate initial scores
    best_points = None
    best_score = -np.inf

    for cand in candidates:
        score = compute_score(cand)
        if score > best_score:
            best_score = score
            best_points = cand.copy()

    # Sort candidates by score
    scored_candidates = [(compute_score(c), c) for c in candidates]
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    # Phase 1: Smooth Optimization (BFGS)
    # We use the smooth objective which normalizes the points internally.
    # This helps the optimizer stay in a good region.
    for score, cand in scored_candidates[:8]:
        if time.time() > deadline - 120:
            break
        try:
            # BFGS is fast for smooth functions
            # We provide the initial guess.
            result = minimize(objective_smooth, cand.flatten(), method='BFGS',
                             options={'maxiter': 300, 'gtol': 1e-5, 'disp': False})
            
            if time.time() > deadline - 120:
                break
                
            opt_points = result.x.reshape(16, 2)
            opt_score = compute_score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception as e:
            pass

    # Phase 2: Polish with Nelder-Mead on Exact Objective
    # Nelder-Mead is derivative-free and handles non-smoothness well.
    # We run it on the best candidates from Phase 1 and the initial top candidates.
    # To save time, we only run it on the top few results.
    
    # Collect top candidates from Phase 1 and initial set
    # Re-evaluate best_points
    polish_candidates = []
    if best_points is not None:
        polish_candidates.append(best_points)
    polish_candidates.extend([c for s, c in scored_candidates[:5]])
    
    # Remove duplicates roughly
    unique_polish = []
    for p in polish_candidates:
        is_dup = False
        for q in unique_polish:
            if np.allclose(p, q, atol=1e-5):
                is_dup = True
                break
        if not is_dup:
            unique_polish.append(p)

    for cand in unique_polish:
        if time.time() > deadline - 60:
            break
        try:
            # Nelder-Mead with tighter tolerances
            result = minimize(objective_exact, cand.flatten(), method='Nelder-Mead',
                             options={'maxiter': 1000, 'fatol': 1e-10, 'xatol': 1e-10, 'disp': False})
            
            if time.time() > deadline - 60:
                break
                
            opt_points = result.x.reshape(16, 2)
            opt_score = compute_score(opt_points)
            if opt_score > best_score:
                best_score = opt_score
                best_points = opt_points.copy()
        except Exception as e:
            pass

    # Phase 3: Random Restarts with Local Optimization
    # If time remains, try to escape local minima by perturbing the best point.
    if time.time() < deadline - 30:
        for attempt in range(10):
            if time.time() > deadline - 30:
                break
            np.random.seed(attempt * 100 + 55)
            # Perturb best point
            noise = np.random.normal(0, 0.05, (16, 2))
            perturbed = best_points + noise
            try:
                # Quick smooth optimization
                result = minimize(objective_smooth, perturbed.flatten(), method='BFGS',
                                 options={'maxiter': 50, 'gtol': 1e-4, 'disp': False})
                opt_points = result.x.reshape(16, 2)
                opt_score = compute_score(opt_points)
                if opt_score > best_score:
                    best_score = opt_score
                    best_points = opt_points.copy()
                    
                    # If improved, do a quick NM polish
                    if time.time() < deadline - 10:
                        result2 = minimize(objective_exact, opt_points.flatten(), method='Nelder-Mead',
                                     options={'maxiter': 200, 'fatol': 1e-10, 'xatol': 1e-10, 'disp': False})
                        opt_points2 = result2.x.reshape(16, 2)
                        opt_score2 = compute_score(opt_points2)
                        if opt_score2 > best_score:
                            best_score = opt_score2
                            best_points = opt_points2.copy()
            except Exception as e:
                pass

    if best_points is None:
        best_points = candidates[0]

    return best_points

# EVOLVE-BLOCK-END
