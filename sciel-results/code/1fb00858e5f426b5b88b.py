"""Heilbronn 11-point problem: maximize minimum triangle area in unit equilateral triangle."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from itertools import combinations

N = 11
S3H = np.sqrt(3) / 2
TRI_AREA = 0.5 * 1.0 * S3H  # area of the equilateral triangle with side 1

# Precompute all (i,j,k) triples
_TRIPLES = list(combinations(range(N), 3))
_N_TRIPLES = len(_TRIPLES)  # 165

# Convert triple indices to flat indices for vectorized area computation
_I_IDX = np.array([t[0] for t in _TRIPLES])
_J_IDX = np.array([t[1] for t in _TRIPLES])
_K_IDX = np.array([t[2] for t in _TRIPLES])


def _min_area(points):
    """Compute minimum triangle area among all triples. points: (N,2) array."""
    p_i = points[_I_IDX]
    p_j = points[_J_IDX]
    p_k = points[_K_IDX]
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    areas = 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)
    return np.min(areas)


def _all_areas(points):
    """Compute all triangle areas. points: (N,2) array."""
    p_i = points[_I_IDX]
    p_j = points[_J_IDX]
    p_k = points[_K_IDX]
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    return 0.5 * np.abs(dx1 * dy2 - dy1 * dx2)


def _soft_min_area(areas, beta):
    """
    Smooth approximation of the minimum using log-sum-exp.
    soft_min ≈ -(1/beta) * log(sum(exp(-beta * areas)))
    As beta → ∞, this approaches the true minimum.
    """
    # Stable computation: factor out max
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum_exp / beta


def _hexagonal_init(rng):
    """
    Generate a hexagonal grid pattern inside the equilateral triangle.
    This gives a good starting configuration with roughly uniform spacing.
    """
    pts = np.zeros((N, 2))
    # Place points in a hexagonal pattern
    # For 11 points, use a 4-3-3-1 or similar arrangement
    # Actually, let's use a more systematic approach:
    # Divide the triangle into smaller triangles and place points at centroids
    
    # Simpler: use a grid in barycentric coordinates
    # Barycentric coords (u, v) with u >= 0, v >= 0, u+v <= 1
    # Map to cartesian: x = u + 0.5*v, y = S3H*v
    
    # Try a pattern: corners + edge midpoints + interior
    # Corner 1: (0, 0) -> barycentric (0, 0)
    # Corner 2: (1, 0) -> barycentric (1, 0)
    # Corner 3: (0.5, S3H) -> barycentric (0, 1)
    
    # Let's use a more spread out pattern
    # Row 0 (bottom): 4 points along bottom edge
    # Row 1: 3 points
    # Row 2: 3 points
    # Row 3: 1 point at top
    
    # Actually, for 11 points in an equilateral triangle, a good pattern is:
    # Think of it as layers. Let me use a systematic grid.
    
    # Use barycentric grid: for each "level" k from 0 to some max
    # At level k, there are k+1 points along the base of that sub-triangle
    
    # Simpler approach: just place points at specific good positions
    # based on known Heilbronn configurations or just a regular grid
    
    # Let's use a 4x3 grid clipped to the triangle
    positions = []
    for i in range(4):
        for j in range(3):
            # Barycentric coords
            u = (i + 0.5) / 4.0
            v = (j + 0.5) / 3.0
            if u + v <= 1.0:
                positions.append((u, v))
    
    # If we don't have exactly N, adjust
    while len(positions) < N:
        # Add more points
        u = rng.random()
        v = rng.random()
        if u + v <= 1.0:
            positions.append((u, v))
        else:
            s = u + v
            positions.append((u/s, v/s))
    
    positions = positions[:N]
    
    for idx, (u, v) in enumerate(positions):
        pts[idx, 0] = u + 0.5 * v
        pts[idx, 1] = S3H * v
    
    return pts


def _project_to_triangle(pts):
    """Project points to be inside the equilateral triangle."""
    result = pts.copy()
    for i in range(N):
        px, py = result[i]
        # Convert to barycentric
        v_i = py / S3H
        u_i = px - 0.5 * v_i
        
        # Clamp
        u_i = max(0.0, min(1.0, u_i))
        v_i = max(0.0, min(1.0, v_i))
        
        if u_i + v_i > 1.0:
            s = u_i + v_i
            u_i /= s
            v_i /= s
        
        result[i, 0] = u_i + 0.5 * v_i
        result[i, 1] = S3H * v_i
    
    return result


def _simulated_annealing(start_pts, time_start, time_limit, rng, beta_start=50.0, beta_end=500.0, 
                         t_start=1.0, t_end=0.001, n_iter=5000):
    """
    Simulated annealing to maximize minimum triangle area.
    Uses soft-min as the energy function for smooth landscape.
    """
    current_pts = start_pts.copy()
    current_score = _min_area(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    n_accepted = 0
    n_rejected = 0
    
    for it in range(n_iter):
        # Check time
        if (time.time() - time_start) > (time_limit - 5.0):
            break
        
        # Temperature schedule (geometric cooling)
        alpha = it / max(1, n_iter - 1)
        temperature = t_start * (t_end / t_start) ** alpha
        beta = beta_start * (beta_end / beta_start) ** alpha
        
        # Generate perturbation
        # Choose a random point to move
        idx = rng.randint(N)
        
        # Random step size that decreases with temperature
        step_size = 0.05 * (temperature / t_start) ** 0.5
        delta = rng.normal(0, step_size, 2)
        
        new_pts = current_pts.copy()
        new_pts[idx] += delta
        new_pts = _project_to_triangle(new_pts)
        
        # Compute scores
        new_areas = _all_areas(new_pts)
        new_min = np.min(new_areas)
        
        # Acceptance criterion
        if new_min >= current_score:
            # Accept if better or equal
            current_pts = new_pts
            current_score = new_min
            n_accepted += 1
            if new_min > best_score:
                best_score = new_min
                best_pts = new_pts.copy()
        else:
            # Accept worse solutions with probability exp(-delta / temperature)
            delta = current_score - new_min
            if delta > 0:
                # Scale delta to be comparable to temperature
                # The area scale is ~0.01, so we need to scale appropriately
                prob = np.exp(-delta / (temperature * 0.01))
                if rng.random() < prob:
                    current_pts = new_pts
                    current_score = new_min
                    n_accepted += 1
                else:
                    n_rejected += 1
            else:
                # Should not happen, but accept anyway
                current_pts = new_pts
                current_score = new_min
                n_accepted += 1
    
    return best_pts, best_score


def _local_polish(pts, time_start, time_limit):
    """
    Local optimization using L-BFGS-B on the min-area objective.
    """
    if (time.time() - time_start) > (time_limit - 10.0):
        return pts, _min_area(pts)
    
    # Convert to barycentric
    u = np.zeros(N)
    v = np.zeros(N)
    for i in range(N):
        px, py = pts[i]
        v[i] = py / S3H
        u[i] = px - 0.5 * v[i]
    
    x0 = np.column_stack([u, v]).flatten()
    
    def objective(x):
        u_opt = x[0::2]
        v_opt = x[1::2]
        
        # Penalty for being outside
        pen = 100.0 * (
            np.sum(np.maximum(0, -u_opt)**2) +
            np.sum(np.maximum(0, -v_opt)**2) +
            np.sum(np.maximum(0, u_opt + v_opt - 1.0)**2)
        )
        
        pts_opt = np.empty((N, 2))
        pts_opt[:, 0] = u_opt + 0.5 * v_opt
        pts_opt[:, 1] = S3H * v_opt
        
        areas = _all_areas(pts_opt)
        min_a = np.min(areas)
        
        return -min_a**2 + pen
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        if res.x is not None:
            u_opt = res.x[0::2]
            v_opt = res.x[1::2]
            u_opt = np.clip(u_opt, 0, 1)
            v_opt = np.clip(v_opt, 0, 1)
            mask = (u_opt + v_opt) > 1.0
            if np.any(mask):
                s = u_opt[mask] + v_opt[mask]
                u_opt[mask] /= s
                v_opt[mask] /= s
            
            opt_pts = np.empty((N, 2))
            opt_pts[:, 0] = u_opt + 0.5 * v_opt
            opt_pts[:, 1] = S3H * v_opt
            
            score = _min_area(opt_pts)
            return opt_pts, score
    except Exception:
        pass
    
    return pts, _min_area(pts)


def construct():
    start_time = time.time()
    TIME_LIMIT = 145  # safety margin before 150s kill
    
    def _time_up():
        return (time.time() - start_time) > TIME_LIMIT
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.RandomState(42)
    
    # Stage 1: Hexagonal initialization + Simulated Annealing
    hex_pts = _hexagonal_init(rng)
    sa_pts, sa_score = _simulated_annealing(
        hex_pts, start_time, TIME_LIMIT, rng,
        beta_start=50.0, beta_end=500.0,
        t_start=1.0, t_end=0.001,
        n_iter=3000
    )
    
    if sa_score > best_score:
        best_score = sa_score
        best_points = sa_pts.copy()
    
    # Stage 2: Local polish
    if best_points is not None and not _time_up():
        polished_pts, polished_score = _local_polish(best_points, start_time, TIME_LIMIT)
        if polished_score > best_score:
            best_score = polished_score
            best_points = polished_pts.copy()
    
    # Stage 3: Multiple SA restarts with different initializations
    n_restarts = 5
    for r in range(n_restarts):
        if _time_up():
            break
        
        # Different initialization strategies
        if r == 0:
            # Perturbed hexagonal
            init_pts = hex_pts + rng.normal(0, 0.05, (N, 2))
            init_pts = _project_to_triangle(init_pts)
        elif r == 1:
            # Random
            init_pts = rng.random((N, 2)) * np.array([1.0, S3H])
            init_pts = _project_to_triangle(init_pts)
        elif r == 2:
            # Clustered near corners
            init_pts = np.zeros((N, 2))
            corners = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, S3H]])
            for i in range(N):
                c = corners[i % 3]
                init_pts[i] = c + rng.normal(0, 0.1, 2)
            init_pts = _project_to_triangle(init_pts)
        elif r == 3:
            # Ring pattern
            init_pts = np.zeros((N, 2))
            center = np.array([0.5, S3H / 3])
            for i in range(N):
                angle = 2 * np.pi * i / N
                radius = 0.2 + 0.1 * (i % 2)
                init_pts[i] = center + radius * np.array([np.cos(angle), np.sin(angle)])
            init_pts = _project_to_triangle(init_pts)
        else:
            # Best point + random perturbation
            if best_points is not None:
                init_pts = best_points + rng.normal(0, 0.03, (N, 2))
            else:
                init_pts = rng.random((N, 2)) * np.array([1.0, S3H])
            init_pts = _project_to_triangle(init_pts)
        
        # SA with different parameters
        sa_pts_r, sa_score_r = _simulated_annealing(
            init_pts, start_time, TIME_LIMIT, rng,
            beta_start=30.0 + 20.0 * r,
            beta_end=300.0 + 100.0 * r,
            t_start=0.5 + 0.2 * r,
            t_end=0.0005,
            n_iter=2000
        )
        
        if sa_score_r > best_score:
            best_score = sa_score_r
            best_points = sa_pts_r.copy()
        
        # Local polish after each SA
        if not _time_up():
            polished_pts, polished_score = _local_polish(best_points, start_time, TIME_LIMIT)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts.copy()
    
    # Stage 4: Final polish
    if best_points is not None and not _time_up():
        final_pts, final_score = _local_polish(best_points, start_time, TIME_LIMIT)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()
    
    # Stage 5: Quick DE if time permits
    if best_points is not None and (TIME_LIMIT - (time.time() - start_time)) > 30:
        try:
            def _de_objective(x):
                u = x[0::2]
                v = x[1::2]
                pen = 100.0 * (
                    np.sum(np.maximum(0, -u)**2) +
                    np.sum(np.maximum(0, -v)**2) +
                    np.sum(np.maximum(0, u + v - 1.0)**2)
                )
                pts = np.empty((N, 2))
                pts[:, 0] = u + 0.5 * v
                pts[:, 1] = S3H * v
                areas = _all_areas(pts)
                return -np.min(areas)**2 + pen
            
            bounds = [(0.0, 1.0)] * (2 * N)
            res_de = differential_evolution(
                _de_objective,
                bounds,
                seed=123,
                popsize=15,
                maxiter=50,
                tol=1e-8,
                polish=False,
                init='sobol'
            )
            
            if res_de.x is not None:
                u = res_de.x[0::2]
                v = res_de.x[1::2]
                u = np.clip(u, 0, 1)
                v = np.clip(v, 0, 1)
                mask = (u + v) > 1.0
                if np.any(mask):
                    s = u[mask] + v[mask]
                    u[mask] /= s
                    v[mask] /= s
                de_pts = np.empty((N, 2))
                de_pts[:, 0] = u + 0.5 * v
                de_pts[:, 1] = S3H * v
                
                de_score = _min_area(de_pts)
                if de_score > best_score:
                    best_score = de_score
                    best_points = de_pts.copy()
                
                # Polish DE result
                if not _time_up():
                    polished_pts, polished_score = _local_polish(best_points, start_time, TIME_LIMIT)
                    if polished_score > best_score:
                        best_score = polished_score
                        best_points = polished_pts.copy()
        except Exception:
            pass
    
    if best_points is None:
        # Fallback
        best_points = _hexagonal_init(np.random.RandomState(0))
    
    return best_points
