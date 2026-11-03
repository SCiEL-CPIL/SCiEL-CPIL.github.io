"""Improved packing of 200 equal circles in a unit square."""
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import Delaunay
from scipy.optimize import minimize

N = 200


def _compute_radius(points):
    """Compute the max common radius for points in [-0.5, 0.5]^2."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    if not np.all((points >= -0.5) & (points <= 0.5)):
        return 0.0
    # Min distance to boundary
    dist_boundary = 0.5 - np.max(np.abs(points), axis=1)
    min_boundary = np.min(dist_boundary)
    if min_boundary < 0:
        return 0.0
    # Min pairwise distance
    if N < 2:
        min_pair = float('inf')
    else:
        d = pdist(points)
        min_pair = np.min(d)
    return min(min_pair / 2.0, min_boundary)


def _generate_hex_initial(n_points, jitter_scale=0.01):
    """
    Generate initial points on a jittered hexagonal lattice.
    This provides a good starting point for optimization.
    """
    # Determine grid dimensions
    # For hex packing, we want roughly sqrt(n_points) rows and cols
    # Hexagonal: each row has ~sqrt(3)/2 * spacing vertical distance
    
    # Try different aspect ratios to find best fit
    best_points = None
    best_radius = 0.0
    
    for aspect in [0.8, 0.9, 1.0, 1.1, 1.2]:
        # Estimate rows and cols
        rows = int(np.sqrt(n_points / aspect))
        cols = int(np.sqrt(n_points * aspect))
        
        if rows < 1 or cols < 1:
            continue
        
        # Generate hex lattice
        # Spacing to fit in unit square with margin
        # Width: (cols-1)*s + s/2 (offset) <= 1 - margin
        # Height: (rows-1)*s*sqrt(3)/2 <= 1 - margin
        
        # Conservative spacing
        s_x = 1.0 / (cols + 0.5)
        s_y = 1.0 / (rows * np.sqrt(3) / 2 + 0.5)
        s = min(s_x, s_y) * 0.95  # 5% margin
        
        points = []
        for row in range(rows):
            offset = (row % 2) * s / 2
            n_in_row = cols + (1 if row % 2 == 0 else 0)  # Alternate row lengths
            for col in range(n_in_row):
                if len(points) >= n_points:
                    break
                x = -0.5 + s / 2 + col * s + offset
                y = -0.5 + s * np.sqrt(3) / 4 + row * s * np.sqrt(3) / 2
                points.append([x, y])
        
        if len(points) < n_points:
            continue
        
        points = np.array(points[:n_points])
        
        # Add small random jitter to break symmetry
        rng = np.random.RandomState(42)
        points += rng.uniform(-jitter_scale, jitter_scale, size=points.shape)
        
        # Clip to bounds
        points = np.clip(points, -0.5 + 0.001, 0.5 - 0.001)
        
        r = _compute_radius(points)
        if r > best_radius:
            best_radius = r
            best_points = points.copy()
    
    if best_points is None:
        # Fallback: simple grid
        m = int(np.ceil(np.sqrt(n_points)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_points = grid[:n_points].copy()
    
    return best_points


def _objective_and_grad(x_flat):
    """
    Objective function for L-BFGS-B: minimize negative radius.
    Uses a smooth penalty-based formulation.
    
    x_flat: 1D array of shape (2*N,) with flattened point coordinates.
    Returns: (objective_value, gradient)
    """
    x = x_flat.reshape(N, 2)
    
    # Boundary penalty: points should be within [-0.5, 0.5]
    # Distance to boundary (positive means inside)
    dist_boundary = 0.5 - np.max(np.abs(x), axis=1)
    
    # Pairwise distances
    if N >= 2:
        d = pdist(x)
        min_pair = np.min(d)
    else:
        min_pair = float('inf')
    
    # True radius
    true_r = min(min_pair / 2.0, np.min(dist_boundary))
    
    # Smooth surrogate using log-sum-exp for min
    # For boundary: soft min of dist_boundary
    max_b = np.max(dist_boundary)
    soft_min_boundary = max_b - np.log(np.sum(np.exp(dist_boundary - max_b))) / N
    
    # For pairs: soft min of d
    if N >= 2:
        max_d = np.max(d)
        soft_min_pair = max_d - np.log(np.sum(np.exp(d - max_d))) / len(d)
        soft_min_pair_half = soft_min_pair / 2.0
    else:
        soft_min_pair_half = float('inf')
    
    # Combine: soft min of (soft_min_pair_half, soft_min_boundary)
    vals = np.array([soft_min_pair_half, soft_min_boundary])
    max_v = np.max(vals)
    soft_min_combined = max_v - np.log(np.sum(np.exp(vals - max_v))) / len(vals)
    
    # Objective: minimize negative soft min
    obj = -soft_min_combined
    
    # Gradient computation (numerical or analytical)
    # For efficiency, use numerical gradient with small step
    eps = 1e-7
    grad = np.zeros_like(x_flat)
    for i in range(len(x_flat)):
        x_plus = x_flat.copy()
        x_plus[i] += eps
        obj_plus = _objective_and_grad(x_plus)[0]
        grad[i] = (obj_plus - obj) / eps
    
    return obj, grad


def _run_gradient_optimization(initial_points, time_budget=60.0):
    """
    Run L-BFGS-B optimization to maximize the minimum radius.
    """
    start_time = time.time()
    
    x0 = initial_points.flatten()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Use L-BFGS-B with bounds
    # We'll iterate multiple times with increasing precision
    
    best_x = x0.copy()
    best_obj = _compute_radius(initial_points)
    
    # First pass: coarse optimization
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 2000}
        )
        
        if result.success or result.fun < 0:
            new_points = result.x.reshape(N, 2)
            new_r = _compute_radius(new_points)
            if new_r > best_obj:
                best_obj = new_r
                best_x = result.x.copy()
    except Exception:
        pass
    
    # Second pass: fine-tune from best result
    if time.time() - start_time < time_budget * 0.7:
        try:
            result2 = minimize(
                _objective_and_grad,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 5000}
            )
            
            if result2.success or result2.fun < 0:
                new_points = result2.x.reshape(N, 2)
                new_r = _compute_radius(new_points)
                if new_r > best_obj:
                    best_obj = new_r
                    best_x = result2.x.copy()
        except Exception:
            pass
    
    return best_x.reshape(N, 2), best_obj


def _local_refine(points, time_budget=10.0):
    """
    Local refinement using coordinate ascent with adaptive step size.
    """
    start_time = time.time()
    points = points.copy()
    current_r = _compute_radius(points)
    
    if current_r <= 0:
        return points, current_r
    
    step = 0.005
    no_improve_count = 0
    
    while time.time() - start_time < time_budget:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
            
            # Try moving point i in 8 directions + diagonal
            for dx in [-step, 0, step]:
                for dy in [-step, 0, step]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = points.copy()
                    new_points[i] += np.array([dx, dy])
                    if np.all((new_points >= -0.5) & (new_points <= 0.5)):
                        new_r = _compute_radius(new_points)
                        if new_r > current_r + 1e-12:
                            current_r = new_r
                            points = new_points
                            improved = True
                            no_improve_count = 0
                            break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= 3:
                step *= 0.5
                if step < 1e-6:
                    break
                no_improve_count = 0
    
    return points, current_r


def construct():
    deadline = time.time() + 145.0  # Safety margin (evaluator kills at 150s)
    
    best_points = None
    best_r = 0.0
    
    # Strategy 1: Generate initial hex lattice with jitter
    hex_points = _generate_hex_initial(N, jitter_scale=0.01)
    r = _compute_radius(hex_points)
    if r > best_r:
        best_r = r
        best_points = hex_points.copy()
    
    # Strategy 2: Run gradient-based optimization
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 15
        if remaining > 10:
            opt_points, opt_r = _run_gradient_optimization(best_points, time_budget=remaining)
            if opt_r > best_r:
                best_r = opt_r
                best_points = opt_points.copy()
    
    # Strategy 3: Local refinement
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 2:
            best_points, best_r = _local_refine(best_points, time_budget=remaining)
    
    return best_points
