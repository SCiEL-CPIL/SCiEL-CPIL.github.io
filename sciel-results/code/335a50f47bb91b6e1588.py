"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.spatial import ConvexHull
import itertools

def _convex_hull_area(points):
    """Compute the area of the convex hull of a set of points."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: use polygon area
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(points)
            return hull.volume
        except:
            return 0.0

def _min_triangle_area(points):
    """Compute the minimum area over all triples of points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                # Area of triangle = 0.5 * |cross product|
                cross = np.cross(p2 - p1, p3 - p1)
                area = 0.5 * abs(cross)
                if area < min_area:
                    min_area = area
    return min_area

def _objective(x, beta=50.0, multi_beta=False):
    """
    Objective function to minimize. We want to maximize the minimum triangle area
    relative to the convex hull area. So we minimize the negative of that.
    
    Using a soft formulation: min over triples of (area / hull_area)
    We approximate min with a soft-min using log-sum-exp or power mean.
    """
    points = x.reshape(-1, 2)
    
    # Compute convex hull area
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-10:
        return 1e10  # Penalize degenerate hulls
    
    # Compute all triangle areas
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1 = points[i]
                p2 = points[j]
                p3 = points[k]
                cross = np.cross(p2 - p1, p3 - p1)
                area = 0.5 * abs(cross)
                areas.append(area)
    
    areas = np.array(areas)
    
    # Normalize by hull area
    normalized_areas = areas / hull_area
    
    if multi_beta:
        # Average over multiple beta values for smoother landscape
        betas = [20.0, 50.0, 100.0]
        soft_min_sum = 0.0
        for b in betas:
            # Soft minimum using power mean approximation
            # min(x) ≈ (1/m) * sum(x_i^(-p))^(1/(-p)) for large p
            # Or use log-sum-exp style: min ≈ -log(sum(exp(-x/b)))/(-1/b) ... 
            # Actually, soft-min: -log(sum(exp(-x_i / beta))) * beta
            # But this can have numerical issues. Let's use:
            # soft_min = (sum(x_i^(-beta)))^(-1/beta)
            # For large beta, this approaches min(x_i)
            try:
                # Use a stable computation
                neg_areas = -normalized_areas
                max_neg = np.max(neg_areas)
                log_sum = np.log(np.sum(np.exp(neg_areas - max_neg))) + max_neg
                soft_min_val = -beta * log_sum
                soft_min_sum += soft_min_val
            except:
                soft_min_sum += np.min(normalized_areas)
        soft_min = soft_min_sum / len(betas)
    else:
        # Single beta soft minimum
        try:
            neg_areas = -normalized_areas
            max_neg = np.max(neg_areas)
            log_sum = np.log(np.sum(np.exp(neg_areas - max_neg))) + max_neg
            soft_min = -beta * log_sum
        except:
            soft_min = np.min(normalized_areas)
    
    return -soft_min  # We minimize, so negate

def _score(points):
    """Compute the score: min triangle area / convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area <= 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    # Initial configurations
    rng = np.random.RandomState(42)
    
    # Try several initial configurations
    initial_configs = []
    
    # 1. Random points in unit square
    initial_configs.append(rng.rand(N, 2))
    
    # 2. Points on a circle (regular polygon)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    circle_points = np.column_stack([np.cos(angles), np.sin(angles)])
    initial_configs.append(circle_points)
    
    # 3. Points on a grid-like pattern
    grid_points = []
    for i in range(N):
        x = (i % 4) / 3.0
        y = (i // 4) / 3.0
        grid_points.append([x, y])
    initial_configs.append(np.array(grid_points))
    
    # 4. Random with some structure
    for _ in range(5):
        initial_configs.append(rng.rand(N, 2) * 0.8 + 0.1)
    
    # Optimize each initial configuration
    for init_idx, init_points in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        x0 = init_points.flatten()
        
        # Multi-start with different beta values
        betas_to_try = [10.0, 30.0, 100.0]
        
        for beta in betas_to_try:
            if time.time() > deadline:
                break
            
            # First pass with single beta
            try:
                result = minimize(
                    lambda x: _objective(x, beta=beta, multi_beta=False),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < 0:
                    points = result.x.reshape(-1, 2)
                    score = _score(points)
                    
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        
            except Exception as e:
                pass
        
        # Second pass with multi-beta polishing
        if time.time() > deadline:
            break
        
        if best_points is not None and init_idx >= 0:
            # Use the best found so far as starting point for multi-beta
            x0 = best_points.flatten()
            try:
                result = minimize(
                    lambda x: _objective(x, beta=50.0, multi_beta=True),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.success or result.fun < 0:
                    points = result.x.reshape(-1, 2)
                    score = _score(points)
                    
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        
            except Exception as e:
                pass
    
    # If we still have time, do additional local refinements
    if best_points is not None and time.time() < deadline - 5:
        # Try perturbing the best solution and re-optimizing
        for _ in range(3):
            if time.time() > deadline:
                break
            
            perturbation = rng.randn(N, 2) * 0.05
            x0 = (best_points + perturbation).flatten()
            
            try:
                result = minimize(
                    lambda x: _objective(x, beta=50.0, multi_beta=True),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.success or result.fun < 0:
                    points = result.x.reshape(-1, 2)
                    score = _score(points)
                    
                    if score > best_score:
                        best_score = score
                        best_points = points.copy()
                        
            except Exception as e:
                pass
    
    if best_points is None:
        # Fallback to random
        return np.array([[rng.random(), rng.random()] for _ in range(N)])
    
    return best_points
# EVOLVE-BLOCK-END
