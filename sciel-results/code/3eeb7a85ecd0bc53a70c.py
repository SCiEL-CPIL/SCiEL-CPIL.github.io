"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import itertools

def _min_triangle_area_fast(points):
    """Compute min area of any triangle formed by the n points.
    points: (n, 2) array. Returns scalar min area."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = points[j] - points[i]  # (2,)
            diff_k = points - points[i]  # (n, 2)
            cross = d[0] * diff_k[:, 1] - d[1] * diff_k[:, 0]  # (n,)
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            areas = 0.5 * np.abs(cross[mask])
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area


def _bary_to_cart(bc):
    """Convert barycentric (b,c) coordinates (n,2) to Cartesian (n,2).
    a = 1 - b - c
    x = b + 0.5*c
    y = c*sqrt(3)/2
    """
    b = bc[:, 0]
    c = bc[:, 1]
    x = b + 0.5 * c
    y = c * (np.sqrt(3) / 2)
    return np.column_stack([x, y])


def _clip_to_triangle(points):
    """Project points into the equilateral triangle."""
    x = points[:, 0].copy()
    y = points[:, 1].copy()
    
    # y bounds
    y_max = np.sqrt(3) / 2
    y = np.clip(y, 0, y_max)
    
    # x bounds based on y
    # Left edge: x >= y / sqrt(3)
    # Right edge: x <= 1 - y / sqrt(3)
    x_min = y / np.sqrt(3)
    x_max = 1 - y / np.sqrt(3)
    
    # If x is out of [x_min, x_max], project to the nearest edge
    # This is a simple clamp, which works for convex sets like a triangle
    x = np.clip(x, x_min, x_max)
    
    return np.column_stack([x, y])


def _objective_bary(bc_flat):
    """
    Objective function for SLSQP.
    Minimizes the negative of the minimum triangle area.
    Input: flat array of length 2*N, representing (b_i, c_i) for each point.
    """
    bc = bc_flat.reshape(N, 2)
    points = _bary_to_cart(bc)
    min_area = _min_triangle_area_fast(points)
    return -min_area


def _soft_min_objective(bc_flat, beta):
    """
    Smooth approximation of -min_area using log-sum-exp.
    Minimizes a smooth surrogate that approximates the negative min area.
    """
    bc = bc_flat.reshape(N, 2)
    points = _bary_to_cart(bc)
    
    n = len(points)
    # Compute all triangle areas
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                cross = (points[j][0] - points[i][0]) * (points[k][1] - points[i][1]) - \
                        (points[j][1] - points[i][1]) * (points[k][0] - points[i][0])
                area = 0.5 * abs(cross)
                areas[i, j, k] = area
                areas[k, j, i] = area
                areas[j, k, i] = area
                areas[i, k, j] = area
                areas[k, i, j] = area
                areas[j, i, k] = area
    
    # Flatten and filter out zeros (degenerate triangles)
    flat_areas = areas.flatten()
    flat_areas = flat_areas[flat_areas > 1e-12]
    
    if len(flat_areas) == 0:
        return 0.0
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # This approximates -min(areas) as beta -> inf
    max_val = np.max(flat_areas)
    exp_terms = np.exp(-beta * (flat_areas - max_val))
    soft_min = -max_val - (1.0 / beta) * np.log(np.sum(exp_terms))
    
    return -soft_min


def _simulated_annealing(points, start_temp, end_temp, n_iter, deadline):
    """
    Simulated annealing to improve the point configuration.
    """
    current_points = points.copy()
    current_score = _min_triangle_area_fast(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    rng = np.random.default_rng(123)
    
    for i in range(n_iter):
        if time.time() > deadline:
            break
        
        # Temperature schedule
        temp = start_temp * (end_temp / start_temp) ** (i / n_iter)
        
        # Generate a perturbation
        new_points = current_points.copy()
        
        # Randomly choose a few points to perturb
        n_perturb = rng.integers(1, 4)
        idxs = rng.choice(N, size=n_perturb, replace=False)
        
        for idx in idxs:
            # Small random displacement
            displacement = rng.normal(0, 0.05, size=2)
            new_points[idx] = new_points[idx] + displacement
        
        # Clip to triangle
        new_points = _clip_to_triangle(new_points)
        
        new_score = _min_triangle_area_fast(new_points)
        
        # Accept if better or with probability exp((new_score - current_score)/temp)
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_points = new_points.copy()
        else:
            prob = np.exp((new_score - current_score) / temp)
            if rng.random() < prob:
                current_points = new_points
                current_score = new_score
    
    return best_points, best_score


def _polish_with_soft_min(points, deadline):
    """
    Polish a good configuration using soft-min optimization.
    """
    bc = np.zeros((N, 2))
    for i in range(N):
        x, y = points[i]
        c = y / (np.sqrt(3) / 2)
        b = x - 0.5 * c
        bc[i, 0] = b
        bc[i, 1] = c
    
    x0 = bc.flatten()
    
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_x = x0.copy()
    best_score = _min_triangle_area_fast(points)
    
    # Try different beta values
    for beta in [10, 50, 100, 200]:
        if time.time() > deadline:
            break
        
        def obj(x):
            return _soft_min_objective(x, beta)
        
        try:
            result = minimize(
                obj,
                best_x,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            
            if result.success:
                bc_opt = result.x.reshape(N, 2)
                points_opt = _bary_to_cart(bc_opt)
                points_opt = _clip_to_triangle(points_opt)
                score = _min_triangle_area_fast(points_opt)
                
                if score > best_score:
                    best_score = score
                    best_x = result.x.copy()
                    best_points = points_opt.copy()
        except Exception:
            continue
    
    if time.time() > deadline:
        return points
    
    # Final polish with exact objective
    def obj_exact(x):
        bc = x.reshape(N, 2)
        pts = _bary_to_cart(bc)
        return -_min_triangle_area_fast(pts)
    
    try:
        result = minimize(
            obj_exact,
            best_x,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        if result.success:
            bc_opt = result.x.reshape(N, 2)
            points_opt = _bary_to_cart(bc_opt)
            points_opt = _clip_to_triangle(points_opt)
            score = _min_triangle_area_fast(points_opt)
            
            if score > best_score:
                return points_opt
    except Exception:
        pass
    
    return best_points


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (limit is 150s)
    
    rng = np.random.default_rng(42)
    best_points = None
    best_score = -float('inf')
    
    # Constraints for SLSQP in barycentric coordinates (b, c)
    constraints = []
    for i in range(N):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1]})
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*i+1]})
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Multi-start SLSQP
    n_starts = 15
    
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        # Generate random initial points in the triangle
        bc_init = np.zeros((N, 2))
        for k in range(N):
            u = rng.uniform(0, 1)
            v = rng.uniform(0, 1)
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            bc_init[k, 0] = u
            bc_init[k, 1] = v
        
        x0 = bc_init.flatten()
        
        try:
            result = minimize(
                _objective_bary,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-8}
            )
            
            if result.success:
                bc_opt = result.x.reshape(N, 2)
                points_opt = _bary_to_cart(bc_opt)
                points_opt = _clip_to_triangle(points_opt)
                
                score = _min_triangle_area_fast(points_opt)
                
                if score > best_score:
                    best_score = score
                    best_points = points_opt.copy()
                    
        except Exception:
            continue
    
    # If we have a good starting point, run simulated annealing
    if best_points is not None and time.time() < deadline - 20:
        sa_points, sa_score = _simulated_annealing(
            best_points, 
            start_temp=0.1, 
            end_temp=0.001, 
            n_iter=2000,
            deadline=deadline - 10
        )
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points
    
    # Final polish
    if best_points is not None and time.time() < deadline - 5:
        best_points = _polish_with_soft_min(best_points, deadline)
        best_score = _min_triangle_area_fast(best_points)
    
    if best_points is None:
        # Fallback: Centered grid
        h = np.sqrt(3) / 2
        best_points = np.array([
            [0.5, 0.0], [0.0, 0.0], [1.0, 0.0],
            [0.25, h/2], [0.75, h/2], [0.5, h],
            [0.5, h/4], [0.25, h/4], [0.75, h/4],
            [0.5, 3*h/4], [0.5, 2*h/4]
        ])
    
    # Final safety check
    best_points = _clip_to_triangle(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
