"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _precompute_indices():
    """Precompute indices for all C(11,3) triangles."""
    idx = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                idx.append((i, j, k))
    return np.array(idx)

TRI_INDICES = _precompute_indices()
M_TRIS = len(TRI_INDICES)

def _points_from_bary(bary_params):
    """
    Convert barycentric parameters (shape 22) to Cartesian points (shape 11, 2).
    bary_params[i] = [lambda1, lambda2] for point i.
    lambda3 = 1 - lambda1 - lambda2.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2).
    """
    h = np.sqrt(3) / 2
    # bary_params is flattened: [l1_0, l2_0, l1_1, l2_1, ...]
    l1 = bary_params[0::2]
    l2 = bary_params[1::2]
    l3 = 1.0 - l1 - l2
    
    # Cartesian: x = l1*V1_x + l2*V2_x + l3*V3_x
    # V1=(0,0), V2=(1,0), V3=(0.5, h)
    x = 1.0 * l2 + 0.5 * l3
    y = h * l3
    
    return np.column_stack((x, y))

def _min_triangle_area_vectorized(points):
    """
    Compute minimum area among all triangles using vectorization.
    points: (N, 2)
    """
    # Get points for each triangle
    # TRI_INDICES shape (M, 3)
    i_idx = TRI_INDICES[:, 0]
    j_idx = TRI_INDICES[:, 1]
    k_idx = TRI_INDICES[:, 2]
    
    p_i = points[i_idx] # (M, 2)
    p_j = points[j_idx] # (M, 2)
    p_k = points[k_idx] # (M, 2)
    
    # Vectors
    v1 = p_j - p_i
    v2 = p_k - p_i
    
    # Cross product in 2D: x1*y2 - y1*x2
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas), areas

def _smooth_min_area(areas, mu):
    """
    Log-Sum-Exp approximation of the minimum area.
    min(a) = -lim_{mu->inf} (1/mu) * log( (1/M) * sum(exp(-mu*a)) )
    """
    # For numerical stability, subtract max (which is min of -a, so max of a)
    # We are minimizing -min_area, so we want to maximize min_area.
    # Let's define the loss as -SmoothMinArea.
    # SmoothMinArea = - (1/mu) * log( sum(exp(-mu * areas)) / M )
    # To avoid overflow/underflow:
    # log_sum_exp(x) = max(x) + log(sum(exp(x - max(x))))
    # Here x = -mu * areas.
    x = -mu * areas
    x_max = np.max(x)
    # If x_max is very negative, exp(x) underflows. 
    # If areas are small, -mu*areas is large positive? No, areas > 0.
    # So x is negative.
    # exp(x) is small.
    # log(sum(exp(x))) is stable if we subtract x_max.
    
    # Note: if mu is large, x is very negative. 
    # Actually, for min, we use:
    # min(a_i) approx -1/mu * log( 1/M * sum(exp(-mu * a_i)) )
    
    # Let's just compute:
    # val = -1.0/mu * (np.log(np.sum(np.exp(x))) - np.log(M_TRIS))
    # But np.exp(x) might underflow to 0 if x is very negative.
    # We should use the stable log-sum-exp.
    
    # Since x = -mu * areas, and areas > 0, x < 0.
    # The max of x corresponds to the min of areas.
    # Let a_min = min(areas). x_max = -mu * a_min.
    # sum(exp(x)) = exp(x_max) * sum(exp(x - x_max))
    # log(sum(exp(x))) = x_max + log(sum(exp(x - x_max)))
    
    shifted = x - x_max
    sum_exp = np.sum(np.exp(shifted))
    log_sum = x_max + np.log(sum_exp)
    
    smooth_min = -1.0 / mu * (log_sum - np.log(M_TRIS))
    return smooth_min

def _objective_bary(params, mu):
    """
    Objective function for SLSQP.
    Returns negative of the smoothed minimum area.
    """
    points = _points_from_bary(params)
    min_area, areas = _min_triangle_area_vectorized(points)
    
    # If points are degenerate or invalid, return large penalty
    # But barycentric constraints ensure they are inside.
    # However, if l1+l2 > 1, they are outside. SLSQP handles constraints, 
    # but we should ensure robustness.
    
    # Check validity: l1>=0, l2>=0, l1+l2<=1
    l1 = params[0::2]
    l2 = params[1::2]
    if np.any(l1 < -1e-6) or np.any(l2 < -1e-6) or np.any(l1 + l2 > 1 + 1e-6):
        return 1000.0
        
    smooth_min = _smooth_min_area(areas, mu)
    return -smooth_min

def _constraints_bary(params):
    """
    Linear constraints for barycentric coordinates.
    g1: l1 >= 0  => l1
    g2: l2 >= 0  => l2
    g3: 1 - l1 - l2 >= 0 => 1 - l1 - l2
    """
    l1 = params[0::2]
    l2 = params[1::2]
    l3 = 1.0 - l1 - l2
    
    # Return array of constraint values, all must be >= 0
    return np.concatenate([l1, l2, l3])

def _local_polish_true(points, deadline):
    """
    Final polish using Nelder-Mead on the true min area (non-smooth).
    """
    current_min_area, _ = _min_triangle_area_vectorized(points)
    
    def true_objective(params):
        pts = _points_from_bary(params)
        # Check if inside (should be, but Nelder-Mead might step out)
        l1 = params[0::2]
        l2 = params[1::2]
        if np.any(l1 < -1e-3) or np.any(l2 < -1e-3) or np.any(l1 + l2 > 1 + 1e-3):
            return 10.0
        min_a, _ = _min_triangle_area_vectorized(pts)
        return -min_a

    try:
        result = minimize(true_objective, points_bary, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if result.fun < -current_min_area:
             # Verify validity
             l1 = result.x[0::2]
             l2 = result.x[1::2]
             if np.any(l1 < -1e-4) or np.any(l2 < -1e-4) or np.any(l1 + l2 > 1 + 1e-4):
                 pass # Discard invalid
             else:
                 new_points = _points_from_bary(result.x)
                 new_min, _ = _min_triangle_area_vectorized(new_points)
                 if new_min > current_min_area:
                     return new_points, new_min
    except Exception:
        pass
    
    return points, current_min_area

def construct():
    deadline = time.time() + 140 # 10s safety margin
    h = np.sqrt(3) / 2
    
    # Initial guess: Vertices, Midpoints, Centroid, etc.
    # Let's use a structured initial guess in Cartesian, then convert to Barycentric
    # Or just random barycentric.
    
    # Better: Start with a known good configuration if possible, or random.
    # Let's try a few random starts and pick the best after a quick DE phase.
    
    best_points = None
    best_min_area = 0.0
    
    # Phase 1: Global Search with Differential Evolution on Barycentric Coordinates
    # DE doesn't handle constraints well natively, but we can use penalty or just bounds.
    # Bounds for l1: [0, 1], l2: [0, 1]. Constraint l1+l2<=1 is tricky for DE.
    # Instead, use DE on Cartesian, then project? No, projection breaks area.
    # Let's use DE on Barycentric with a penalty for l1+l2>1.
    
    def de_objective(params):
        l1 = params[0::2]
        l2 = params[1::2]
        # Penalty for invalid barycentric
        if np.any(l1 < 0) or np.any(l2 < 0) or np.any(l1 + l2 > 1):
            return 10.0
        points = _points_from_bary(params)
        min_a, _ = _min_triangle_area_vectorized(points)
        return -min_a

    bounds_de = [(0.0, 1.0)] * (2 * N)
    
    try:
        if time.time() < deadline - 60:
            de_result = differential_evolution(
                de_objective, bounds_de,
                seed=42, maxiter=50, popsize=15,
                tol=1e-5, mutation=(0.5, 1.5), recombination=0.7,
                polish=False # We will polish later
            )
            if de_result.fun < 10.0:
                best_bary = de_result.x
                best_points = _points_from_bary(best_bary)
                best_min_area, _ = _min_triangle_area_vectorized(best_points)
    except Exception:
        pass

    # If DE failed or didn't find anything, initialize randomly
    if best_points is None:
        rng = np.random.RandomState(42)
        l1 = rng.random(N)
        l2 = rng.random(N)
        # Ensure l1+l2 <= 1
        l2 = l2 * (1.0 - l1) * 0.99 # Scale down to ensure inside
        best_bary = np.concatenate([l1, l2]).flatten()
        best_points = _points_from_bary(best_bary)
        best_min_area, _ = _min_triangle_area_vectorized(best_points)

    # Phase 2: Smoothed Optimization with SLSQP (Annealing mu)
    # Start from best_bary
    current_bary = best_bary.copy()
    
    mu_values = [5, 10, 20, 50, 100]
    
    for mu in mu_values:
        if time.time() > deadline - 20:
            break
            
        # Define objective and constraints for current mu
        def obj_func(params):
            return _objective_bary(params, mu)
            
        def con_func(params):
            return _constraints_bary(params)
        
        constraints = {'type': 'ineq', 'fun': con_func}
        
        try:
            res = minimize(
                obj_func, current_bary,
                method='SLSQP',
                constraints=constraints,
                bounds=[(0.0, 1.0)] * (2 * N), # l1, l2 in [0,1]
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.fun < 1000.0: # Check if valid
                new_bary = res.x
                new_points = _points_from_bary(new_bary)
                new_min_area, _ = _min_triangle_area_vectorized(new_points)
                
                if new_min_area > best_min_area:
                    best_min_area = new_min_area
                    best_points = new_points
                    current_bary = new_bary
        except Exception:
            pass

    # Phase 3: Final Polish with True Objective
    if time.time() < deadline - 5:
        best_points, best_min_area = _local_polish_true(best_points, deadline)
        
    return best_points
# EVOLVE-BLOCK-END
