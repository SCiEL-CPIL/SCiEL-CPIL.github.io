"""Rewritten program for packing 200 equal circles in a unit square."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return min_boundary
    
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    
    return min(min_pair_dist / 2.0, min_boundary)

def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta

def _smooth_objective(params, beta):
    """Smooth objective: minimize -soft_min(radii_components)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if n < 2:
        return -min_boundary
    
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    
    all_dists = np.concatenate([pair_dists_half, boundary_dists.ravel()])
    r_soft = _soft_min(all_dists, beta)
    
    if r_soft <= 0:
        return 1.0
    return -r_soft

def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 20):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    
    if best_c is not None:
        return best_c
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()

def _get_active_constraints(c, r, tol_factor=1.05):
    """
    Identify active constraints for SLSQP.
    Active pair: dist < 2 * r * tol_factor
    Active boundary: dist_to_boundary < r * tol_factor
    """
    n = len(c)
    pair_indices = []
    boundary_indices = []
    
    # Pairwise
    dists = pdist(c, metric='euclidean')
    # pdist returns condensed form. We need to map back to (i, j)
    # For N=200, number of pairs is ~20k. This is fast enough.
    # However, we can use a spatial grid if N was larger.
    # For now, just iterate.
    
    # Optimization: Only check if dist is close to 2r
    # We can't easily filter pdist output without mapping.
    # Let's just compute all and filter.
    
    # Actually, for SLSQP, we want a *small* number of constraints.
    # If we pass 20k constraints, SLSQP will be very slow.
    # We need to be strict.
    
    threshold = 2.0 * r * tol_factor
    
    # Get indices of pairs closer than threshold
    # pdist doesn't return indices.
    # We can use scipy.spatial.distance.squareform to get full matrix, but that's O(N^2) memory.
    # N=200 -> 40k floats. Fine.
    
    dist_matrix = squareform(dists)
    
    # Find pairs where dist < threshold and i < j
    # Use np.where
    i_idx, j_idx = np.where(dist_matrix < threshold)
    # Filter i < j
    mask = i_idx < j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    
    pair_indices = list(zip(i_idx, j_idx))
    
    # Boundary
    boundary_dists = 0.5 - np.abs(c)
    # boundary_dists shape (N, 2)
    min_boundary_per_point = np.min(boundary_dists, axis=1)
    
    boundary_threshold = r * tol_factor
    b_idx = np.where(min_boundary_per_point < boundary_threshold)[0]
    boundary_indices = list(b_idx)
    
    return pair_indices, boundary_indices

from scipy.spatial.distance import squareform

def _slsqp_polish(c0, max_time=20.0):
    """
    Polish the solution using SLSQP with sparse active constraints.
    """
    start_time = time.time()
    c = c0.copy()
    r = _compute_radius(c)
    best_c = c.copy()
    best_r = r
    
    # SLSQP variables: [x1, y1, ..., xN, yN, r]
    # Total size: 2*N + 1
    
    bounds = [(-0.5, 0.5)] * (2 * N) + [(0.0, 0.5)]
    
    # Number of SLSQP iterations
    # Each iteration involves a full SLSQP solve which can be expensive.
    # We limit the number of outer loops.
    max_outer_iters = 20 
    
    for k in range(max_outer_iters):
        if time.time() > start_time + max_time - 2.0:
            break
            
        pair_indices, boundary_indices = _get_active_constraints(c, r, tol_factor=1.02)
        
        # If no active constraints, we can't improve r via SLSQP easily 
        # (it would just push everything to center? No, bounds prevent that).
        # If no constraints, SLSQP will try to maximize r indefinitely until bounds hit.
        # But bounds are on c, not r directly (other than 0.5).
        # Actually, if no constraints are active, the current r is determined by the *next* closest pair/boundary.
        # SLSQP needs gradients. If we don't include the *next* closest, it might move away from them.
        # So we should include a slightly larger tolerance to capture the "next" closest.
        
        if len(pair_indices) == 0 and len(boundary_indices) == 0:
             # Fallback: use a slightly larger tolerance
             pair_indices, boundary_indices = _get_active_constraints(c, r, tol_factor=1.1)
             if len(pair_indices) == 0 and len(boundary_indices) == 0:
                 break

        def objective(params):
            # Minimize -r
            return -params[-1]
        
        def objective_grad(params):
            grad = np.zeros_like(params)
            grad[-1] = -1.0
            return grad
            
        constraints = []
        
        # Pair constraints: ||c_i - c_j|| - 2r >= 0
        for i, j in pair_indices:
            def make_pair_con(i, j):
                def con(params):
                    ci = params[2*i:2*i+2]
                    cj = params[2*j:2*j+2]
                    r_val = params[-1]
                    dist = np.linalg.norm(ci - cj)
                    return dist - 2.0 * r_val
                
                def con_grad(params):
                    ci = params[2*i:2*i+2]
                    cj = params[2*j:2*j+2]
                    r_val = params[-1]
                    
                    diff = ci - cj
                    dist = np.linalg.norm(diff)
                    if dist < 1e-12:
                        # Avoid division by zero
                        grad_c = np.zeros(2*N+1)
                        grad_c[-1] = -2.0
                        return grad_c
                        
                    grad_c = np.zeros(2*N+1)
                    # d(dist)/d(ci) = diff / dist
                    grad_c[2*i:2*i+2] = diff / dist
                    # d(dist)/d(cj) = -diff / dist
                    grad_c[2*j:2*j+2] = -diff / dist
                    # d(-2r)/dr = -2
                    grad_c[-1] = -2.0
                    return grad_c
                
                return {'type': 'ineq', 'fun': con, 'jac': con_grad}
            
            constraints.append(make_pair_con(i, j))
            
        # Boundary constraints: 0.5 - |c_ik| - r >= 0
        # This is non-smooth at c_ik = 0.
        # To make it smooth, we can use:
        # (0.5 - x)^2 - r^2 >= 0  => 0.5 - x >= r (for x < 0.5)
        # But SLSQP handles non-smoothness poorly.
        # Better: Use the fact that we are near the boundary.
        # If c_ik > 0, constraint is 0.5 - c_ik - r >= 0.
        # If c_ik < 0, constraint is 0.5 + c_ik - r >= 0.
        # We can split into 4 constraints per point (left, right, bottom, top) but only one is active.
        # Or we can just use the absolute value and hope SLSQP's BFGS approximation handles it,
        # or we can smooth it: sqrt((0.5-x)^2 + eps) - r >= 0? No, that changes the geometry.
        
        # Standard trick: Use squared distance to boundary?
        # (0.5 - |x|)^2 - r^2 >= 0.
        # Gradient: 2(0.5 - |x|) * (-sign(x)) - 2r * 0 (if r is separate var? No, r is var)
        # Let's stick to the linear form but split by sign.
        # For each point k, we check which side it is close to.
        
        for k in boundary_indices:
            x, y = c[k]
            
            # Determine active sides
            # Left: x is close to -0.5 -> 0.5 + x - r >= 0
            # Right: x is close to 0.5 -> 0.5 - x - r >= 0
            # Bottom: y is close to -0.5 -> 0.5 + y - r >= 0
            # Top: y is close to 0.5 -> 0.5 - y - r >= 0
            
            # We add constraints for all sides that are "close"
            # Close means distance < r * tol_factor
            
            dist_left = x + 0.5
            dist_right = 0.5 - x
            dist_bottom = y + 0.5
            dist_top = 0.5 - y
            
            tol = r * 1.02
            
            if dist_left < tol:
                def make_left_con(k):
                    def con(params):
                        x = params[2*k]
                        r_val = params[-1]
                        return 0.5 + x - r_val
                    def con_grad(params):
                        grad = np.zeros(2*N+1)
                        grad[2*k] = 1.0
                        grad[-1] = -1.0
                        return grad
                    return {'type': 'ineq', 'fun': con, 'jac': con_grad}
                constraints.append(make_left_con(k))
                
            if dist_right < tol:
                def make_right_con(k):
                    def con(params):
                        x = params[2*k]
                        r_val = params[-1]
                        return 0.5 - x - r_val
                    def con_grad(params):
                        grad = np.zeros(2*N+1)
                        grad[2*k] = -1.0
                        grad[-1] = -1.0
                        return grad
                    return {'type': 'ineq', 'fun': con, 'jac': con_grad}
                constraints.append(make_right_con(k))
                
            if dist_bottom < tol:
                def make_bottom_con(k):
                    def con(params):
                        y = params[2*k+1]
                        r_val = params[-1]
                        return 0.5 + y - r_val
                    def con_grad(params):
                        grad = np.zeros(2*N+1)
                        grad[2*k+1] = 1.0
                        grad[-1] = -1.0
                        return grad
                    return {'type': 'ineq', 'fun': con, 'jac': con_grad}
                constraints.append(make_bottom_con(k))
                
            if dist_top < tol:
                def make_top_con(k):
                    def con(params):
                        y = params[2*k+1]
                        r_val = params[-1]
                        return 0.5 - y - r_val
                    def con_grad(params):
                        grad = np.zeros(2*N+1)
                        grad[2*k+1] = -1.0
                        grad[-1] = -1.0
                        return grad
                    return {'type': 'ineq', 'fun': con, 'jac': con_grad}
                constraints.append(make_top_con(k))
        
        if not constraints:
            break
            
        # Prepare initial guess
        x0 = np.zeros(2*N + 1)
        x0[:2*N] = c.flatten()
        x0[-1] = r
        
        try:
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                jac=objective_grad,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 100, 'ftol': 1e-12, 'disp': False}
            )
            
            if result.success:
                new_c = result.x[:2*N].reshape(-1, 2)
                new_r = result.x[-1]
                
                # Clip to ensure validity
                new_c = np.clip(new_c, -0.5, 0.5)
                new_r = _compute_radius(new_c)
                
                if new_r > best_r:
                    best_r = new_r
                    best_c = new_c.copy()
                    c = new_c.copy()
                    r = new_r
                else:
                    # If SLSQP didn't improve, break to avoid wasting time
                    # or try again with looser tolerance?
                    # For now, break.
                    break
            else:
                # SLSQP failed, try to salvage by taking the best point found?
                # result.x might be valid.
                new_c = result.x[:2*N].reshape(-1, 2)
                new_c = np.clip(new_c, -0.5, 0.5)
                new_r = _compute_radius(new_c)
                if new_r > best_r:
                    best_r = new_r
                    best_c = new_c.copy()
                    c = new_c.copy()
                    r = new_r
                else:
                    break
                    
        except Exception as e:
            # If SLSQP crashes, break
            break
            
    return best_c, best_r


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    betas = [15.0, 30.0, 60.0, 120.0, 250.0, 500.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 5:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': 120, 'ftol': 1e-13, 'gtol': 1e-10}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
            
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_radius = 0.0
    best_c = None
    
    # 1. Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    if r_hex > best_radius:
        best_radius = r_hex
        best_c = c_hex.copy()
        
    # 2. Smooth Optimization (Warm-up)
    if time.time() < deadline - 60:
        remaining = deadline - time.time() - 60
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=min(40.0, remaining))
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
            
    # 3. SLSQP Polish (High Precision)
    if best_c is not None and time.time() < deadline - 10:
        remaining = min(30.0, deadline - time.time() - 10)
        c_polished, r_polished = _slsqp_polish(best_c, max_time=remaining)
        if r_polished > best_radius:
            best_radius = r_polished
            best_c = c_polished.copy()
            
    # 4. Final Safety Polish
    if best_c is not None and time.time() < deadline - 2:
        remaining = max(2.0, deadline - time.time() - 2)
        # One more quick SLSQP pass
        c_final, r_final = _slsqp_polish(best_c, max_time=remaining)
        if r_final > best_radius:
            best_radius = r_final
            best_c = c_final
    
    if best_c is None:
        best_c = _initialize_hexagonal()
        
    best_c = np.clip(best_c, -0.5, 0.5)
    return best_c
# EVOLVE-BLOCK-END
