"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _score(c):
    """Compute the largest common radius for centres c (shape (N,3))."""
    if c.shape != (N, 3):
        return -1.0
    if not np.all(np.isfinite(c)):
        return -1.0
    # distance to walls
    # c is in [-0.5, 0.5]
    # wall dist for point (x,y,z) is min(0.5-|x|, 0.5-|y|, 0.5-|z|)
    # Actually, the distance to the boundary of the cube [-0.5, 0.5]^3 is min(0.5 - x, 0.5 + x, ...)
    # which is min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|)
    abs_c = np.abs(c)
    wall_dists = 0.5 - abs_c
    min_wall = np.min(wall_dists, axis=1)
    min_wall_global = np.min(min_wall)
    
    # pairwise distances
    # Use vectorized calculation
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    return min(min_wall_global, min_dist / 2.0)

def _init_grid():
    """Simple cubic grid as starting point."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _penalty_objective_and_grad(x, r_target, beta):
    """
    Objective: minimize penalty for violations.
    Variables: x is flattened (N*3).
    Constraints:
    1. Pairwise: d_ij >= 2 * r_target
    2. Wall: dist_to_wall_i >= r_target
    
    Penalty: sum over all pairs max(0, 2*r_target - d_ij)^2 
             + sum over all points max(0, r_target - dist_to_wall_i)^2
             
    We use a soft violation function to make it differentiable.
    Let v_ij = 2*r_target - d_ij.
    Penalty term is v_ij^2 if v_ij > 0, else 0.
    To make it smooth, we can use a smooth approximation or just rely on L-BFGS-B 
    handling the kink? L-BFGS-B prefers smoothness.
    Actually, the standard penalty method uses a smooth function.
    Let's use: P(v) = v^2 * sigmoid(k*v) ? No, that's complex.
    
    Alternative: Minimize the maximum violation? That's non-smooth.
    
    Let's stick to the sum of squared violations, but only for active sets?
    If we include all pairs, the gradient is dense and expensive.
    However, for N=100, pairs are ~5000. That's manageable.
    
    Let's define the objective as:
    f(x) = sum_{i<j} max(0, 2*r - d_ij)^2 + sum_i max(0, r - w_i)^2
    
    The gradient of max(0, z)^2 is 2*z * dz/dx if z>0, else 0.
    This is continuous but not differentiable at z=0. L-BFGS-B can often handle this,
    but it might struggle.
    
    Better approach: Use a smooth approximation for the "max" part?
    Or, just use the raw squared distance difference?
    If we minimize sum (2*r - d_ij)^2, we are minimizing the variance of distances?
    No, that pulls everything to 2*r. If some are < 2*r and some > 2*r, it balances.
    But we want ALL to be >= 2*r.
    
    Let's use the penalty: sum_{i<j} (max(0, 2*r - d_ij))^2.
    This is 0 if all constraints are satisfied.
    We want to minimize this to 0.
    
    Gradient calculation:
    d_ij = ||c_i - c_j||
    d(d_ij)/d(c_i) = (c_i - c_j) / d_ij
    d(d_ij)/d(c_j) = (c_j - c_i) / d_ij
    
    Let v_ij = 2*r - d_ij.
    If v_ij > 0:
       d(v_ij^2)/d(c_i) = 2 * v_ij * d(v_ij)/d(c_i) = 2 * v_ij * (- (c_i - c_j) / d_ij)
       d(v_ij^2)/d(c_j) = 2 * v_ij * d(v_ij)/d(c_j) = 2 * v_ij * ( (c_i - c_j) / d_ij)
    Else:
       0
    
    Wall constraint:
    w_i = min(0.5 - |x|, 0.5 - |y|, 0.5 - |z|)
    Let's assume the binding wall is the one with min distance.
    For a point (x,y,z), the distance to the boundary is min(0.5-x, 0.5+x, 0.5-y, 0.5+y, 0.5-z, 0.5+z).
    Let w_i be this min.
    Constraint: w_i >= r.
    Violation: v_wall_i = r - w_i.
    If v_wall_i > 0:
       Penalty: v_wall_i^2.
       Gradient depends on which wall is binding.
       If x > 0, binding wall is x=0.5. w_i = 0.5 - x.
       d(w_i)/dx = -1.
       d(v_wall_i)/dx = -(-1) = 1.
       d(v_wall_i^2)/dx = 2 * v_wall_i * 1 = 2 * v_wall_i.
       If x < 0, binding wall is x=-0.5. w_i = 0.5 + x.
       d(w_i)/dx = 1.
       d(v_wall_i)/dx = -1.
       d(v_wall_i^2)/dx = 2 * v_wall_i * (-1) = -2 * v_wall_i.
       
    This logic needs to be implemented carefully in the gradient function.
    """
    c = x.reshape(N, 3)
    
    # Pairwise
    # Compute all pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Violations
    v_pairs = 2.0 * r_target - dist
    
    # Only consider positive violations for penalty
    # Mask
    mask_pairs = v_pairs > 0
    
    # Objective
    obj = 0.0
    if np.any(mask_pairs):
        obj += np.sum(v_pairs[mask_pairs] ** 2)
        
    # Wall
    abs_c = np.abs(c)
    # Dist to each of 6 walls
    # x: 0.5 - x, 0.5 + x
    # y: 0.5 - y, 0.5 + y
    # z: 0.5 - z, 0.5 + z
    walls = np.stack([
        0.5 - c[:, 0], 0.5 + c[:, 0],
        0.5 - c[:, 1], 0.5 + c[:, 1],
        0.5 - c[:, 2], 0.5 + c[:, 2]
    ], axis=1)
    
    min_wall_dist = np.min(walls, axis=1)
    # Index of min wall
    min_wall_idx = np.argmin(walls, axis=1)
    
    v_walls = r_target - min_wall_dist
    mask_walls = v_walls > 0
    
    if np.any(mask_walls):
        obj += np.sum(v_walls[mask_walls] ** 2)
        
    # Gradient
    grad = np.zeros_like(c)
    
    # Pairwise Gradient
    if np.any(mask_pairs):
        # We need to accumulate gradients
        # v_ij > 0
        # d(v_ij^2)/d(c_i) = -2 * v_ij * (c_i - c_j) / d_ij
        # d(v_ij^2)/d(c_j) =  2 * v_ij * (c_i - c_j) / d_ij
        
        # Create a matrix of gradients for each pair
        # This is O(N^2) memory if we do it all at once, which is fine for N=100.
        # coeff_ij = -2 * v_ij / d_ij for c_i, and +2 * v_ij / d_ij for c_j
        
        # dist is (N, N)
        # v_pairs is (N, N)
        
        # We only care about upper triangle to avoid double counting?
        # The objective sum is over all i<j.
        # In the matrix, we have both (i,j) and (j,i).
        # Let's sum over all i != j and divide by 2?
        # Or just iterate?
        # Vectorized accumulation:
        
        # coeff for c_i in pair (i,j) is -2 * v_ij / d_ij
        # coeff for c_j in pair (i,j) is  2 * v_ij / d_ij
        
        # Let's compute the force vector for each particle i.
        # F_i = sum_{j != i} d(v_ij^2)/d(c_i)
        # For a specific j, the term is -2 * v_ij * (c_i - c_j) / d_ij
        # Note: v_ij = v_ji. d_ij = d_ji.
        # So term for i from j is -2 * v_ij * (c_i - c_j) / d_ij.
        
        # Let's compute a matrix M where M[i,j] = -2 * v_ij / d_ij if v_ij > 0 else 0.
        # Then grad_i = sum_j M[i,j] * (c_i - c_j)
        
        # Safe division
        dist_safe = np.where(dist > 1e-8, dist, 1e-8)
        
        # Masked v_pairs
        v_masked = np.where(mask_pairs, v_pairs, 0.0)
        
        # Coefficient matrix
        # M[i,j] = -2 * v_masked[i,j] / dist_safe[i,j]
        M = -2.0 * v_masked / dist_safe
        
        # grad_i = sum_j M[i,j] * (c_i - c_j)
        # c_i is c[i, :], c_j is c[j, :]
        # (c_i - c_j) is diff[i, j]
        
        # grad[i] = sum_j M[i, j] * diff[i, j]
        
        # M is (N, N), diff is (N, N, 3)
        # We want to multiply M[i,j] with diff[i,j,k] and sum over j.
        # grad[:, k] = sum_j M[:, j] * diff[:, j, k]
        
        # Use einsum or simple multiplication
        # M[:, :, np.newaxis] * diff
        weighted_diff = M[:, :, np.newaxis] * diff
        grad += np.sum(weighted_diff, axis=1)
        
    # Wall Gradient
    if np.any(mask_walls):
        # For each i where v_walls[i] > 0
        # grad_i = 2 * v_walls[i] * d(v_walls[i])/d(c_i)
        # v_walls[i] = r - min_wall_dist[i]
        # d(v_walls[i])/d(c_i) = - d(min_wall_dist[i])/d(c_i)
        
        # We need d(min_wall_dist[i])/d(c_i)
        # This is -1 for the coordinate of the binding wall (if wall is at +0.5 and x>0, dist=0.5-x, d/dx=-1, so d(v)/dx = 1)
        # Wait.
        # If binding wall is x=0.5 (x>0), w = 0.5 - x. dw/dx = -1.
        # v = r - w. dv/dx = -dw/dx = 1.
        # grad_x = 2 * v * 1 = 2v.
        
        # If binding wall is x=-0.5 (x<0), w = 0.5 + x. dw/dx = 1.
        # v = r - w. dv/dx = -1.
        # grad_x = 2 * v * (-1) = -2v.
        
        # So the sign is opposite to the sign of the coordinate?
        # If x > 0, binding wall is usually +x side?
        # Not necessarily. If x is close to 0, it could be either.
        # But min_wall_idx tells us which wall.
        
        # Wall indices:
        # 0: x=0.5 (dist 0.5-x) -> d/dx = -1 -> d(v)/dx = 1
        # 1: x=-0.5 (dist 0.5+x) -> d/dx = 1  -> d(v)/dx = -1
        # 2: y=0.5 (dist 0.5-y) -> d/dy = -1 -> d(v)/dy = 1
        # 3: y=-0.5 (dist 0.5+y) -> d/dy = 1  -> d(v)/dy = -1
        # 4: z=0.5 (dist 0.5-z) -> d/dz = -1 -> d(v)/dz = 1
        # 5: z=-0.5 (dist 0.5+z) -> d/dz = 1  -> d(v)/dz = -1
        
        # So d(v)/d(coord) is +1 for even indices (0,2,4) and -1 for odd indices (1,3,5).
        # Let's verify:
        # idx 0 (x>0 side): w=0.5-x. v=r-w. dv/dx = 1. Correct.
        # idx 1 (x<0 side): w=0.5+x. v=r-w. dv/dx = -1. Correct.
        
        # So we can construct a matrix of signs.
        # sign_mat[i, coord] = 1 if min_wall_idx[i] is even and matches coord, -1 if odd and matches coord, else 0.
        
        # Let's just loop over the active walls for simplicity, N is small.
        for i in np.where(mask_walls)[0]:
            v = v_walls[i]
            if v <= 0: continue
            idx = min_wall_idx[i]
            # Determine coordinate and sign
            coord = idx // 2
            # 0:x, 1:y, 2:z
            # Sign: +1 for even idx (0,2,4), -1 for odd idx (1,3,5)
            sign = 1.0 if idx % 2 == 0 else -1.0
            
            # grad[i, coord] += 2 * v * sign
            grad[i, coord] += 2.0 * v * sign
            
    return obj, grad.flatten()

def _optimize_radius(c0, r0, time_limit):
    """
    Iteratively increase r and minimize penalty.
    """
    c = c0.copy()
    r = r0
    beta = 1.0 # Penalty weight? We are just minimizing the penalty sum.
    
    best_c = c.copy()
    best_r = r
    best_score = _score(c)
    
    start_time = time.time()
    
    # We want to maximize r.
    # Strategy:
    # 1. Start with r = current score.
    # 2. Try to increase r slightly.
    # 3. Minimize penalty for this new r.
    # 4. If penalty is close to 0, accept and increase r more.
    # 5. If penalty is high, revert or reduce r increment.
    
    r_increment = 0.001
    max_iter = 100
    
    for _ in range(max_iter):
        if time.time() > time_limit:
            break
            
        # Try to increase r
        r_try = r + r_increment
        
        # Check if r_try is physically possible (max possible is 0.5)
        if r_try > 0.5:
            break
            
        # Define objective for this r_try
        def obj_grad(x):
            obj, grad = _penalty_objective_and_grad(x, r_try, 1.0)
            return obj, grad
            
        # Initial point is current c
        # L-BFGS-B
        try:
            res = minimize(obj_grad, c.flatten(), jac=True, method='L-BFGS-B', 
                           options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-5, 'disp': False})
            
            c_new = res.x.reshape(N, 3)
            
            # Check if constraints are satisfied
            # Score of new config
            sc_new = _score(c_new)
            
            # If the score is at least r_try (or close), we can accept
            # Actually, if penalty is 0, score >= r_try.
            # If penalty > 0, score < r_try.
            
            if sc_new >= r_try - 1e-5:
                # Success
                c = c_new
                r = r_try
                best_c = c.copy()
                best_r = r
                best_score = sc_new
                # Increase increment slightly?
                # r_increment *= 1.1
            else:
                # Failed to satisfy r_try
                # Maybe reduce increment
                r_increment *= 0.8
                # If we can't even make small progress, break?
                if r_increment < 1e-5:
                    break
                    
        except Exception as e:
            # print(e)
            break
            
    return best_c, best_score

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin for 150s limit
    
    # Start with a good grid initialization
    c0 = _init_grid()
    r0 = _score(c0)
    
    best_c = c0.copy()
    best_score = r0
    
    # Run the penalty optimization
    c_opt, sc_opt = _optimize_radius(c0, r0, deadline - 5.0)
    
    if sc_opt > best_score:
        best_score = sc_opt
        best_c = c_opt
    
    # If we have time, try a random start
    if time.time() < deadline - 15.0:
        np.random.seed(42) # Reproducibility for testing? Or random?
        c_rand = np.random.uniform(-0.5, 0.5, size=(N, 3))
        # Grid is usually better start. Let's try a perturbed grid.
        c_pert = c0 + np.random.normal(0, 0.01, size=(N, 3))
        c_pert = np.clip(c_pert, -0.5, 0.5)
        
        remaining = deadline - 5.0 - time.time()
        if remaining > 10.0:
            c_opt2, sc_opt2 = _optimize_radius(c_pert, _score(c_pert), remaining)
            if sc_opt2 > best_score:
                best_score = sc_opt2
                best_c = c_opt2
                
    return best_c
# EVOLVE-BLOCK-END
