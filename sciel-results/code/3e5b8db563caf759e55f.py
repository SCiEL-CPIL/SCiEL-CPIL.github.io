"""
Circle packing in unit square - 26 circles maximizing sum of radii.
Strategy: Alternating Optimization with Exact Linear Programming for Radii.
1. Initialize with diverse starting points (hexagonal, grid, corners).
2. Optimize positions using L-BFGS-B on a smooth objective with soft overlap penalties.
3. Optimize radii using Linear Programming (scipy.optimize.linprog) given fixed positions.
4. Alternate between position and radius optimization until convergence or time limit.
5. This approach ensures that for any given position configuration, the radii are maximally 
   efficient (optimal for that graph), avoiding the "slack" often left by penalty-based methods.
"""
import numpy as np
import time
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import cdist

N = 26

# EVOLVE-BLOCK-START

def _is_valid(pos, radii, tol=1e-7):
    """Strict validity check."""
    if np.any(pos[:, 0] - radii < -tol) or np.any(pos[:, 0] + radii > 1.0 + tol):
        return False
    if np.any(pos[:, 1] - radii < -tol) or np.any(pos[:, 1] + radii > 1.0 + tol):
        return False
    dist = cdist(pos, pos)
    min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    if np.any(dist[mask] < (min_dist[mask] - tol)):
        return False
    return True

def _post_process(pos, radii):
    """Shrink radii to ensure strict validity."""
    pos = pos.copy()
    radii = radii.copy()
    for _ in range(100):
        valid = True
        # Boundary check
        for i in range(N):
            r = radii[i]
            x, y = pos[i]
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            if r > max_r:
                radii[i] = max(max_r * 0.9999, 1e-5)
                valid = False
        # Overlap check
        dist = cdist(pos, pos)
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        violations = min_dist[mask] - dist[mask]
        if np.any(violations > 0):
            # Find the pair with max violation and shrink
            # Simple approach: shrink all radii slightly proportional to violation
            # More efficient: shrink specific pairs
            idx_i, idx_j = np.unravel_index(np.argmax(violations), mask.shape)
            # Actually, let's just shrink all radii by a small factor if any violation
            # Or better, shrink the specific pair involved in max violation
            # For robustness in post-processing, a global small shrink is safer
            max_viol = np.max(violations)
            if max_viol > 0:
                # Shrink all radii by 1% + violation amount
                shrink_factor = 1.0 - 0.01 - max_viol
                radii *= shrink_factor
                valid = False
        if valid:
            break
    return pos, radii

def _init_hexagonal():
    """Hexagonal lattice initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    r = 0.10
    dx = r * np.sqrt(3)
    dy = r * 1.5
    k = 0
    for row in range(6):
        offset = dx / 2 if row % 2 == 1 else 0
        start_x = r + offset
        y = row * dy + r
        for col in range(5):
            if k >= N:
                break
            x = start_x + col * dx
            if x + r <= 1.0 and x - r >= 0.0 and y + r <= 1.0 and y - r >= 0.0:
                pos[k] = [x, y]
                radii[k] = r
                k += 1
        if k >= N:
            break
    
    rng = np.random.RandomState(42)
    while k < N:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        min_dist = min(x, 1 - x, y, 1 - y)
        for j in range(k):
            d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
            min_dist = min(min_dist, d - radii[j])
        if min_dist > 0.02:
            pos[k] = [x, y]
            radii[k] = min_dist * 0.9
            k += 1
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
    return pos, radii

def _init_grid():
    """Grid initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            if idx >= N: break
            x = (i + 0.5) / 5.0
            y = (j + 0.5) / 5.0
            pos[idx] = [x, y]
            radii[idx] = 0.09
    # Add 26th
    rng = np.random.RandomState(43)
    k = N - 1
    while k < N:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        min_dist = min(x, 1 - x, y, 1 - y)
        for j in range(k):
            d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
            min_dist = min(min_dist, d - radii[j])
        if min_dist > 0.02:
            pos[k] = [x, y]
            radii[k] = min_dist * 0.9
            k += 1
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
    return pos, radii

def _init_corners():
    """Corner biased initialization."""
    pos = np.zeros((N, 2))
    radii = np.zeros(N)
    # Place circles in corners first
    corners = [(0.15, 0.15), (0.85, 0.15), (0.15, 0.85), (0.85, 0.85)]
    for i in range(4):
        pos[i] = corners[i]
        radii[i] = 0.15
    
    # Fill rest randomly
    rng = np.random.RandomState(44)
    k = 4
    while k < N:
        x = rng.uniform(0.1, 0.9)
        y = rng.uniform(0.1, 0.9)
        min_dist = min(x, 1 - x, y, 1 - y)
        for j in range(k):
            d = np.sqrt((x - pos[j, 0]) ** 2 + (y - pos[j, 1]) ** 2)
            min_dist = min(min_dist, d - radii[j])
        if min_dist > 0.02:
            pos[k] = [x, y]
            radii[k] = min_dist * 0.9
            k += 1
        else:
            pos[k] = [0.5, 0.5]
            radii[k] = 0.01
            k += 1
    return pos, radii

def _solve_radii_lp(pos):
    """
    Solve for maximum sum of radii given fixed positions using Linear Programming.
    Maximize sum(r_i)
    Subject to:
    r_i + r_j <= dist(i,j) for all i < j
    r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    r_i >= 0
    """
    n_vars = N
    # Objective: minimize -sum(r)
    c = -np.ones(n_vars)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # r_i + r_j <= dist_ij
    # r_i <= bound_i
    
    # Number of pairs
    n_pairs = N * (N - 1) // 2
    A_pairs = np.zeros((n_pairs, n_vars))
    b_pairs = np.zeros(n_pairs)
    
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            A_pairs[idx, i] = 1
            A_pairs[idx, j] = 1
            dist = np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2)
            b_pairs[idx] = dist
            idx += 1
            
    # Boundary constraints
    # r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    # This is equivalent to 4 constraints per circle, but we can just take the min
    # r_i <= bound_i
    A_bound = np.eye(n_vars)
    b_bound = np.array([min(x, 1-x, y, 1-y) for x, y in pos])
    
    A_ub = np.vstack([A_pairs, A_bound])
    b_ub = np.concatenate([b_pairs, b_bound])
    
    bounds = [(0, None) for _ in range(n_vars)]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x
    else:
        # Fallback: simple greedy or uniform
        return np.full(N, 0.01)

def _optimize_positions(pos, radii, deadline):
    """
    Optimize positions using L-BFGS-B.
    Objective: -sum(radii) + penalty(overlaps) + penalty(boundaries)
    Note: We keep radii fixed during this step to find a good geometry for them.
    Actually, a better approach is to optimize positions to *allow* for larger radii.
    But since radii are coupled, we use a soft penalty approach here.
    """
    # We want to minimize: -sum(r) is not differentiable w.r.t pos if r is fixed.
    # Instead, we minimize the "cost" of the current packing.
    # Cost = sum of overlaps + boundary violations.
    # And we want to maximize density.
    # Let's use a combined objective:
    # Minimize: Penalty(overlaps) + Penalty(boundaries) - lambda * sum(radii) is not right because r is fixed.
    # 
    # Better: We want to find positions such that the *optimal* radii (from LP) are as large as possible.
    # But LP is non-differentiable.
    # Approximation: Use a smooth penalty for overlaps and boundaries.
    # We treat the current radii as "target" sizes and penalize overlaps with them.
    
    def objective(x):
        p = x[:2*N].reshape(N, 2)
        
        # Boundary penalty
        pen_bound = 0.0
        for i in range(N):
            x_i, y_i = p[i]
            r_i = radii[i]
            # Soft plus penalty
            # If x_i < r_i, penalize
            # Use smooth approximation: softplus(k * (r_i - x_i))
            k = 50.0
            pen_bound += np.log(1 + np.exp(k * (r_i - x_i))) / k
            pen_bound += np.log(1 + np.exp(k * (r_i - (1 - x_i)))) / k
            pen_bound += np.log(1 + np.exp(k * (r_i - y_i))) / k
            pen_bound += np.log(1 + np.exp(k * (r_i - (1 - y_i)))) / k
            
        # Overlap penalty
        pen_overlap = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    # Quadratic penalty for overlap
                    pen_overlap += (sum_r - dist)**2
                    # Or smooth: softplus
                    # pen_overlap += np.log(1 + np.exp(50*(sum_r - dist))) / 50
        
        return 100 * pen_bound + 10 * pen_overlap

    x0 = pos.flatten()
    
    bounds = [(0.0, 1.0) for _ in range(2*N)]
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 50, 'ftol': 1e-8})
    
    if res.success:
        return res.x[:2*N].reshape(N, 2)
    else:
        return pos

def _alternating_optimize(pos, radii, deadline):
    """
    Alternate between position optimization and LP radius optimization.
    """
    best_score = np.sum(radii)
    best_pos = pos.copy()
    best_radii = radii.copy()
    
    # Initial LP solve to get valid radii for the initial positions
    radii = _solve_radii_lp(pos)
    pos, radii = _post_process(pos, radii)
    score = np.sum(radii)
    if score > best_score:
        best_score = score
        best_pos = pos.copy()
        best_radii = radii.copy()
        
    for _ in range(20):
        if time.time() > deadline - 5:
            break
            
        # 1. Optimize positions given current radii
        pos_new = _optimize_positions(pos, radii, deadline)
        
        # 2. Solve for optimal radii given new positions
        radii_new = _solve_radii_lp(pos_new)
        
        # 3. Post-process to ensure validity
        pos_pp, radii_pp = _post_process(pos_new, radii_new)
        
        score = np.sum(radii_pp)
        
        if score > best_score:
            best_score = score
            best_pos = pos_pp.copy()
            best_radii = radii_pp.copy()
            
        # Update for next iteration
        pos = pos_pp
        radii = radii_pp
        
        # Check convergence
        if np.linalg.norm(pos_new - pos) < 1e-5 and np.linalg.norm(radii_new - radii) < 1e-5:
            break
            
    return best_pos, best_radii

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Try multiple initializations
    initializers = [
        _init_hexagonal,
        _init_grid,
        _init_corners
    ]
    
    best_pos = None
    best_radii = None
    best_score = -np.inf
    
    for init_func in initializers:
        if time.time() > deadline - 30:
            break
            
        pos, radii = init_func()
        
        # Run alternating optimization
        pos_res, radii_res = _alternating_optimize(pos, radii, deadline)
        
        score = np.sum(radii_res)
        if score > best_score:
            best_score = score
            best_pos = pos_res.copy()
            best_radii = radii_res.copy()
            
    # Final Polish with SLSQP on the best solution found
    if best_pos is not None and time.time() < deadline - 10:
        x0 = np.concatenate([best_pos.flatten(), best_radii])
        
        bounds = []
        for i in range(N):
            bounds.append((0.001, 0.999))
            bounds.append((0.001, 0.999))
        for i in range(N):
            bounds.append((0.001, 0.4))
            
        constraints = []
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i] - x[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i] - x[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[2*i+1] - x[2*N+i]})
            constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[2*i+1] - x[2*N+i]})
        for i in range(N):
            for j in range(i + 1, N):
                constraints.append({'type': 'ineq',
                                    'fun': lambda x, i=i, j=j:
                                    (x[2*i] - x[2*j])**2 + (x[2*i+1] - x[2*j+1])**2 -
                                    (x[2*N+i] + x[2*N+j])**2})
                
        def objective(x):
            return -np.sum(x[2*N:])
            
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                              constraints=constraints,
                              options={'maxiter': 100, 'ftol': 1e-10})
            if result.success or result.fun < objective(x0):
                x = result.x
                pos_out = x[:2*N].reshape(N, 2)
                radii_out = np.maximum(x[2*N:], 1e-4)
                pos_out, radii_out = _post_process(pos_out, radii_out)
                if _is_valid(pos_out, radii_out):
                    score = np.sum(radii_out)
                    if score > best_score:
                        best_score = score
                        best_pos = pos_out.copy()
                        best_radii = radii_out.copy()
        except Exception:
            pass

    # Fallback
    if best_pos is None:
        pos, radii = _init_hexagonal()
        pos, radii = _post_process(pos, radii)
        best_pos = pos
        best_radii = radii

    result = np.column_stack([best_pos, best_radii])
    return result

# EVOLVE-BLOCK-END
