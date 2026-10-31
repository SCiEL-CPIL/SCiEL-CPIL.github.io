import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _objective_and_grad(v, penalty_weight=1000.0):
    """
    Objective function with penalty for constraint violations.
    v: vector of shape (3*N,) containing [x1, y1, r1, x2, y2, r2, ...]
    Returns: objective value, gradient
    """
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    
    n = len(r)
    
    # Base objective: minimize -sum(r)
    obj = -np.sum(r)
    grad = np.zeros_like(v)
    grad[2::3] = -1.0
    
    # Penalty for boundaries: r_i <= x_i <= 1-r_i and r_i <= y_i <= 1-r_i
    # Violations:
    # x_i < r_i  -> x_i - r_i < 0
    # x_i > 1 - r_i -> x_i + r_i - 1 > 0
    # y_i < r_i  -> y_i - r_i < 0
    # y_i > 1 - r_i -> y_i + r_i - 1 > 0
    
    # We use squared penalty for smoothness
    # P = sum (max(0, r_i - x_i)^2 + max(0, x_i + r_i - 1)^2 + max(0, r_i - y_i)^2 + max(0, y_i + r_i - 1)^2)
    
    # Calculate violations
    v_x_left = np.maximum(0, r - x)
    v_x_right = np.maximum(0, x + r - 1)
    v_y_left = np.maximum(0, r - y)
    v_y_right = np.maximum(0, y + r - 1)
    
    pen = np.sum(v_x_left**2) + np.sum(v_x_right**2) + np.sum(v_y_left**2) + np.sum(v_y_right**2)
    obj += penalty_weight * pen
    
    # Gradients for boundary penalties
    # d/dx (r-x)^2 = -2(r-x) if r>x else 0
    # d/dx (x+r-1)^2 = 2(x+r-1) if x+r>1 else 0
    # d/dr (r-x)^2 = 2(r-x) if r>x else 0
    # d/dr (x+r-1)^2 = 2(x+r-1) if x+r>1 else 0
    
    grad_x = -2 * v_x_left + 2 * v_x_right
    grad_y = -2 * v_y_left + 2 * v_y_right
    grad_r = 2 * v_x_left + 2 * v_x_right + 2 * v_y_left + 2 * v_y_right
    
    grad[0::3] += penalty_weight * grad_x
    grad[1::3] += penalty_weight * grad_y
    grad[2::3] += penalty_weight * grad_r
    
    # Penalty for overlaps: (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # Violation: (r_i + r_j)^2 - ((x_i - x_j)^2 + (y_i - y_j)^2) > 0
    # Let d_ij = sqrt(dx^2 + dy^2). Violation is max(0, r_i + r_j - d_ij)
    # Squared penalty: (max(0, r_i + r_j - d_ij))^2
    
    # Vectorized calculation for all pairs i < j
    # This is O(N^2). N=26 is small, so it's fine.
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2 + 1e-12) # add epsilon to avoid div by 0 in grad
    
    r_sum = r[:, None] + r[None, :]
    
    # We only care about i < j
    # Create mask for upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Extract values
    d = dist[mask]
    rs = r_sum[mask]
    
    # Violation: max(0, rs - d)
    viol = np.maximum(0, rs - d)
    
    pen_overlap = np.sum(viol**2)
    obj += penalty_weight * pen_overlap
    
    # Gradients for overlap penalty
    # f = (rs - d)^2 if rs > d else 0
    # d/dx_i f = 2(rs - d) * d/dx_i (rs - d) = 2(rs - d) * (-d/dx_i d)
    # d/dx_i d = (x_i - x_j) / d
    # So d/dx_i f = -2(rs - d) * (x_i - x_j) / d
    
    # We need to accumulate gradients for all i and j involved in each pair.
    # This is tricky to vectorize perfectly for gradient accumulation without loops or sparse matrices.
    # Given N=26, a simple loop over pairs for gradient calculation is fast enough.
    
    grad_overlap = np.zeros_like(v)
    
    # Indices for upper triangle
    i_idx, j_idx = np.where(mask)
    
    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        if viol[k] > 0:
            # Gradient contribution for pair (i, j)
            # Term: 2 * viol[k] * (rs[k] - d[k]) ... wait, f = (rs - d)^2
            # df/dx_i = 2(rs - d) * (-(x_i - x_j)/d) = -2 viol[k] * (x_i - x_j) / d[k]
            # df/dx_j = 2(rs - d) * (-(x_j - x_i)/d) = 2 viol[k] * (x_i - x_j) / d[k]
            # df/dy_i = -2 viol[k] * (y_i - y_j) / d[k]
            # df/dy_j = 2 viol[k] * (y_i - y_j) / d[k]
            # df/dr_i = 2(rs - d) * 1 = 2 viol[k]
            # df/dr_j = 2(rs - d) * 1 = 2 viol[k]
            
            coeff = -2 * viol[k] / d[k]
            
            grad_overlap[3*i] += penalty_weight * coeff * (x[i] - x[j])
            grad_overlap[3*j] += penalty_weight * coeff * (x[j] - x[i])
            
            grad_overlap[3*i + 1] += penalty_weight * coeff * (y[i] - y[j])
            grad_overlap[3*j + 1] += penalty_weight * coeff * (y[j] - y[i])
            
            grad_overlap[3*i + 2] += penalty_weight * 2 * viol[k]
            grad_overlap[3*j + 2] += penalty_weight * 2 * viol[k]
            
    grad += grad_overlap
    
    return obj, grad

def _is_feasible(v, tol=1e-6):
    """Check if the configuration is feasible."""
    x = v[0::3]
    y = v[1::3]
    r = v[2::3]
    n = len(r)
    
    # Check bounds
    if np.any(x < r - tol) or np.any(x > 1 - r + tol) or np.any(y < r - tol) or np.any(y > 1 - r + tol):
        return False
    
    # Check overlaps
    for i in range(n):
        for j in range(i+1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _make_grid_initial(n_circles, jitter=0.0):
    """Create a grid-based initial guess."""
    # Use a grid that can fit n_circles
    # For N=26, 5x5 is 25. We can use 5x5 and add one, or 4x7.
    # Let's use a grid of size side x side where side*side >= n_circles
    side = int(np.ceil(np.sqrt(n_circles)))
    
    # Generate grid points
    xs = []
    ys = []
    for i in range(side):
        for j in range(side):
            xs.append((i + 0.5) / side)
            ys.append((j + 0.5) / side)
    
    # Take first n_circles
    xs = np.array(xs[:n_circles])
    ys = np.array(ys[:n_circles])
    
    # Initial radius: slightly less than half cell size
    r = np.full(n_circles, 0.5 / side - 1e-3)
    
    if jitter > 0:
        xs += np.random.uniform(-jitter, jitter, n_circles)
        ys += np.random.uniform(-jitter, jitter, n_circles)
        # Clip to stay roughly in bounds for initial guess
        # Don't clip too hard, let optimizer fix
        xs = np.clip(xs, 0.01, 0.99)
        ys = np.clip(ys, 0.01, 0.99)
        
    v = np.zeros(3 * n_circles)
    v[0::3] = xs
    v[1::3] = ys
    v[2::3] = r
    
    return v

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s safety margin
    
    best_score = -np.inf
    best_v = None
    
    # Multiple restarts
    n_restarts = 100
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Create initial guess
        # Alternate between grid and random
        if i % 2 == 0:
            # Grid with jitter
            v0 = _make_grid_initial(N, jitter=0.05)
        else:
            # Random initialization
            v0 = np.random.uniform(0.1, 0.9, 3*N)
            # Ensure r is small initially
            v0[2::3] = np.random.uniform(0.05, 0.15, N)
            # Ensure x,y are at least r away from boundaries? Not strictly necessary for penalty method
            
        # Optimization
        # Use SLSQP with penalty objective
        # We don't pass constraints to SLSQP, we rely on the penalty in the objective
        # This allows the optimizer to move through infeasible regions if it helps find a better feasible region later
        
        try:
            res = minimize(
                _objective_and_grad,
                v0,
                args=(1000.0,), # penalty weight
                method='SLSQP',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-9}
            )
            
            if res.success or res.fun < best_score:
                v_res = res.x
                # Check feasibility
                if _is_feasible(v_res):
                    score = np.sum(v_res[2::3])
                    if score > best_score:
                        best_score = score
                        best_v = v_res.copy()
                        
        except Exception as e:
            continue

    # If no feasible solution found (unlikely with grid start), return a valid trivial one
    if best_v is None:
        v0 = _make_grid_initial(N, jitter=0.0)
        return v0.reshape(N, 3)
    
    # Post-process: Ensure strict feasibility and maximize radii slightly if possible
    # Simple check and return
    return best_v.reshape(N, 3)

# EVOLVE-BLOCK-END
