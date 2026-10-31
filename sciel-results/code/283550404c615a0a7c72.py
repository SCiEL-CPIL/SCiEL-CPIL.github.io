"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint

def construct():
    """
    Pack 26 circles in unit square to maximize sum of radii.
    Uses a physics-inspired initialization followed by local optimization
    with a smooth penalty for overlaps and wall violations.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    def objective(params):
        """
        params: array of shape (3*N,) -> [x0, y0, r0, x1, y1, r1, ...]
        Returns negative sum of radii (to minimize).
        """
        n = N
        xs = params[0::3]
        ys = params[1::3]
        rs = params[2::3]
        
        # Clip radii to be positive
        rs = np.maximum(rs, 1e-6)
        
        # Wall constraints: circles must be inside [0,1]^2
        # x_i - r_i >= 0  =>  r_i <= x_i
        # x_i + r_i <= 1  =>  r_i <= 1 - x_i
        # y_i - r_i >= 0  =>  r_i <= y_i
        # y_i + r_i <= 1  =>  r_i <= 1 - y_i
        
        # Soft penalty for wall violations
        wall_pen = 0.0
        for i in range(n):
            # Left wall: x_i >= r_i
            viol_l = max(0, rs[i] - xs[i])
            # Right wall: x_i + r_i <= 1
            viol_r = max(0, xs[i] + rs[i] - 1.0)
            # Bottom wall: y_i >= r_i
            viol_b = max(0, rs[i] - ys[i])
            # Top wall: y_i + r_i <= 1
            viol_t = max(0, ys[i] + rs[i] - 1.0)
            wall_pen += viol_l**2 + viol_r**2 + viol_b**2 + viol_t**2
        
        # Pairwise overlap penalty
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    overlap_pen += (min_dist - dist)**2
        
        # We want to maximize sum(rs), so minimize -sum(rs) + penalties
        return -np.sum(rs) + 100.0 * (wall_pen + overlap_pen)
    
    def grad(params):
        """
        Gradient of the objective function.
        """
        n = N
        xs = params[0::3].copy()
        ys = params[1::3].copy()
        rs = params[2::3].copy()
        
        rs = np.maximum(rs, 1e-6)
        
        grad = np.zeros(3 * n)
        
        # Derivative of -sum(rs)
        grad[2::3] = -1.0
        
        # Wall penalties
        for i in range(n):
            # Left: viol_l = max(0, r_i - x_i)
            if rs[i] > xs[i]:
                d = rs[i] - xs[i]
                grad[2*i + 2] += 200.0 * d  # d/d r_i
                grad[2*i] += -200.0 * d     # d/d x_i
            # Right: viol_r = max(0, x_i + r_i - 1)
            if xs[i] + rs[i] > 1.0:
                d = xs[i] + rs[i] - 1.0
                grad[2*i + 2] += 200.0 * d
                grad[2*i] += 200.0 * d
            # Bottom: viol_b = max(0, r_i - y_i)
            if rs[i] > ys[i]:
                d = rs[i] - ys[i]
                grad[2*i + 2] += 200.0 * d
                grad[2*i + 1] += -200.0 * d
            # Top: viol_t = max(0, y_i + r_i - 1)
            if ys[i] + rs[i] > 1.0:
                d = ys[i] + rs[i] - 1.0
                grad[2*i + 2] += 200.0 * d
                grad[2*i + 1] += 200.0 * d
        
        # Pairwise overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                if dist_sq < min_dist**2:
                    dist = np.sqrt(dist_sq)
                    if dist > 1e-10:
                        d = min_dist - dist
                        # d/dx_i of (min_dist - dist)^2
                        # = 2*(min_dist - dist) * (-1) * d(dist)/dx_i
                        # d(dist)/dx_i = dx / dist
                        factor = 200.0 * d / dist
                        grad[2*i] += -factor * dx
                        grad[2*i + 1] += -factor * dy
                        grad[2*j] += factor * dx
                        grad[2*j + 1] += factor * dy
                        # d/dr_i
                        grad[2*i + 2] += 200.0 * d
                        grad[2*j + 2] += 200.0 * d
                    else:
                        # Degenerate case
                        grad[2*i + 2] += 200.0 * min_dist
                        grad[2*j + 2] += 200.0 * min_dist
        
        return grad
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(N)))
    init_params = []
    for i in range(N):
        row = i // side
        col = i % side
        x = (col + 0.5) / side
        y = (row + 0.5) / side
        r = 0.5 / side - 0.01
        init_params.extend([x, y, r])
    init_params = np.array(init_params)
    
    # Try multiple random initializations
    best_result = None
    best_value = -np.inf
    
    # First try the grid initialization
    try:
        res = minimize(objective, init_params, method='L-BFGS-B', jac=grad,
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.success or res.fun < best_value:
            best_result = res.x
            best_value = res.fun
    except Exception:
        pass
    
    # Try random initializations
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        
        # Random positions and radii
        xs = np.random.uniform(0.1, 0.9, N)
        ys = np.random.uniform(0.1, 0.9, N)
        rs = np.random.uniform(0.02, 0.1, N)
        
        params = np.zeros(3 * N)
        params[0::3] = xs
        params[1::3] = ys
        params[2::3] = rs
        
        try:
            res = minimize(objective, params, method='L-BFGS-B', jac=grad,
                           options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_value:
                best_result = res.x
                best_value = res.fun
        except Exception:
            pass
    
    # Extract best result
    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    xs = best_result[0::3]
    ys = best_result[1::3]
    rs = best_result[2::3]
    
    # Clip to ensure valid configuration
    rs = np.maximum(rs, 1e-6)
    
    # Ensure circles are within bounds by shrinking if needed
    for i in range(N):
        # Check wall constraints
        r_max = min(xs[i], 1 - xs[i], ys[i], 1 - ys[i])
        if rs[i] > r_max:
            rs[i] = max(0, r_max - 1e-4)
    
    # Check pairwise constraints and shrink if needed
    for iteration in range(10):
        modified = False
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < rs[i] + rs[j]:
                    # Shrink both circles
                    overlap = rs[i] + rs[j] - dist
                    if overlap > 0:
                        # Reduce radii proportionally
                        total_r = rs[i] + rs[j]
                        if total_r > 1e-10:
                            scale = (dist - 1e-4) / total_r
                            if scale > 0:
                                rs[i] *= scale
                                rs[j] *= scale
                                modified = True
        if not modified:
            break
    
    # Re-check wall constraints after shrinking
    for i in range(N):
        r_max = min(xs[i], 1 - xs[i], ys[i], 1 - ys[i])
        if rs[i] > r_max:
            rs[i] = max(0, r_max - 1e-4)
    
    result = np.column_stack([xs, ys, rs])
    return result

# EVOLVE-BLOCK-END
