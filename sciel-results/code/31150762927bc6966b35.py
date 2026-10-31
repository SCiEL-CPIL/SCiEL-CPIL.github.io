import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave safety margin

    best_score = -np.inf
    best_result = None

    # Variables: x0..xN-1, y0..yN-1, r0..rN-1, W, H
    # Total: 3*N + 2 = 65

    def objective_penalty(x):
        """Penalty-based objective: minimize -sum(r) + penalty for violations."""
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        W = x[3*N]
        H = x[3*N+1]
        
        # Base objective: negative sum of radii (we want to maximize)
        base = -np.sum(rs)
        
        # Perimeter constraint: 2(W+H) <= 4 => W+H <= 2
        perim_viol = max(0, W + H - 2.0)
        
        # Boundary violations
        bound_viol = 0.0
        for i in range(N):
            bound_viol += max(0, rs[i] - xs[i])
            bound_viol += max(0, xs[i] + rs[i] - W)
            bound_viol += max(0, rs[i] - ys[i])
            bound_viol += max(0, ys[i] + rs[i] - H)
            bound_viol += max(0, -rs[i])
        
        # Overlap violations
        overlap_viol = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                rad_sum = rs[i] + rs[j]
                if rad_sum > 0:
                    overlap_viol += max(0, rad_sum - np.sqrt(dist_sq + 1e-12))
        
        # Penalty weight
        penalty_weight = 10.0
        return base + penalty_weight * (perim_viol + bound_viol + overlap_viol)
    
    def gradient_penalty(x):
        """Analytic gradient of the penalty objective."""
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        W = x[3*N]
        H = x[3*N+1]
        
        grad = np.zeros(3*N + 2)
        
        # Base gradient: d/dx_i of -sum(r) = 0 for positions, -1 for radii
        grad[2*N:3*N] = -1.0
        
        # Perimeter constraint: d/dW = 1, d/dH = 1 if W+H > 2
        perim_viol = max(0, W + H - 2.0)
        if perim_viol > 0:
            grad[3*N] += 10.0
            grad[3*N+1] += 10.0
        
        # Boundary violations
        for i in range(N):
            # x_i - r_i >= 0 => violation = max(0, r_i - x_i)
            if rs[i] - xs[i] > 0:
                grad[i] -= 10.0
                grad[2*N+i] += 10.0
            
            # W - x_i - r_i >= 0 => violation = max(0, x_i + r_i - W)
            if xs[i] + rs[i] - W > 0:
                grad[i] += 10.0
                grad[2*N+i] += 10.0
                grad[3*N] -= 10.0
            
            # y_i - r_i >= 0
            if rs[i] - ys[i] > 0:
                grad[N+i] -= 10.0
                grad[2*N+i] += 10.0
            
            # H - y_i - r_i >= 0
            if ys[i] + rs[i] - H > 0:
                grad[N+i] += 10.0
                grad[2*N+i] += 10.0
                grad[3*N+1] -= 10.0
            
            # -r_i >= 0 => r_i <= 0
            if -rs[i] > 0:
                grad[2*N+i] -= 10.0
        
        # Overlap violations
        for i in range(N):
            for j in range(i+1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy + 1e-12)
                rad_sum = rs[i] + rs[j]
                
                if dist > 1e-10:
                    overlap_viol = max(0.0, rad_sum - dist)
                    if overlap_viol > 0:
                        # d/dx_i of (rad_sum - dist) = -dx/dist
                        # d/dx_j of (rad_sum - dist) = +dx/dist
                        # d/dr_i = 1, d/dr_j = 1
                        grad[i] += 10.0 * (-dx / dist)
                        grad[j] += 10.0 * (dx / dist)
                        grad[N+i] += 10.0 * (-dy / dist)
                        grad[N+j] += 10.0 * (dy / dist)
                        grad[2*N+i] += 10.0
                        grad[2*N+j] += 10.0
        
        return grad
    
    def is_valid(x):
        xs = x[:N]
        ys = x[N:2*N]
        rs = x[2*N:3*N]
        W = x[3*N]
        H = x[3*N+1]
        
        if W <= 0 or H <= 0:
            return False
        if 2*(W+H) > 4 + 1e-5:
            return False
        
        for i in range(N):
            if xs[i] < rs[i] - 1e-6 or xs[i] > W - rs[i] + 1e-6:
                return False
            if ys[i] < rs[i] - 1e-6 or ys[i] > H - rs[i] + 1e-6:
                return False
            if rs[i] < 1e-7:
                return False
                
        for i in range(N):
            for j in range(i+1, N):
                dist_sq = (xs[i]-xs[j])**2 + (ys[i]-ys[j])**2
                rad_sum_sq = (rs[i]+rs[j])**2
                if dist_sq < rad_sum_sq - 1e-5:
                    return False
        return True
    
    def get_score(x):
        if not is_valid(x):
            return -np.inf
        rs = x[2*N:3*N]
        return np.sum(rs)
    
    def grid_init(W, H):
        """Create a grid-based initial layout."""
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        # Try different grid configurations
        best_grid_score = -np.inf
        best_grid = None
        
        for cols in range(3, 7):
            rows = int(np.ceil(N / cols))
            if rows < 1:
                continue
            
            dx = W / cols
            dy = H / rows
            
            x_temp = np.zeros(N)
            y_temp = np.zeros(N)
            r_temp = np.zeros(N)
            
            k = 0
            for i in range(cols):
                for j in range(rows):
                    if k >= N:
                        break
                    x_temp[k] = (i + 0.5) * dx
                    y_temp[k] = (j + 0.5) * dy
                    r_temp[k] = 0.45 * min(dx, dy)
                    k += 1
            
            # Check validity and score
            test_x = np.concatenate([x_temp, y_temp, r_temp, [W, H]])
            if is_valid(test_x):
                score = np.sum(r_temp)
                if score > best_grid_score:
                    best_grid_score = score
                    best_grid = test_x.copy()
        
        if best_grid is not None:
            return best_grid
        
        # Fallback: simple grid
        cols = 5
        rows = int(np.ceil(N / cols))
        dx = W / cols
        dy = H / rows
        
        k = 0
        for i in range(cols):
            for j in range(rows):
                if k >= N:
                    break
                xs[k] = (i + 0.5) * dx
                ys[k] = (j + 0.5) * dy
                rs[k] = 0.4 * min(dx, dy)
                k += 1
        
        return np.concatenate([xs, ys, rs, [W, H]])
    
    def hex_init(W, H):
        """Create a hexagonal-style initial layout."""
        xs = np.zeros(N)
        ys = np.zeros(N)
        rs = np.zeros(N)
        
        # Approximate radius for hex packing
        # Area per circle ~ 2.77 * r^2, total area = W*H
        # N * 2.77 * r^2 <= W*H => r <= sqrt(W*H / (N * 2.77))
        r_est = np.sqrt(W * H / (N * 2.77)) * 0.8
        
        # Place in staggered rows
        dx = 2 * r_est
        dy = np.sqrt(3) * r_est
        
        k = 0
        row = 0
        while k < N:
            if row % 2 == 0:
                n_in_row = int(W // (2 * r_est)) + 1
            else:
                n_in_row = int(W // (2 * r_est)) + 1
            
            for col in range(n_in_row):
                if k >= N:
                    break
                x_offset = r_est if row % 2 == 0 else 2 * r_est
                xs[k] = x_offset + col * 2 * r_est
                ys[k] = r_est + row * dy
                rs[k] = r_est
                k += 1
            
            row += 1
            if row * dy > H:
                break
        
        # If we didn't place all, use remaining spots
        if k < N:
            for i in range(k, N):
                xs[i] = np.random.uniform(r_est, W - r_est)
                ys[i] = np.random.uniform(r_est, H - r_est)
                rs[i] = r_est * 0.8
        
        # Clip to bounds
        for i in range(N):
            xs[i] = np.clip(xs[i], rs[i], W - rs[i])
            ys[i] = np.clip(ys[i], rs[i], H - rs[i])
        
        return np.concatenate([xs, ys, rs, [W, H]])
    
    def optimize_from(x0, max_iter=500):
        """Run L-BFGS-B from initial point."""
        try:
            res = minimize(
                objective_penalty,
                x0,
                jac=gradient_penalty,
                method='L-BFGS-B',
                options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            return res.x
        except Exception:
            return x0
    
    # Multi-start optimization
    attempts = 0
    max_attempts = 30
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        
        # Vary aspect ratio: W + H = 2
        # W in [0.6, 1.4] to avoid extreme aspect ratios
        W = np.random.uniform(0.7, 1.3)
        H = 2.0 - W
        
        # Choose initialization
        if attempts % 4 == 0:
            x0 = hex_init(W, H)
        else:
            x0 = grid_init(W, H)
        
        # Add small perturbation
        x0[:3*N] += np.random.normal(0, 0.02, 3*N)
        
        # Optimize
        x_opt = optimize_from(x0)
        
        # Check validity and score
        if is_valid(x_opt):
            score = get_score(x_opt)
            if score > best_score:
                best_score = score
                best_result = x_opt.copy()
        
        # Try local refinement with tighter penalty
        if best_result is not None:
            # Small local search around best
            x_refine = best_result + np.random.normal(0, 0.01, 3*N + 2)
            x_refine[3*N] = W  # Keep W fixed for refinement
            x_refine[3*N+1] = H
            x_opt2 = optimize_from(x_refine, max_iter=200)
            if is_valid(x_opt2):
                score2 = get_score(x_opt2)
                if score2 > best_score:
                    best_score = score2
                    best_result = x_opt2.copy()
    
    # If no valid result found, use fallback
    if best_result is None:
        # Simple grid in 1x1 square
        W = 1.0
        H = 1.0
        side = 5
        xs = []
        ys = []
        rs = []
        for i in range(5):
            for j in range(5):
                if len(xs) >= N:
                    break
                xs.append((i + 0.5) / 5.0)
                ys.append((j + 0.5) / 5.0)
                rs.append(0.45 / 5.0)
        best_result = np.array(xs + ys + rs + [W, H])
    
    xs = best_result[:N]
    ys = best_result[N:2*N]
    rs = best_result[2*N:3*N]
    
    return np.column_stack((xs, ys, rs))
