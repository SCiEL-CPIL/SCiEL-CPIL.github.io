import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_value = -np.inf
    best_solution = None
    
    # Precompute indices for pairwise distances
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def softplus(x, beta=50.0):
        # Numerically stable softplus
        xb = beta * x
        return np.where(xb > 0, x + np.log1p(np.exp(-xb)) / beta, np.log1p(np.exp(xb)) / beta)
        
    def softplus_grad(x, beta=50.0):
        # Gradient of softplus is sigmoid(beta*x)
        xb = beta * x
        return 1.0 / (1.0 + np.exp(-xb))

    def objective_and_grad(x, W, H, beta=50.0):
        """
        x: shape (3N,) = [x0, y0, r0, x1, y1, r1, ...]
        Returns: (objective_value, gradient_vector)
        Objective: -sum(r) + penalty_terms
        """
        c = x.reshape(-1, 3)
        xs = c[:, 0]
        ys = c[:, 1]
        rs = c[:, 2]
        
        obj = -np.sum(rs)
        grad = np.zeros_like(x)
        
        # 1. Boundary Penalties
        # Left: x - r < 0  => r - x > 0
        v_left = rs - xs
        obj += np.sum(softplus(v_left, beta))
        g_left = softplus_grad(v_left, beta)
        grad[0::3] -= g_left
        grad[2::3] += g_left
        
        # Right: x + r > W  => x + r - W > 0
        v_right = xs + rs - W
        obj += np.sum(softplus(v_right, beta))
        g_right = softplus_grad(v_right, beta)
        grad[0::3] += g_right
        grad[2::3] += g_right
        
        # Bottom: y - r < 0  => r - y > 0
        v_bottom = rs - ys
        obj += np.sum(softplus(v_bottom, beta))
        g_bottom = softplus_grad(v_bottom, beta)
        grad[1::3] -= g_bottom
        grad[2::3] += g_bottom
        
        # Top: y + r > H  => y + r - H > 0
        v_top = ys + rs - H
        obj += np.sum(softplus(v_top, beta))
        g_top = softplus_grad(v_top, beta)
        grad[1::3] += g_top
        grad[2::3] += g_top
        
        # 2. Overlap Penalties
        # For each pair i, j: dist < r_i + r_j  => r_i + r_j - dist > 0
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist_sq = dx*dx + dy*dy
        # Avoid division by zero
        dist = np.sqrt(dist_sq + 1e-12)
        
        v_overlap = rs[i_idx] + rs[j_idx] - dist
        obj += np.sum(softplus(v_overlap, beta))
        g_overlap = softplus_grad(v_overlap, beta)
        
        # Gradients for overlap term
        # d/dx_i (r_i + r_j - dist) = -dx/dist
        # d/dy_i (r_i + r_j - dist) = -dy/dist
        # d/dr_i (r_i + r_j - dist) = 1
        
        # We need to accumulate gradients for i and j
        # Using np.add.at for accumulation
        # Note: i_idx and j_idx are 1D arrays of length M
        
        # Contribution to i
        grad_i_x = -g_overlap * (dx / dist)
        grad_i_y = -g_overlap * (dy / dist)
        grad_i_r = g_overlap
        
        # Contribution to j
        grad_j_x = g_overlap * (dx / dist)
        grad_j_y = g_overlap * (dy / dist)
        grad_j_r = g_overlap
        
        # Accumulate into global grad vector
        # x coordinates are at indices 0, 3, 6...
        # y coordinates are at indices 1, 4, 7...
        # r coordinates are at indices 2, 5, 8...
        
        idx_i = i_idx
        idx_j = j_idx
        
        np.add.at(grad, 3*idx_i, grad_i_x)
        np.add.at(grad, 3*idx_i + 1, grad_i_y)
        np.add.at(grad, 3*idx_i + 2, grad_i_r)
        
        np.add.at(grad, 3*idx_j, grad_j_x)
        np.add.at(grad, 3*idx_j + 1, grad_j_y)
        np.add.at(grad, 3*idx_j + 2, grad_j_r)
        
        return obj, grad

    def check_validity(circles, W, H):
        """
        circles: shape (N, 3)
        Returns: (is_valid, sum_r)
        """
        if circles is None:
            return False, 0.0
        xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
        
        # Boundary check
        if np.any(xs - rs < -1e-5) or np.any(xs + rs > W + 1e-5) or \
           np.any(ys - rs < -1e-5) or np.any(ys + rs > H + 1e-5) or \
           np.any(rs < 1e-6):
            return False, 0.0
            
        # Overlap check
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx*dx + dy*dy)
        req = rs[i_idx] + rs[j_idx]
        if np.any(dist < req - 1e-5):
            return False, 0.0
            
        return True, np.sum(rs)

    def hex_init(W, H):
        """
        Generate an initial hexagonal packing.
        """
        # Estimate radius
        # Area of rect = W*H. Area of N circles ~ N * pi * r^2.
        # Packing density ~ 0.9.
        # r ~ sqrt(W*H*0.9 / (N*pi))
        r_est = np.sqrt(W * H * 0.8 / (N * np.pi))
        
        # Create a grid of points
        # Rows and columns
        # Let's try to fit as many as possible in a hex pattern
        # dx = 2*r, dy = sqrt(3)*r
        
        points = []
        row = 0
        y = r_est
        while y < H - r_est + 1e-6:
            x = r_est if row % 2 == 0 else 2*r_est
            while x < W - r_est + 1e-6:
                points.append([x, y, r_est])
                x += 2 * r_est
            if row % 2 == 0:
                # Shift next row
                pass
            row += 1
            y += np.sqrt(3) * r_est
            
        # If we have more than N, take the first N (or center them?)
        # If we have less, we need to adjust.
        # For simplicity, let's just use the first N if available, else pad with random
        if len(points) < N:
            # Pad with random points
            for _ in range(N - len(points)):
                px = np.random.uniform(r_est, W - r_est)
                py = np.random.uniform(r_est, H - r_est)
                points.append([px, py, r_est])
        else:
            points = points[:N]
            
        return np.array(points).flatten()

    def grid_init(W, H):
        """
        Generate a simple grid initial packing.
        """
        # Find a grid a x b such that a*b >= N
        a = int(np.ceil(np.sqrt(N * W / H)))
        b = int(np.ceil(N / a))
        while a * b < N:
            b += 1
            
        points = []
        for i in range(N):
            col = i % a
            row = i // a
            # Center in cell
            x = (col + 0.5) * W / a
            y = (row + 0.5) * H / b
            r = min(W / a, H / b) * 0.45 # Safe radius
            points.append([x, y, r])
            
        return np.array(points).flatten()

    # Aspect ratios to try
    # W + H = 2. W in [1, 2].
    # We will sample W.
    # To save time, we can use a coarse grid first, then refine around the best.
    # But N=21 is small enough that we can just try ~20 aspect ratios.
    
    aspect_ratios = np.linspace(1.0, 2.0, 21)
    
    np.random.seed(42)
    
    for W in aspect_ratios:
        H = 2.0 - W
        
        if H < 0.1: # Too skewed
            continue
            
        # Try a few different initializations
        inits = [
            hex_init(W, H),
            grid_init(W, H)
        ]
        
        for x0 in inits:
            if time.time() > deadline - 5.0:
                break
                
            # Bounds for optimizer
            # x in [0, W], y in [0, H], r in [0, min(W,H)/2]
            max_r = min(W, H) / 2.0
            bounds = []
            for _ in range(N):
                bounds.append((0, W))
                bounds.append((0, H))
                bounds.append((1e-6, max_r))
            
            # Optimize
            try:
                res = minimize(
                    lambda x: objective_and_grad(x, W, H)[0],
                    x0,
                    jac=lambda x: objective_and_grad(x, W, H)[1],
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12}
                )
            except Exception:
                continue
                
            if res.success or res.fun < 1e-1:
                x_opt = res.x.reshape(-1, 3)
                is_valid, val = check_validity(x_opt, W, H)
                if is_valid and val > best_value:
                    best_value = val
                    best_solution = x_opt.copy()
                    
    # If no valid solution found, return a fallback
    if best_solution is None:
        # Fallback: simple grid in square
        W = 1.0
        H = 1.0
        x0 = grid_init(W, H)
        x_opt = x0.reshape(-1, 3)
        # Just return this, it might be invalid but it's better than crashing
        # Actually, let's make sure it's valid by shrinking radii
        for i in range(N):
            x_opt[i, 2] = min(x_opt[i, 2], x_opt[i, 0], W-x_opt[i, 0], x_opt[i, 1], H-x_opt[i, 1]) * 0.99
        best_solution = x_opt
        best_value = np.sum(x_opt[:, 2])
        
    return best_solution
