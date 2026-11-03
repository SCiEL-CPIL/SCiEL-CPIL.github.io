def _solve_penalty(x0_flat, W, H):
    n = N
    def objective(v):
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        
        # Maximize sum of radii -> Minimize -sum(rs)
        obj = -np.sum(rs)
        
        # Penalty for boundary violations
        # x_i >= r_i, x_i <= W - r_i
        # y_i >= r_i, y_i <= H - r_i
        # We use squared penalty for smoothness
        px = np.maximum(0, rs - xs) + np.maximum(0, xs - (W - rs))
        py = np.maximum(0, rs - ys) + np.maximum(0, ys - (H - rs))
        penalty_boundary = np.sum(px**2) + np.sum(py**2)
        
        # Penalty for overlaps
        penalty_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    penalty_overlap += (min_dist - dist)**2
        
        return obj + 100 * (penalty_boundary + penalty_overlap)
        
    def gradient(v):
        xs = v[0::3]
        ys = v[1::3]
        rs = v[2::3]
        grad = np.zeros_like(v)
        
        # Gradient of -sum(rs)
        grad[2::3] = -1.0
        
        # Gradient of boundary penalty
        # d/dx_i (max(0, r_i - x_i)^2) = -2(r_i - x_i) if x_i < r_i else 0
        # d/dx_i (max(0, x_i - (W-r_i))^2) = 2(x_i - (W-r_i)) if x_i > W-r_i else 0
        # Note: max is not differentiable at 0, but subgradient 0 is fine for L-BFGS-B usually, 
        # or we can use a smooth approximation. Let's stick to simple subgradient for now.
        
        # Actually, implementing the full gradient for the penalty is tedious.
        # L-BFGS-B can estimate gradients if we don't provide them, but it's slower.
        # Or we can use a smooth penalty: soft-plus or similar?
        # Given the time limit, let's try providing a finite difference gradient or just letting L-BFGS-B do it?
        # No, L-BFGS-B without gradient is slower.
        # Let's use a simpler penalty: L1 penalty? No, L2 is better for convergence.
        
        # Let's just use `minimize` with `method='L-BFGS-B'` and NO explicit gradient, 
        # relying on finite differences. With 63 variables, FD is 64 function calls per step.
        # This might be slow but likely faster than SLSQP with 210 constraints.
        
        return None # Let scipy estimate? Or I should implement it.
