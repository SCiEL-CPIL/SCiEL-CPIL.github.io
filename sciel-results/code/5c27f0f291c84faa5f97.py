import numpy as np
import time
from scipy.optimize import minimize

N = 21

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s
    
    def evaluate(circles, w, h):
        """Return sum of radii if valid, else -inf."""
        n = len(circles)
        for i in range(n):
            x, y, r = circles[i]
            if x - r < -1e-6 or x + r > w + 1e-6 or y - r < -1e-6 or y + r > h + 1e-6:
                return -np.inf
        for i in range(n):
            for j in range(i+1, n):
                xi, yi, ri = circles[i]
                xj, yj, rj = circles[j]
                d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if d < ri + rj - 1e-6:
                    return -np.inf
        return float(np.sum(circles[:, 2]))
    
    def hex_lattice_solution():
        """
        Generate hexagonal lattice packing.
        For N=21, try different hexagonal arrangements.
        """
        best = None
        
        # Hexagonal arrangements for 21 circles
        # Row counts that sum to 21 (or close)
        row_configs = [
            [3, 4, 4, 4, 3, 3],  # 21
            [4, 4, 4, 4, 3, 2],  # 21
            [2, 4, 4, 4, 4, 3],  # 21
            [3, 4, 4, 4, 4, 2],  # 21
            [4, 3, 4, 4, 4, 2],  # 21
            [2, 3, 4, 4, 4, 4],  # 21
            [3, 3, 4, 4, 4, 3],  # 21
            [4, 4, 3, 4, 4, 2],  # 21
            [2, 4, 3, 4, 4, 4],  # 21
            [3, 4, 3, 4, 4, 3],  # 21
        ]
        
        for row_counts in row_configs:
            if sum(row_counts) != N:
                continue
            
            # Try different aspect ratios and shears
            for aspect in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 0.7, 1.4, 0.6]:
                w = 2.0 * aspect / (1.0 + aspect)
                h = 2.0 / (1.0 + aspect)
                
                # Hexagonal spacing
                # For hex packing, vertical spacing = r*sqrt(3), horizontal = 2r
                # We'll estimate radius from available space
                n_rows = len(row_counts)
                max_cols = max(row_counts)
                
                # Estimate radius
                # Horizontal: need max_cols circles with hex offset
                # Vertical: need n_rows rows with hex spacing
                est_r_h = w / (max_cols + 0.5)  # rough
                est_r_v = h / (n_rows * np.sqrt(3) / 2 + 1)  # rough
                r_est = min(est_r_h, est_r_v)
                
                # Build hex lattice
                circles = []
                row_offset = 0
                for row_idx, n_cols in enumerate(row_counts):
                    y = (row_idx + 0.5) * r_est * np.sqrt(3)
                    # Offset alternating rows
                    x_offset = 0.5 * r_est if row_idx % 2 == 1 else 0.0
                    for col_idx in range(n_cols):
                        x = x_offset + (col_idx + 0.5) * 2 * r_est
                        # Center the row
                        row_width = n_cols * 2 * r_est
                        center_x = w / 2
                        x = center_x - row_width / 2 + (col_idx + 0.5) * 2 * r_est
                        circles.append([x, y, r_est])
                
                circles = np.array(circles)
                
                # Check if valid and get value
                val = evaluate(circles, w, h)
                
                if val > (best[0] if best else -np.inf):
                    best = (val, circles.copy(), w, h)
        
        if best is None:
            # Fallback to square grid
            w, h = 1.0, 1.0
            circles = []
            for i in range(N):
                c = i % 5
                r = i // 5
                circles.append([0.1 + c * 0.2, 0.1 + r * 0.2, 0.09])
            circles = np.array(circles)
            best = (evaluate(circles, w, h), circles, w, h)
        
        return best[1], best[2], best[3], best[0]
    
    def softplus(x, beta=100.0):
        """Numerically stable softplus: log(1+exp(beta*x))/beta"""
        x_beta = beta * x
        result = np.where(x_beta > 0, 
                         x + np.log1p(np.exp(-x_beta)) / beta,
                         np.log1p(np.exp(x_beta)) / beta)
        return result
    
    def softplus_grad(x, beta=100.0):
        """Gradient of softplus: sigmoid(beta*x)"""
        return 1.0 / (1.0 + np.exp(-beta * x))
    
    def optimize(circles0, w, h, max_iter=300, beta=100.0, equalize_weight=0.0):
        """Optimize from initial configuration using smooth softplus penalties."""
        x0 = circles0.flatten()
        
        def objective_and_grad(x):
            c = x.reshape(-1, 3)
            val = -np.sum(c[:, 2])
            
            penalty = 0.0
            grad = np.zeros_like(x)
            
            # Wall constraints
            for i in range(N):
                xi, yi, ri = c[i]
                
                v1 = ri - xi
                penalty += softplus(v1, beta)
                g1 = softplus_grad(v1, beta)
                grad[3*i] += -g1
                grad[3*i+2] += g1
                
                v2 = xi + ri - w
                penalty += softplus(v2, beta)
                g2 = softplus_grad(v2, beta)
                grad[3*i] += g2
                grad[3*i+2] += g2
                
                v3 = ri - yi
                penalty += softplus(v3, beta)
                g3 = softplus_grad(v3, beta)
                grad[3*i+1] += -g3
                grad[3*i+2] += g3
                
                v4 = yi + ri - h
                penalty += softplus(v4, beta)
                g4 = softplus_grad(v4, beta)
                grad[3*i+1] += g4
                grad[3*i+2] += g4
                
                v5 = -ri
                penalty += softplus(v5, beta)
                g5 = softplus_grad(v5, beta)
                grad[3*i+2] += -g5
            
            # Non-overlap constraints
            for i in range(N):
                for j in range(i+1, N):
                    xi, yi, ri = c[i]
                    xj, yj, rj = c[j]
                    dx = xi - xj
                    dy = yi - yj
                    d = np.sqrt(dx*dx + dy*dy + 1e-12)
                    
                    v = ri + rj - d
                    penalty += softplus(v, beta)
                    g = softplus_grad(v, beta)
                    
                    grad[3*i] += -g * dx / d
                    grad[3*i+1] += -g * dy / d
                    grad[3*i+2] += g
                    grad[3*j] += g * dx / d
                    grad[3*j+1] += g * dy / d
                    grad[3*j+2] += g
            
            # Equalization pressure: penalize variance of radii
            if equalize_weight > 0.0:
                radii = c[:, 2]
                r_mean = np.mean(radii)
                for i in range(N):
                    diff = radii[i] - r_mean
                    penalty += equalize_weight * diff * diff
                    grad[3*i+2] += 2.0 * equalize_weight * diff
            
            obj = val + penalty
            for i in range(N):
                grad[3*i+2] += -1.0
            
            return obj, grad
        
        result = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True,
                         options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        
        circles = result.x.reshape(-1, 3)
        val = evaluate(circles, w, h)
        return circles, val
    
    def multi_phase_polish(circles0, w, h, time_budget=60.0):
        """Multi-phase polish with increasing beta for tighter convergence."""
        current = circles0.copy()
        phases = [
            (50, 100, 0.01),    # (max_iter, beta, equalize_weight)
            (100, 200, 0.005),
            (150, 400, 0.001),
            (200, 800, 0.0),
            (300, 1500, 0.0),
        ]
        
        elapsed = 0.0
        for max_iter, beta, eq_w in phases:
            if elapsed > time_budget * 0.8:
                break
            t0 = time.time()
            current, val = optimize(current, w, h, max_iter=max_iter, beta=beta, equalize_weight=eq_w)
            elapsed += time.time() - t0
        
        return current, evaluate(current, w, h)
    
    best_value = -np.inf
    best_result = None
    best_wh = None
    
    # Strategy 1: Hexagonal lattice initialization
    circles0, w, h, val0 = hex_lattice_solution()
    if val0 > best_value:
        best_value = val0
        best_result = circles0.copy()
        best_wh = (w, h)
    
    # Multi-phase polish from hex lattice
    if time.time() - start_time < deadline - 40:
        polished, val = multi_phase_polish(circles0, w, h, time_budget=50.0)
        if val > best_value:
            best_value = val
            best_result = polished.copy()
            best_wh = (w, h)
    
    # Strategy 2: Try multiple aspect ratios with grid inits
    if time.time() - start_time < deadline - 20:
        aspects = [1.0, 0.85, 1.15, 0.75, 1.25, 0.65, 1.4, 0.55, 1.6, 0.5]
        for aspect in aspects:
            if time.time() - start_time > deadline - 10:
                break
            
            w2 = 2.0 * aspect / (1.0 + aspect)
            h2 = 2.0 / (1.0 + aspect)
            
            # Simple grid initialization
            cols = 5
            rows = 5
            cell_w = w2 / cols
            cell_h = h2 / rows
            circles_grid = []
            for i in range(N):
                c = i % cols
                r = i // cols
                cx = (c + 0.5) * cell_w
                cy = (r + 0.5) * cell_h
                rad = 0.5 * min(cell_w, cell_h) - 1e-4
                circles_grid.append([cx, cy, rad])
            circles_grid = np.array(circles_grid)
            
            val_grid = evaluate(circles_grid, w2, h2)
            if val_grid > best_value:
                best_value = val_grid
                best_result = circles_grid.copy()
                best_wh = (w2, h2)
            
            circles_opt, val_opt = optimize(circles_grid, w2, h2, max_iter=200, beta=100.0)
            if val_opt > best_value:
                best_value = val_opt
                best_result = circles_opt.copy()
                best_wh = (w2, h2)
    
    # Final high-precision polish on best result
    if best_result is not None and best_wh is not None and time.time() - start_time < deadline - 5:
        w, h = best_wh
        polished, val = multi_phase_polish(best_result, w, h, time_budget=20.0)
        if val > best_value:
            best_value = val
            best_result = polished.copy()
    
    if best_result is None:
        # Fallback
        circles, w, h, _ = hex_lattice_solution()
        best_result = circles
    
    return best_result
