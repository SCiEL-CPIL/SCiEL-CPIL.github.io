import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 115.0  # Safety margin before 120s kill

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_circles = None
    
    # Objective function for L-BFGS-B (Penalty method)
    def objective_penalty(params, W, H):
        x = params[0::3]
        y = params[1::3]
        r = params[2::3]
        
        # Base objective: maximize sum of radii
        val = -np.sum(r)
        
        # Vectorized distance calculation
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        d2 = dx**2 + dy**2
        d = np.sqrt(d2)
        
        # Required distance
        req_d = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Exclude diagonal
        np.fill_diagonal(req_d, 0)
        np.fill_diagonal(d, np.inf)
        
        # Overlap penalty
        violations = np.maximum(0, req_d - d)
        overlap_penalty = np.sum(violations**2)
        
        # Boundary penalty
        v_left = np.maximum(0, r - x)
        v_right = np.maximum(0, r - (W - x))
        v_bottom = np.maximum(0, r - y)
        v_top = np.maximum(0, r - (H - y))
        
        boundary_penalty = np.sum(v_left**2) + np.sum(v_right**2) + np.sum(v_bottom**2) + np.sum(v_top**2)
        
        weight = 100.0
        val += weight * (overlap_penalty + boundary_penalty)
        
        return val

    # Constraints for SLSQP
    def cons_inside(vars_, W, H):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        return np.concatenate([
            x - r,
            W - x - r,
            y - r,
            H - y - r
        ])
    
    def cons_non_overlap(vars_):
        x = vars_[0::3]
        y = vars_[1::3]
        r = vars_[2::3]
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                cons.append(dist - r[i] - r[j])
        return np.array(cons)
    
    def cons_positive_r(vars_):
        r = vars_[2::3]
        return r

    def is_valid(circles, W, H):
        x = circles[:, 0]
        y = circles[:, 1]
        r = circles[:, 2]
        
        if np.any(r < 0): return False
        if np.any(x < r - 1e-6) or np.any(x > W - r + 1e-6): return False
        if np.any(y < r - 1e-6) or np.any(y > H - r + 1e-6): return False
        
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        d = np.sqrt(dx**2 + dy**2)
        
        req_d = r[:, np.newaxis] + r[np.newaxis, :]
        np.fill_diagonal(req_d, 0)
        np.fill_diagonal(d, np.inf)
        
        if np.any(req_d > d + 1e-6): return False
        return True

    def refine_with_slsqp(x0, W, H):
        """
        Takes a valid or near-valid packing from L-BFGS-B and refines it with SLSQP.
        """
        constraints = [
            {'type': 'ineq', 'fun': lambda v: cons_inside(v, W, H)},
            {'type': 'ineq', 'fun': cons_non_overlap},
            {'type': 'ineq', 'fun': cons_positive_r}
        ]
        
        bounds = []
        for i in range(N):
            bounds.append((0.0, W)) # x
            bounds.append((0.0, H)) # y
            bounds.append((0.0, 1.0)) # r
            
        try:
            res = minimize(
                lambda v: -np.sum(v[2::3]),
                x0,
                method='SLSQP',
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            if res.success or (res.fun < -1e-5): # Check if it improved
                x_res = res.x.reshape(N, 3)
                # SLSQP might be slightly off, validate strictly
                if is_valid(x_res, W, H):
                    return x_res
                else:
                    # Try to shrink slightly to make valid
                    x_res = x_res.copy()
                    for _ in range(20):
                        if is_valid(x_res, W, H):
                            return x_res
                        x_res[:, 2] *= 0.995
                    return None
            return None
        except Exception:
            return None

    # Aspect ratios to try: W + H = 2
    # We scan a range of widths. Optimal is often near square but can vary.
    widths = np.arange(0.7, 1.31, 0.05)
    
    np.random.seed(42)
    
    for W in widths:
        if time.time() - start_time > DEADLINE - 10:
            break
        H = 2.0 - W
        
        # Try a few restarts for this aspect ratio
        for restart in range(3):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # 1. Initialization
            # Strategy: Grid layout with jitter
            # 5x4 grid = 20, add 1 in center or edge? 
            # Better: 5x5 grid, pick 21.
            # Or Hexagonal. Let's stick to Grid for robustness.
            
            cols = 5
            rows = 5
            # If W < H, maybe swap grid orientation?
            # For now, assume square-ish.
            
            x0 = np.zeros(3*N)
            
            # Generate 25 grid points, pick 21
            indices = np.random.choice(25, N, replace=False)
            
            for k, idx in enumerate(indices):
                i = idx // 5
                j = idx % 5
                
                cx = (j + 0.5) * W / 5
                cy = (i + 0.5) * H / 5
                
                # Add jitter
                jitter = 0.02
                cx += np.random.uniform(-jitter, jitter)
                cy += np.random.uniform(-jitter, jitter)
                
                # Clamp
                cx = np.clip(cx, 0.01, W - 0.01)
                cy = np.clip(cy, 0.01, H - 0.01)
                
                x0[3*k] = cx
                x0[3*k+1] = cy
                x0[3*k+2] = 0.08 # Initial radius guess
                
            # 2. Phase 1: L-BFGS-B with Penalty
            try:
                res1 = minimize(
                    objective_penalty, 
                    x0, 
                    args=(W, H), 
                    method='L-BFGS-B', 
                    bounds=[(0, W), (0, H), (0, 1)] * N,
                    options={'maxiter': 200, 'ftol': 1e-6}
                )
                
                x_opt1 = res1.x.reshape(N, 3)
                
                # Validate Phase 1 result
                if is_valid(x_opt1, W, H):
                    score1 = np.sum(x_opt1[:, 2])
                    
                    # 3. Phase 2: SLSQP Refinement
                    x_opt2 = refine_with_slsqp(x_opt1.flatten(), W, H)
                    
                    if x_opt2 is not None:
                        score2 = np.sum(x_opt2[:, 2])
                        if score2 > best_score:
                            best_score = score2
                            best_circles = x_opt2
                    else:
                        # Use Phase 1 if Phase 2 failed
                        if score1 > best_score:
                            best_score = score1
                            best_circles = x_opt1
                            
            except Exception:
                pass

    # Fallback if nothing found
    if best_circles is None:
        # Simple grid in 1x1 square
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([
                        (i + 0.5) / side,
                        (j + 0.5) / side,
                        0.5 / side - 0.01
                    ])
        best_circles = np.array(cells[:N])
        
    return best_circles
