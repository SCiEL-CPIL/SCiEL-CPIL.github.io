import numpy as np
from scipy.optimize import minimize
import time

N = 21
DEADLINE = 120.0  # seconds, leave margin before 150s

def construct():
    start_time = time.time()
    
    def time_left():
        return DEADLINE - (time.time() - start_time)
    
    best_result = None
    best_value = -np.inf
    
    def evaluate(arr):
        """Evaluate a configuration: returns (value, is_valid)"""
        # arr: (N, 3) with [x, y, r]
        w = arr[:, 0].max() + arr[:, 2].max()  # This isn't right
        # Actually we need to find the bounding rectangle
        # Let me reconsider: the rectangle is fixed, circles are inside it
        # I'll handle this differently
        pass
    
    # Let me restructure: I'll optimize over (w, centers, radii)
    # where h = 2 - w
    
    def soft_penalty(violations, beta=50.0):
        """Log-sum-exp soft penalty"""
        if len(violations) == 0:
            return 0.0
        violations = np.array(violations)
        max_v = np.max(violations)
        if max_v < 0:
            return 0.0
        return np.log(np.sum(np.exp(beta * violations))) / beta + max_v
    
    def objective(params, w):
        """params: (N*3,) = [x_1, y_1, r_1, x_2, y_2, r_2, ...]"""
        h = 2.0 - w
        if h <= 0 or w <= 0:
            return 1e10
        
        centers = params[:N*2].reshape(N, 2)
        radii = params[N*2:]
        
        # Objective: maximize sum of radii -> minimize -sum(radii)
        obj = -np.sum(radii)
        
        # Constraint violations
        violations = []
        
        # Boundary constraints: x_i >= r_i, x_i <= w-r_i, y_i >= r_i, y_i <= h-r_i
        for i in range(N):
            x, y, r = centers[i, 0], centers[i, 1], radii[i]
            if r < 0:
                violations.append(1.0)
                continue
            violations.append(max(0, r - x))
            violations.append(max(0, r - (w - x)))
            violations.append(max(0, r - y))
            violations.append(max(0, r - (h - y)))
        
        # Pairwise constraints
        for i in range(N):
            for j in range(i+1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = radii[i] + radii[j]
                if min_dist > 0:
                    violations.append(max(0, min_dist - np.sqrt(dist_sq)))
        
        penalty = soft_penalty(violations, beta=100.0)
        return obj + penalty
    
    def make_initial(w, h):
        """Create initial configuration"""
        centers = np.zeros((N, 2))
        radii = np.zeros(N)
        
        # Try grid-like arrangement
        # 5 columns x 4 rows = 20, plus 1
        cols = 5
        rows = 4
        count = 0
        for j in range(rows):
            for i in range(cols):
                if count >= N:
                    break
                x = (i + 0.5) * w / cols
                y = (j + 0.5) * h / rows
                centers[count] = [x, y]
                r = min(w/cols, h/rows) * 0.45
                radii[count] = r
                count += 1
            if count >= N:
                break
        
        # Add extra circle if needed
        if count < N:
            centers[count] = [w/2, h/2]
            radii[count] = 0.05
        
        return np.concatenate([centers.flatten(), radii])
    
    def try_configuration(w, n_restarts=3):
        """Try to optimize for given width w"""
        h = 2.0 - w
        if h <= 0.01 or w <= 0.01:
            return None
        
        best_local = None
        best_local_val = -np.inf
        
        for restart in range(n_restarts):
            if time_left() < 10:
                break
            
            params = make_initial(w, h)
            
            # Add perturbation for multi-start
            if restart > 0:
                noise = np.random.randn(len(params)) * 0.01 * np.array(
                    [1.0, 1.0, 0.1] * N  # scale: x, y, r
                )
                params = params + noise
                # Clamp radii to be positive
                params[N*2:] = np.maximum(params[N*2:], 0.001)
            
            try:
                res = minimize(
                    objective, params, args=(w,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
            except Exception:
                continue
            
            if res.success or res.fun < best_local_val:
                if res.fun < best_local_val:
                    best_local_val = res.fun
                    best_local = res.x
        
        if best_local is None:
            return None
        
        # Extract final configuration
        centers = best_local[:N*2].reshape(N, 2)
        radii = best_local[N*2:]
        
        # Check validity
        valid = True
        for i in range(N):
            x, y, r = centers[i]
            if r < 0 or x < r - 1e-6 or x > w - r + 1e-6 or y < r - 1e-6 or y > h - r + 1e-6:
                valid = False
                break
        
        if valid:
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i, 0] - centers[j, 0]
                    dy = centers[i, 1] - centers[j, 1]
                    if np.sqrt(dx*dx + dy*dy) < radii[i] + radii[j] - 1e-6:
                        valid = False
                        break
                if not valid:
                    break
        
        if not valid:
            # Try to fix by shrinking radii
            for i in range(N):
                x, y = centers[i]
                max_r = min(x, w-x, y, h-y)
                radii[i] = min(radii[i], max_r * 0.999)
            # Re-check
            valid = True
            for i in range(N):
                x, y, r = centers[i]
                if r < 0 or x < r - 1e-4 or x > w - r + 1e-4 or y < r - 1e-4 or y > h - r + 1e-4:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i+1, N):
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        if np.sqrt(dx*dx + dy*dy) < radii[i] + radii[j] - 1e-4:
                            valid = False
                            break
                    if not valid:
                        break
            if not valid:
                return None
        
        result = np.column_stack([centers, radii])
        value = np.sum(radii)
        return result, value
    
    # Try different aspect ratios
    aspect_ratios = [1.0, 0.8, 1.25, 0.7, 1.4, 0.6, 1.667, 0.5, 2.0]
    
    for ar in aspect_ratios:
        if time_left() < 15:
            break
        
        w = 2.0 * ar / (1 + ar)
        h = 2.0 - w
        
        res = try_configuration(w, n_restarts=3)
        if res is not None:
            result, value = res
            if value > best_value:
                best_value = value
                best_result = result.copy()
    
    # Final polish: try to improve the best result
    if best_result is not None and time_left() > 20:
        # Do a few more optimization steps on the best
        w_best = None
        # Reconstruct w from the best result... this is tricky
        # Let me just try a final optimization with the best aspect ratio
        
    if best_result is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
