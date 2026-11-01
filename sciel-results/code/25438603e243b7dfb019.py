"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    def validate_and_fix(x, y, r):
        """Validate circles are disjoint and inside unit square, fix if needed."""
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        r = np.array(r, dtype=float)
        
        # Clip radii to be positive
        r = np.maximum(r, 1e-6)
        
        # Clip positions to keep circles inside
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        
        # Iteratively fix overlaps by shrinking radii
        for iteration in range(50):
            if time.time() > deadline:
                break
            fixed = False
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx*dx + dy*dy)
                    min_d = r[i] + r[j]
                    if d < min_d - 1e-9:
                        # Shrink both radii proportionally
                        overlap = min_d - d
                        shrink = overlap / 2 + 1e-8
                        r[i] = max(r[i] - shrink, 1e-6)
                        r[j] = max(r[j] - shrink, 1e-6)
                        fixed = True
            if not fixed:
                break
        
        # Final validation
        valid = True
        for i in range(N):
            if x[i] < r[i] - 1e-6 or x[i] + r[i] > 1.0 + 1e-6:
                valid = False
            if y[i] < r[i] - 1e-6 or y[i] + r[i] > 1.0 + 1e-6:
                valid = False
            for j in range(i + 1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        return x, y, r, valid
    
    def objective_and_grad(vars):
        """Objective: maximize sum(r) with penalties for violations."""
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        obj = -np.sum(r)
        grad = np.zeros_like(vars)
        grad[2::3] = -1.0
        
        C = 500.0  # Penalty weight
        
        # Pairwise overlap penalties
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                d_sq = dx*dx + dy*dy
                d = np.sqrt(d_sq) if d_sq > 1e-12 else 1e-6
                r_sum = r[i] + r[j]
                
                if d < r_sum:
                    V = r_sum - d
                    coeff = 2.0 * C * V
                    inv_d = 1.0 / d
                    
                    grad[3*i + 0] += coeff * (-dx * inv_d)
                    grad[3*i + 1] += coeff * (-dy * inv_d)
                    grad[3*i + 2] += coeff
                    grad[3*j + 0] += coeff * (dx * inv_d)
                    grad[3*j + 1] += coeff * (dy * inv_d)
                    grad[3*j + 2] += coeff
                    obj += C * V * V
        
        # Boundary penalties
        for i in range(N):
            if x[i] < r[i]:
                V = r[i] - x[i]
                coeff = 2.0 * C * V
                grad[3*i + 0] += coeff * (-1.0)
                grad[3*i + 2] += coeff
                obj += C * V * V
            if x[i] + r[i] > 1.0:
                V = x[i] + r[i] - 1.0
                coeff = 2.0 * C * V
                grad[3*i + 0] += coeff
                grad[3*i + 2] += coeff
                obj += C * V * V
            if y[i] < r[i]:
                V = r[i] - y[i]
                coeff = 2.0 * C * V
                grad[3*i + 1] += coeff * (-1.0)
                grad[3*i + 2] += coeff
                obj += C * V * V
            if y[i] + r[i] > 1.0:
                V = y[i] + r[i] - 1.0
                coeff = 2.0 * C * V
                grad[3*i + 1] += coeff
                grad[3*i + 2] += coeff
                obj += C * V * V
        
        return obj, grad
    
    def optimize_from_start(x0, max_iter=1000, lr_init=0.02, momentum=0.9):
        """Run gradient descent with momentum from a starting point."""
        vars_ = x0.copy()
        velocity = np.zeros_like(vars_)
        lr = lr_init
        best_obj = np.inf
        best_vars = vars_.copy()
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            obj, grad = objective_and_grad(vars_)
            
            # Track best by sum of radii (with validity check)
            x = vars_[0::3]
            y = vars_[1::3]
            r = vars_[2::3]
            current_sum = np.sum(r)
            
            # Check if this is valid
            valid = True
            for i in range(N):
                if x[i] < r[i] - 1e-5 or x[i] + r[i] > 1.0 + 1e-5:
                    valid = False
                    break
                if y[i] < r[i] - 1e-5 or y[i] + r[i] > 1.0 + 1e-5:
                    valid = False
                    break
                if valid:
                    for j in range(i + 1, N):
                        d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                        if d < r[i] + r[j] - 1e-5:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid and obj < best_obj:
                best_obj = obj
                best_vars = vars_.copy()
            
            # Momentum update
            velocity = momentum * velocity - lr * grad
            vars_ = vars_ + velocity
            
            # Project: keep radii positive, positions in [0,1]
            vars_[2::3] = np.maximum(vars_[2::3], 0.005)
            vars_[0::3] = np.clip(vars_[0::3], 0.0, 1.0)
            vars_[1::3] = np.clip(vars_[1::3], 0.0, 1.0)
            
            # Decay learning rate
            lr *= 0.998
            
            # Early stopping if gradient is small
            if np.max(np.abs(grad)) < 1e-6:
                break
        
        return best_vars
    
    def try_grid_init(rows, cols, jitter=0.0):
        """Create a grid initialization with optional jitter."""
        x = np.zeros(N)
        y = np.zeros(N)
        r = np.zeros(N)
        
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N:
                    break
                cx = (j + 0.5) / cols
                cy = (i + 0.5) / rows
                if jitter > 0:
                    cx += np.random.uniform(-jitter, jitter)
                    cy += np.random.uniform(-jitter, jitter)
                x[idx] = np.clip(cx, 0.01, 0.99)
                y[idx] = np.clip(cy, 0.01, 0.99)
                r[idx] = 0.04
                idx += 1
        
        # Fill remaining with random positions
        while idx < N:
            x[idx] = np.random.uniform(0.1, 0.9)
            y[idx] = np.random.uniform(0.1, 0.9)
            r[idx] = 0.04
            idx += 1
        
        vars_ = np.zeros(3 * N)
        vars_[0::3] = x
        vars_[1::3] = y
        vars_[2::3] = r
        return vars_
    
    def try_random_init(n_random=5):
        """Try multiple random initializations."""
        best_vars = None
        best_sum = -np.inf
        
        for _ in range(n_random):
            x = np.random.uniform(0.1, 0.9, N)
            y = np.random.uniform(0.1, 0.9, N)
            r = np.full(N, 0.04)
            
            vars_ = np.zeros(3 * N)
            vars_[0::3] = x
            vars_[1::3] = y
            vars_[2::3] = r
            
            result = optimize_from_start(vars_, max_iter=500)
            current_sum = np.sum(result[2::3])
            
            # Check validity
            x_r = result[0::3]
            y_r = result[1::3]
            r_r = result[2::3]
            valid = True
            for i in range(N):
                if x_r[i] < r_r[i] - 1e-5 or x_r[i] + r_r[i] > 1.0 + 1e-5:
                    valid = False
                    break
                if y_r[i] < r_r[i] - 1e-5 or y_r[i] + r_r[i] > 1.0 + 1e-5:
                    valid = False
                    break
                if valid:
                    for j in range(i + 1, N):
                        d = np.sqrt((x_r[i]-x_r[j])**2 + (y_r[i]-y_r[j])**2)
                        if d < r_r[i] + r_r[j] - 1e-5:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid and current_sum > best_sum:
                best_sum = current_sum
                best_vars = result
        
        return best_vars
    
    # Strategy: Try multiple initialization strategies and keep the best
    best_result = None
    best_sum = -np.inf
    
    # 1. Grid initialization (5x7 or 6x6)
    for rows, cols in [(5, 7), (6, 6), (4, 8)]:
        if time.time() > deadline:
            break
        x0 = try_grid_init(rows, cols, jitter=0.02)
        result = optimize_from_start(x0, max_iter=800)
        current_sum = np.sum(result[2::3])
        
        x_r = result[0::3]
        y_r = result[1::3]
        r_r = result[2::3]
        valid = True
        for i in range(N):
            if x_r[i] < r_r[i] - 1e-5 or x_r[i] + r_r[i] > 1.0 + 1e-5:
                valid = False
                break
            if y_r[i] < r_r[i] - 1e-5 or y_r[i] + r_r[i] > 1.0 + 1e-5:
                valid = False
                break
            if valid:
                for j in range(i + 1, N):
                    d = np.sqrt((x_r[i]-x_r[j])**2 + (y_r[i]-y_r[j])**2)
                    if d < r_r[i] + r_r[j] - 1e-5:
                        valid = False
                        break
            if not valid:
                break
        
        if valid and current_sum > best_sum:
            best_sum = current_sum
            best_result = result
    
    # 2. Random restarts
    if time.time() < deadline - 30:
        random_result = try_random_init(n_random=3)
        if random_result is not None:
            current_sum = np.sum(random_result[2::3])
            x_r = random_result[0::3]
            y_r = random_result[1::3]
            r_r = random_result[2::3]
            valid = True
            for i in range(N):
                if x_r[i] < r_r[i] - 1e-5 or x_r[i] + r_r[i] > 1.0 + 1e-5:
                    valid = False
                    break
                if y_r[i] < r_r[i] - 1e-5 or y_r[i] + r_r[i] > 1.0 + 1e-5:
                    valid = False
                    break
                if valid:
                    for j in range(i + 1, N):
                        d = np.sqrt((x_r[i]-x_r[j])**2 + (y_r[i]-y_r[j])**2)
                        if d < r_r[i] + r_r[j] - 1e-5:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid and current_sum > best_sum:
                best_sum = current_sum
                best_result = random_result
    
    # 3. Final polish: if we have a good result, do a high-precision refinement
    if best_result is not None and time.time() < deadline - 10:
        x0 = best_result.copy()
        polished = optimize_from_start(x0, max_iter=500, lr_init=0.005, momentum=0.95)
        current_sum = np.sum(polished[2::3])
        
        x_r = polished[0::3]
        y_r = polished[1::3]
        r_r = polished[2::3]
        valid = True
        for i in range(N):
            if x_r[i] < r_r[i] - 1e-6 or x_r[i] + r_r[i] > 1.0 + 1e-6:
                valid = False
                break
            if y_r[i] < r_r[i] - 1e-6 or y_r[i] + r_r[i] > 1.0 + 1e-6:
                valid = False
                break
            if valid:
                for j in range(i + 1, N):
                    d = np.sqrt((x_r[i]-x_r[j])**2 + (y_r[i]-y_r[j])**2)
                    if d < r_r[i] + r_r[j] - 1e-6:
                        valid = False
                        break
            if not valid:
                break
        
        if valid and current_sum > best_sum:
            best_sum = current_sum
            best_result = polished
    
    # Fallback if no valid result found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    x_best = best_result[0::3]
    y_best = best_result[1::3]
    r_best = best_result[2::3]
    
    # Final validation and fix
    x_best, y_best, r_best, valid = validate_and_fix(x_best, y_best, r_best)
    
    if not valid:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return np.column_stack((x_best, y_best, r_best))

# EVOLVE-BLOCK-END
