"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START

def _penalty(x, y, r, margin=0.0):
    """Return a penalty value for a circle configuration. Lower is better."""
    # Boundaries
    pen = 0.0
    for i in range(len(x)):
        # x boundary
        if x[i] < margin:
            pen += (margin - x[i]) ** 2
        if x[i] > 1.0 - margin:
            pen += (x[i] - (1.0 - margin)) ** 2
        # y boundary
        if y[i] < margin:
            pen += (margin - y[i]) ** 2
        if y[i] > 1.0 - margin:
            pen += (y[i] - (1.0 - margin)) ** 2
        # radius must be positive
        if r[i] < 1e-6:
            pen += (1e-6 - r[i]) ** 2
    # Pairwise distances
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = r[i] + r[j]
            if dist < min_dist:
                pen += (min_dist - dist) ** 2
    return pen

def _is_valid(x, y, r, tol=1e-5):
    """Check if configuration is valid."""
    for i in range(len(x)):
        if x[i] < -tol or x[i] > 1 + tol or y[i] < -tol or y[i] > 1 + tol:
            return False
        if r[i] < 1e-6:
            return False
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _sum_radii(x, y, r):
    return np.sum(r)

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    best_x = None
    best_y = None
    best_r = None
    best_sum = 0.0
    
    # Initial grid configuration
    side = int(np.ceil(np.sqrt(N)))
    x_init = np.array([(i + 0.5) / side for i in range(side) for j in range(side)][:N])
    y_init = np.array([(j + 0.5) / side for i in range(side) for j in range(side)][:N])
    r_init = np.full(N, 0.5 / side - 1e-3)
    
    best_x = x_init.copy()
    best_y = y_init.copy()
    best_r = r_init.copy()
    best_sum = _sum_radii(best_x, best_y, best_r)
    
    # Try multiple random initializations and local optimizations
    n_attempts = 0
    max_attempts = 50
    
    while n_attempts < max_attempts and time.time() < deadline:
        n_attempts += 1
        
        # Random initialization
        x = np.random.uniform(0.1, 0.9, N)
        y = np.random.uniform(0.1, 0.9, N)
        r = np.random.uniform(0.02, 0.12, N)
        
        # Optimize: maximize sum of radii subject to constraints
        # We minimize negative sum of radii plus penalty
        
        def objective(vars):
            x = vars[:N]
            y = vars[N:2*N]
            r = vars[2*N:]
            return -np.sum(r) + 100.0 * _penalty(x, y, r)
        
        # Initial guess
        x0 = np.concatenate([x, y, r])
        
        # Try a few different optimization methods
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                          options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = res.x[2*N:]
            
            # Project to feasible region
            x_opt = np.clip(x_opt, 0, 1)
            y_opt = np.clip(y_opt, 0, 1)
            r_opt = np.maximum(r_opt, 1e-6)
            
            # Check validity
            if _is_valid(x_opt, y_opt, r_opt):
                s = _sum_radii(x_opt, y_opt, r_opt)
                if s > best_sum:
                    best_sum = s
                    best_x = x_opt.copy()
                    best_y = y_opt.copy()
                    best_r = r_opt.copy()
        except Exception:
            pass
        
        # Also try L-BFGS-B with bounds
        if time.time() < deadline - 10:
            try:
                bounds = []
                for i in range(N):
                    bounds.append((0, 1))  # x
                for i in range(N):
                    bounds.append((0, 1))  # y
                for i in range(N):
                    bounds.append((1e-6, 0.5))  # r
                
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 2000, 'ftol': 1e-8})
                
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:]
                
                x_opt = np.clip(x_opt, 0, 1)
                y_opt = np.clip(y_opt, 0, 1)
                r_opt = np.maximum(r_opt, 1e-6)
                
                if _is_valid(x_opt, y_opt, r_opt):
                    s = _sum_radii(x_opt, y_opt, r_opt)
                    if s > best_sum:
                        best_sum = s
                        best_x = x_opt.copy()
                        best_y = y_opt.copy()
                        best_r = r_opt.copy()
            except Exception:
                pass
    
    # Final refinement with smaller steps
    if time.time() < deadline - 20:
        # Local refinement around best solution
        for _ in range(10):
            if time.time() > deadline - 20:
                break
            x = best_x + np.random.normal(0, 0.01, N)
            y = best_y + np.random.normal(0, 0.01, N)
            r = best_r + np.random.normal(0, 0.005, N)
            
            x = np.clip(x, 0, 1)
            y = np.clip(y, 0, 1)
            r = np.maximum(r, 1e-6)
            
            try:
                res = minimize(objective, np.concatenate([x, y, r]), method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
                
                x_opt = res.x[:N]
                y_opt = res.x[N:2*N]
                r_opt = res.x[2*N:]
                
                x_opt = np.clip(x_opt, 0, 1)
                y_opt = np.clip(y_opt, 0, 1)
                r_opt = np.maximum(r_opt, 1e-6)
                
                if _is_valid(x_opt, y_opt, r_opt):
                    s = _sum_radii(x_opt, y_opt, r_opt)
                    if s > best_sum:
                        best_sum = s
                        best_x = x_opt.copy()
                        best_y = y_opt.copy()
                        best_r = r_opt.copy()
            except Exception:
                pass
    
    # Ensure the result is valid; if not, fall back to grid
    if not _is_valid(best_x, best_y, best_r):
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return np.column_stack([best_x, best_y, best_r])

# EVOLVE-BLOCK-END
