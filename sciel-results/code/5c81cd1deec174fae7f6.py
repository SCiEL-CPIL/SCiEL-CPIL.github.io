import time
import numpy as np
from scipy.optimize import minimize

N = 21
_DEADLINE = time.time() + 135.0  # safety margin under 150s kill

def _validate(arr):
    if arr is None or arr.shape != (N, 3):
        return None
    if not np.all(np.isfinite(arr)):
        return None
    w, h, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r < 0):
        return None
    
    min_x = np.min(w)
    max_x = np.max(w)
    min_y = np.min(h)
    max_y = np.max(h)
    
    BW = max_x - min_x
    BH = max_y - min_y
    if 2.0 * (BW + BH) > 4.0 + 1e-6:
        return None
    
    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(w[i] - w[j], h[i] - h[j])
            if d < r[i] + r[j] - 1e-7:
                return None
    return arr

def _solve_aspect(a, rng, time_left_fn):
    # w/h = a, w+h=2 => h = 2/(a+1), w = 2a/(a+1)
    h = 2.0 / (a + 1.0)
    w = 2.0 - h

    n = N
    best_val = -np.inf
    best_arr = None
    
    # Number of restarts depends on time
    # L-BFGS-B is fast. Let's try 10-20 restarts per aspect.
    n_restarts = 15
    
    for k in range(n_restarts):
        if time_left_fn() < 5:
            break
            
        # Random initial guess
        # Positions in [0, w] x [0, h]
        # Radii small
        x0 = np.empty(3 * n)
        for i in range(n):
            x0[2*i] = rng.uniform(0.1*w, 0.9*w)
            x0[2*i+1] = rng.uniform(0.1*h, 0.9*h)
            x0[2*n+i] = rng.uniform(0.05, 0.1)
            
        # Objective function
        def objective(x):
            pos_x = x[:2*n:2]
            pos_y = x[1:2*n:2]
            radii = x[2*n:]
            
            # Penalty for overlaps
            # Vectorized calculation of all pair distances is O(N^2)
            # For N=21, 210 pairs.
            pen_overlap = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    dx = pos_x[i] - pos_x[j]
                    dy = pos_y[i] - pos_y[j]
                    dist = np.sqrt(dx*dx + dy*dy)
                    r_sum = radii[i] + radii[j]
                    if dist < r_sum:
                        pen_overlap += (r_sum - dist)**2
            
            # Penalty for boundaries
            # Circles must be inside [0, w] x [0, h]
            pen_bound = 0.0
            for i in range(n):
                r = radii[i]
                if pos_x[i] < r: pen_bound += (r - pos_x[i])**2
                if pos_x[i] > w - r: pen_bound += (pos_x[i] - (w-r))**2
                if pos_y[i] < r: pen_bound += (r - pos_y[i])**2
                if pos_y[i] > h - r: pen_bound += (pos_y[i] - (h-r))**2
            
            # Term to maximize radii
            # We minimize penalty - lambda * sum(radii)
            lambda_val = 10.0 
            return pen_overlap + pen_bound - lambda_val * np.sum(radii)
        
        # Gradient (optional, L-BFGS-B can estimate it)
        # Providing gradient is faster.
        def jac(x):
            pos_x = x[:2*n:2]
            pos_y = x[1:2*n:2]
            radii = x[2*n:]
            grad = np.zeros(3*n)
            
            # Overlap gradient
            for i in range(n):
                for j in range(i+1, n):
                    dx = pos_x[i] - pos_x[j]
                    dy = pos_y[i] - pos_y[j]
                    dist = np.sqrt(dx*dx + dy*dy) + 1e-9
                    r_sum = radii[i] + radii[j]
                    if dist < r_sum:
                        diff = r_sum - dist
                        # d/dpos of (r_sum - dist)^2 = 2*diff * d(dist)/dpos
                        # d(dist)/dxi = -dx/dist, d(dist)/dxi_j = dx/dist
                        grad[2*i] += 2 * diff * (-dx/dist)
                        grad[2*i+1] += 2 * diff * (-dy/dist)
                        grad[2*j] += 2 * diff * (dx/dist)
                        grad[2*j+1] += 2 * diff * (dy/dist)
                        # d/dri of (r_sum - dist)^2 = 2*diff
                        grad[2*n+i] += 2 * diff
                        grad[2*n+j] += 2 * diff
            
            # Boundary gradient
            for i in range(n):
                r = radii[i]
                if pos_x[i] < r:
                    diff = r - pos_x[i]
                    grad[2*i] += 2 * diff * (-1) # d/dxi (r-xi)^2 = -2(r-xi)
                    grad[2*n+i] += 2 * diff
                if pos_x[i] > w - r:
                    diff = pos_x[i] - (w-r)
                    grad[2*i] += 2 * diff
                    grad[2*n+i] += 2 * diff * (-1)
                if pos_y[i] < r:
                    diff = r - pos_y[i]
                    grad[2*i+1] += 2 * diff * (-1)
                    grad[2*n+i] += 2 * diff
                if pos_y[i] > h - r:
                    diff = pos_y[i] - (h-r)
                    grad[2*i+1] += 2 * diff
                    grad[2*n+i] += 2 * diff * (-1)
            
            # Radius term gradient: -lambda
            grad[2*n:] -= lambda_val
            
            return grad

        bounds = [(0.0, w)] * n + [(0.0, h)] * n + [(0.0, 1.0)] * n
        
        try:
            res = minimize(objective, x0, jac=jac, method='L-BFGS-B', bounds=bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success:
                x = res.x
                arr = np.zeros((n, 3))
                arr[:, 0] = x[:2*n:2]
                arr[:, 1] = x[1:2*n:2]
                arr[:, 2] = x[2*n:]
                
                # Validate
                if _validate(arr) is not None:
                    val = np.sum(arr[:, 2])
                    if val > best_val:
                        best_val = val
                        best_arr = arr
        except Exception:
            pass

    return best_arr

def construct():
    t_start = time.time()
    rng = np.random.default_rng(42)
    
    best_overall = None
    best_val = -np.inf
    
    aspects = [1.0, 1.2, 0.83, 1.5, 0.67, 2.0, 0.5]
    
    # Time budget allocation
    # 140s available.
    # We will iterate aspects and restarts.
    
    for a in aspects:
        if time.time() > _DEADLINE:
            break
        arr = _solve_aspect(a, rng, lambda: _DEADLINE - time.time())
        if arr is not None:
            val = np.sum(arr[:, 2])
            if val > best_val:
                best_val = val
                best_overall = arr
    
    if best_overall is None:
        # Fallback
        side = 5
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) < N:
                    cells.append([0.5 + i*0.2, 0.5 + j*0.2, 0.09])
        best_overall = np.array(cells[:N])
        
    return best_overall
