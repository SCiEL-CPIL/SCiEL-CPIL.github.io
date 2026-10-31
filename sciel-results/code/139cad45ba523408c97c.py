"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _penalty(x, y, r, x2, y2, r2):
    """Penalty for overlap and boundary violations."""
    # Boundary: circle must be inside [0,1]^2
    pen = 0.0
    # Left/Right
    if x - r < 0:
        pen += (x - r)**2
    if x + r > 1:
        pen += (x + r - 1)**2
    # Bottom/Top
    if y - r < 0:
        pen += (y - r)**2
    if y + r > 1:
        pen += (y + r - 1)**2
    
    # Overlap with other circles
    dx = x - x2
    dy = y - y2
    dist = np.sqrt(dx**2 + dy**2)
    min_dist = r + r2
    if dist < min_dist:
        pen += (min_dist - dist)**2
    
    return pen

def _objective_and_grad(params, N):
    """
    params: array of shape (3*N,) = [x1,y1,r1, x2,y2,r2, ...]
    Returns: objective = -sum(r) + penalty
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Objective: maximize sum of radii => minimize -sum(r)
    obj = -np.sum(r)
    
    # Penalty
    pen = 0.0
    grad = np.zeros(3*N)
    
    for i in range(N):
        # Boundary penalties
        # Left: x - r < 0
        if x[i] - r[i] < 0:
            v = x[i] - r[i]
            pen += v*v
            grad[3*i + 0] += 2*v  # d/dx
            grad[3*i + 2] += -2*v  # d/dr
        
        # Right: x + r > 1
        if x[i] + r[i] > 1:
            v = x[i] + r[i] - 1
            pen += v*v
            grad[3*i + 0] += 2*v
            grad[3*i + 2] += 2*v
        
        # Bottom: y - r < 0
        if y[i] - r[i] < 0:
            v = y[i] - r[i]
            pen += v*v
            grad[3*i + 1] += 2*v
            grad[3*i + 2] += -2*v
        
        # Top: y + r > 1
        if y[i] + r[i] > 1:
            v = y[i] + r[i] - 1
            pen += v*v
            grad[3*i + 1] += 2*v
            grad[3*i + 2] += 2*v
        
        # Overlap penalties
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            min_dist = r[i] + r[j]
            
            if dist < 1e-10:
                dist = 1e-10
                dx = 1e-10
                dy = 0.0
            
            if dist < min_dist:
                v = min_dist - dist
                pen += v*v
                
                # Gradient w.r.t. x[i]
                gx_i = -2*v * (dx / dist)
                gy_i = -2*v * (dy / dist)
                gr_i = 2*v
                
                # Gradient w.r.t. x[j]
                gx_j = -2*v * (-dx / dist)
                gy_j = -2*v * (-dy / dist)
                gr_j = 2*v
                
                grad[3*i + 0] += gx_i
                grad[3*i + 1] += gy_i
                grad[3*i + 2] += gr_i
                grad[3*j + 0] += gx_j
                grad[3*j + 1] += gy_j
                grad[3*j + 2] += gr_j
    
    obj += 1000.0 * pen  # Weight penalty heavily
    grad += 1000.0 * grad
    
    return obj, grad

def _is_valid(params, N, tol=1e-6):
    """Check if configuration is valid (no overlaps, inside square)."""
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    # Check boundary
    for i in range(N):
        if x[i] - r[i] < -tol or x[i] + r[i] > 1 + tol:
            return False
        if y[i] - r[i] < -tol or y[i] + r[i] > 1 + tol:
            return False
        if r[i] <= 0:
            return False
    
    # Check overlaps
    for i in range(N):
        for j in range(i+1, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < r[i] + r[j] - tol:
                return False
    
    return True

def _sum_radii(params, N):
    """Sum of radii."""
    return np.sum(params[2::3])

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: grid placement
    side = int(np.ceil(np.sqrt(N)))
    r0 = 0.5 / side - 0.01
    x0 = np.array([(i + 0.5) / side for i in range(side) for j in range(side)][:N])
    y0 = np.array([(j + 0.5) / side for i in range(side) for j in range(side)][:N])
    r0_arr = np.full(N, r0)
    
    params0 = np.zeros(3*N)
    params0[0::3] = x0
    params0[1::3] = y0
    params0[2::3] = r0_arr
    
    best_params = params0.copy()
    best_score = _sum_radii(best_params, N) if _is_valid(best_params, N) else -np.inf
    
    # Try multiple restarts with different initial configurations
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate diverse initial guess
        if restart == 0:
            params = params0.copy()
        else:
            # Random jitter around grid
            np.random.seed(42 + restart)
            jitter = 0.05
            params = params0.copy()
            params[0::3] += np.random.uniform(-jitter, jitter, N)
            params[1::3] += np.random.uniform(-jitter, jitter, N)
            params[2::3] = np.clip(params[2::3] * np.random.uniform(0.5, 1.5, N), 0.01, 0.2)
            # Clip positions to valid range
            params[0::3] = np.clip(params[0::3], params[2::3], 1 - params[2::3])
            params[1::3] = np.clip(params[1::3], params[2::3], 1 - params[2::3])
        
        # L-BFGS-B optimization
        try:
            result = minimize(
                _objective_and_grad,
                params,
                args=(N,),
                method='L-BFGS-B',
                jac=True,
                bounds=[(0.01, 0.99)] * N + [(0.01, 0.99)] * N + [(0.005, 0.25)] * N,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.success or result.fun < (best_score * -1 + 1000 * 0 if best_score != -np.inf else np.inf):
                # Check validity
                if _is_valid(result.x, N, tol=1e-5):
                    score = _sum_radii(result.x, N)
                    if score > best_score:
                        best_score = score
                        best_params = result.x.copy()
        except Exception:
            pass
        
        # Try SLSQP polish if we have time
        if time.time() > deadline - 5.0:
            break
        
        if best_score != -np.inf:
            try:
                result2 = minimize(
                    _objective_and_grad,
                    best_params,
                    args=(N,),
                    method='SLSQP',
                    jac=True,
                    bounds=[(0.01, 0.99)] * N + [(0.01, 0.99)] * N + [(0.005, 0.25)] * N,
                    options={'maxiter': 200, 'ftol': 1e-12, 'eps': 1e-8}
                )
                
                if _is_valid(result2.x, N, tol=1e-5):
                    score = _sum_radii(result2.x, N)
                    if score > best_score:
                        best_score = score
                        best_params = result2.x.copy()
            except Exception:
                pass
    
    # Final check: if best_params is not valid, fall back to grid
    if not _is_valid(best_params, N, tol=1e-4):
        # Use the simple grid as fallback
        side = int(np.ceil(np.sqrt(N)))
        r_fallback = 0.5 / side - 0.005
        x_f = np.array([(i + 0.5) / side for i in range(side) for j in range(side)][:N])
        y_f = np.array([(j + 0.5) / side for i in range(side) for j in range(side)][:N])
        r_f = np.full(N, r_fallback)
        result = np.zeros((N, 3))
        result[:, 0] = x_f
        result[:, 1] = y_f
        result[:, 2] = r_f
        return result
    
    # Format result
    result = np.zeros((N, 3))
    result[:, 0] = best_params[0::3]
    result[:, 1] = best_params[1::3]
    result[:, 2] = best_params[2::3]
    
    return result
# EVOLVE-BLOCK-END
