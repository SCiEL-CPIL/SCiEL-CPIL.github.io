import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2

# Precompute indices for combinations
TRIPLETS = list(itertools.combinations(range(N), 3))
IDX_I = np.array([t[0] for t in TRIPLETS])
IDX_J = np.array([t[1] for t in TRIPLETS])
IDX_K = np.array([t[2] for t in TRIPLETS])

def construct():
    start_time = time.time()
    timeout = 140.0 # Safety margin for 150s limit

    # Barycentric to Cartesian
    # V0=(0,0), V1=(1,0), V2=(0.5, H)
    # P = u*V0 + v*V1 + w*V2, w = 1-u-v
    # x = v + 0.5(1-u-v) = 0.5 + 0.5v - 0.5u
    # y = H(1-u-v)
    def bary_to_cart(u, v):
        x = 0.5 + 0.5 * v - 0.5 * u
        y = H * (1.0 - u - v)
        return x, y

    def compute_min_area(u, v):
        """
        Computes the minimum area of triangles formed by points defined by barycentric coords u, v.
        Returns 0 if any point is invalid (u<0, v<0, u+v>1).
        """
        # Check validity
        if np.any(u < -1e-9) or np.any(v < -1e-9) or np.any(u + v > 1.0 + 1e-9):
            return 0.0
        
        # Clip small numerical errors
        u = np.clip(u, 0.0, 1.0)
        v = np.clip(v, 0.0, 1.0)
        # Ensure u+v <= 1 by clipping v if necessary? 
        # Actually, if u+v > 1, it's invalid. We return 0.
        # For valid points, compute areas.
        
        x, y = bary_to_cart(u, v)
        
        xi = x[IDX_I]; xj = x[IDX_J]; xk = x[IDX_K]
        yi = y[IDX_I]; yj = y[IDX_J]; yk = y[IDX_K]
        
        # Area = 0.5 * | xi(yj - yk) + xj(yk - yi) + xk(yi - yj) |
        areas = 0.5 * np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        
        return np.min(areas)

    def soft_min_objective(params, beta):
        """
        Objective for L-BFGS-B. Minimizes -soft_min(areas) + penalty.
        """
        u = params[:N]
        v = params[N:]
        
        # Penalty for invalid points
        viol = np.maximum(0.0, u + v - 1.0)
        viol_u = np.maximum(0.0, -u)
        viol_v = np.maximum(0.0, -v)
        penalty = 100.0 * (np.sum(viol**2) + np.sum(viol_u**2) + np.sum(viol_v**2))
        
        if penalty > 1e-6:
            return 1.0 + penalty # Large positive value to discourage
            
        x, y = bary_to_cart(u, v)
        
        xi = x[IDX_I]; xj = x[IDX_J]; xk = x[IDX_K]
        yi = y[IDX_I]; yj = y[IDX_J]; yk = y[IDX_K]
        
        areas = 0.5 * np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        
        # Soft min: -1/beta * log(sum(exp(-beta * areas)))
        # We want to maximize min area, so we minimize -soft_min
        # -soft_min = 1/beta * log(sum(exp(-beta * areas)))
        
        # Numerical stability
        min_area = np.min(areas)
        if min_area < 1e-10:
             return 1.0 + penalty
             
        exp_term = np.exp(-beta * (areas - min_area))
        log_sum = np.log(np.sum(exp_term))
        soft_min_val = min_area - 1.0 / beta * log_sum
        
        obj = -soft_min_val + penalty
        return obj

    # 1. Global Optimization with Differential Evolution
    # Bounds: u in [0, 1], v in [0, 1]
    # We rely on the penalty in compute_min_area to reject invalid points
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # DE minimizes, so we negate the min area
    def de_objective(params):
        u = params[:N]
        v = params[N:]
        min_a = compute_min_area(u, v)
        return -min_a

    best_de_result = None
    try:
        # popsize=20, maxiter=100 is a reasonable start.
        # We monitor time. DE doesn't have a timeout, so we must be conservative.
        # Let's estimate: 1 eval ~ 0.1ms. 20 pop * 100 iter = 2000 evals ~ 0.2s.
        # We can afford more. Let's try maxiter=300.
        res_de = differential_evolution(
            de_objective, 
            bounds, 
            seed=42, 
            maxiter=300, 
            popsize=20, 
            tol=1e-8,
            init='sobol'
        )
        if time.time() - start_time < timeout - 20:
            best_de_result = res_de.x
    except Exception as e:
        pass

    # 2. Local Polish with L-BFGS-B
    if best_de_result is None:
        # Fallback to random valid start
        while True:
            u_init = np.random.rand(N)
            v_init = np.random.rand(N)
            if np.all(u_init + v_init <= 1.0):
                break
        best_de_result = np.concatenate([u_init, v_init])

    p0 = best_de_result
    bounds_lbfgs = [(0.0, 1.0)] * (2 * N)
    
    best_params = p0.copy()
    best_val = -compute_min_area(best_params[:N], best_params[N:]) # Negative of min area
    
    # Try a few restarts from the DE result with different betas
    betas = [50, 100, 200]
    
    for beta in betas:
        if time.time() - start_time > timeout - 10:
            break
            
        # Small perturbation to escape potential flat spots
        p_start = best_params + np.random.randn(2*N) * 0.01
        p_start = np.clip(p_start, 0.0, 1.0)
        
        try:
            res_lbfgs = minimize(
                soft_min_objective, 
                p_start, 
                args=(beta,), 
                method='L-BFGS-B', 
                bounds=bounds_lbfgs, 
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            # Check if this result is better
            u_opt = res_lbfgs.x[:N]
            v_opt = res_lbfgs.x[N:]
            
            # Validate
            if np.all(u_opt >= -1e-9) and np.all(v_opt >= -1e-9) and np.all(u_opt + v_opt <= 1.0 + 1e-9):
                u_opt = np.clip(u_opt, 0.0, 1.0)
                v_opt = np.clip(v_opt, 0.0, 1.0)
                min_a = compute_min_area(u_opt, v_opt)
                if -min_a < best_val:
                    best_val = -min_a
                    best_params = res_lbfgs.x.copy()
        except Exception:
            pass

    # Finalize
    u_final = best_params[:N]
    v_final = best_params[N:]
    
    # Final validity check and clip
    u_final = np.clip(u_final, 0.0, 1.0)
    v_final = np.clip(v_final, 0.0, 1.0)
    
    # If u+v > 1, scale down to boundary? 
    # Better to just return the best valid point found.
    # The optimization should have kept them valid.
    
    x, y = bary_to_cart(u_final, v_final)
    points = np.column_stack([x, y])
    
    return points
