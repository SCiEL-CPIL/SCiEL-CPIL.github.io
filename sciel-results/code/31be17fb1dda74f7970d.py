import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_S = 40.0
    best_params = None

    def check_packing_vectorized(params, S):
        # params: (N, 3)
        base_angles = np.deg2rad(np.arange(0, 360, 60))
        angles = np.deg2rad(params[:, 2][:, np.newaxis]) + base_angles[np.newaxis, :]
        
        vx = np.cos(angles)
        vy = np.sin(angles)
        
        px = params[:, 0][:, np.newaxis] + vx
        py = params[:, 1][:, np.newaxis] + vy
        
        px = px.flatten()
        py = py.flatten()
        
        phis = np.deg2rad(30 + np.arange(0, 360, 60))
        c = np.cos(phis)
        s = np.sin(phis)
        
        vals = np.outer(px, c) + np.outer(py, s)
        
        max_vals = np.max(vals, axis=1)
        
        limit = S * np.sqrt(3) / 2.0
        violations = np.maximum(0, max_vals - limit)
        
        return np.max(violations) if violations.size > 0 else 0.0

    def try_solve(S, seed):
        bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 180.0)] * N
        
        def obj(p):
            return check_packing_vectorized(np.array(p).reshape(N, 3), S)
            
        # Use a heuristic initial guess for DE? DE ignores x0 mostly, but we can use it for polish.
        # Let's just run DE.
        result = differential_evolution(
            obj,
            bounds,
            seed=seed,
            maxiter=30,
            popsize=10,
            tol=1e-5,
            polish=True
        )
        
        if result.fun < 1e-6:
            return True, result.x
        return False, None

    # Try a few S values
    # We want to find the smallest S.
    # Start from 3.5 up to 4.5
    # If we find one, we can try to go lower with binary search?
    # Or just pick the best found.
    
    candidates = [3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2]
    
    for S in candidates:
        if time.time() > deadline - 20:
            break
            
        found, params = try_solve(S, seed=int(S*100))
        if found:
            best_S = S
            best_params = params
            # We found a valid packing for S.
            # Can we go lower?
            # Let's try to binary search between 3.46 and S
            low = 3.46
            high = S
            for _ in range(5):
                mid = (low + high) / 2.0
                if time.time() > deadline - 10:
                    break
                found_mid, params_mid = try_solve(mid, seed=42)
                if found_mid:
                    high = mid
                    best_S = mid
                    best_params = params_mid
                else:
                    low = mid
            
            break
            
    if best_params is None:
        # Fallback
        best_S = 40.0
        best_params = np.array([[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)])
    else:
        # Polish
        if time.time() < deadline - 5:
            def obj_polish(p):
                return check_packing_vectorized(np.array(p).reshape(N, 3), best_S)
            
            res = minimize(
                obj_polish,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.fun < 1e-6:
                best_params = res.x

    inner = []
    for i in range(N):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
        
    return inner, (0.0, 0.0), best_S, 0.0
