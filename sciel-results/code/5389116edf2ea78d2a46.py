"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin for safety

    sqrt3 = np.sqrt(3)
    h = sqrt3 / 2.0
    
    # Bounds for x and y for each point
    # x in [0, 1], y in [0, sqrt(3)/2]
    bounds = [(0, 1), (0, h)] * N

    def penalty(points):
        # points is shape (N, 2)
        xs = points[:, 0]
        ys = points[:, 1]
        
        # Check triangle constraints
        # 1. y >= 0 (handled by bounds)
        # 2. y <= sqrt(3) * x
        if np.any(ys > sqrt3 * xs + 1e-9):
            return 1e6
        # 3. y <= sqrt(3) * (1 - x)
        if np.any(ys > sqrt3 * (1 - xs) + 1e-9):
            return 1e6
            
        return 0.0

    def objective(params):
        points = params.reshape(N, 2)
        
        # Check constraints
        p_val = penalty(points)
        if p_val > 0:
            return p_val

        # Calculate all triangle areas
        min_area = float('inf')
        
        for i in range(N-2):
            for j in range(i+1, N-1):
                for k in range(j+1, N):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    
                    area = 0.5 * abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))
                    if area < min_area:
                        min_area = area
                        
        return -min_area

    best_params = None
    best_val = float('inf')

    # Strategy: Run Differential Evolution
    try:
        # Use a seed for reproducibility in testing, but random is fine for search
        res = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=100, 
            popsize=15,
            tol=1e-5,
            polish=False 
        )
        best_params = res.x
        best_val = res.fun
    except Exception as e:
        # Fallback to random if DE fails or times out (though it shouldn't)
        points = np.random.rand(N, 2)
        points[:, 0] *= 1.0
        points[:, 1] *= h
        # Simple projection to ensure inside triangle
        for i in range(N):
            x, y = points[i]
            if y > sqrt3 * x:
                y = sqrt3 * x
            if y > sqrt3 * (1 - x):
                y = sqrt3 * (1 - x)
            points[i] = [x, y]
        return points

    # Refine with Nelder-Mead if time permits
    if time.time() < deadline - 5.0:
        try:
            res_local = minimize(
                objective, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-12}
            )
            if res_local.fun < best_val:
                best_params = res_local.x
                best_val = res_local.fun
        except Exception:
            pass

    final_points = best_params.reshape(N, 2)
    return final_points

# EVOLVE-BLOCK-END
