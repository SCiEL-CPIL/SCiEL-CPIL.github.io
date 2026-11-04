import numpy as np
from scipy.optimize import differential_evolution

N = 11
S3 = np.sqrt(3)
H = S3 / 2

def construct():
    bounds = [(0, 1), (0, 1)] * N  # u, v for each point

    def objective(params):
        uv = params.reshape((N, 2))
        u = uv[:, 0]
        v = uv[:, 1]
        
        # Check u+v <= 1
        sum_uv = u + v
        if np.any(sum_uv > 1 + 1e-9):
            return 1e6

        # Convert to Cartesian
        xs = u + 0.5 * v
        ys = v * H
        
        points = np.column_stack((xs, ys))
        
        min_area = np.inf
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    ax, ay = points[i]
                    bx, by = points[j]
                    cx, cy = points[k]
                    area = 0.5 * abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
                    if area < min_area:
                        min_area = area
        return -min_area

    res = differential_evolution(
        objective, 
        bounds, 
        seed=42, 
        maxiter=300, 
        popsize=25, 
        tol=1e-7,
        polish=True
    )
    
    best_params = res.x
    uv = best_params.reshape((N, 2))
    u = uv[:, 0]
    v = uv[:, 1]
    xs = u + 0.5 * v
    ys = v * H
    
    return np.column_stack((xs, ys))
