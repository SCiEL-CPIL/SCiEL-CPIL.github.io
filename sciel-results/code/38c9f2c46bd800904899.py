import numpy as np
from scipy.optimize import differential_evolution
import time

N = 11

def construct():
    start_time = time.time()
    deadline = 110.0
    
    H = np.sqrt(3) / 2
    
    bounds = [
        (0, H), (0, H), (0, H),
        (0, 0.5), (0, 0.5), (0, 0.5), (0, 0.5),
        (0, 1), (0, 1), (0, 1), (0, 1)
    ]
    
    def objective(params):
        a1, a2, a3, b1, b2, b3, b4, u1, u2, u3, u4 = params
        
        points = np.empty((11, 2))
        points[0] = [0.5, a1]
        points[1] = [0.5, a2]
        points[2] = [0.5, a3]
        
        idx = 3
        for i in range(4):
            b = params[3 + i]
            u = params[7 + i]
            c = np.sqrt(3) * b * u
            points[idx] = [b, c]
            points[idx + 1] = [1 - b, c]
            idx += 2
            
        min_area = float('inf')
        for i in range(11):
            pi = points[i]
            for j in range(i + 1, 11):
                pj = points[j]
                v1 = pj - pi
                for k in range(j + 1, 11):
                    pk = points[k]
                    v2 = pk - pi
                    area = 0.5 * abs(v1[0] * v2[1] - v1[1] * v2[0])
                    if area < min_area:
                        min_area = area
                        
        return -min_area

    try:
        result = differential_evolution(
            objective, 
            bounds, 
            seed=42, 
            maxiter=200, 
            popsize=20,
            tol=1e-8,
            init='sobol',
            polish=True
        )
        
        if time.time() < deadline:
            best_params = result.x
            a1, a2, a3, b1, b2, b3, b4, u1, u2, u3, u4 = best_params
            points = np.empty((11, 2))
            points[0] = [0.5, a1]
            points[1] = [0.5, a2]
            points[2] = [0.5, a3]
            
            idx = 3
            for i in range(4):
                b = best_params[3 + i]
                u = best_params[7 + i]
                c = np.sqrt(3) * b * u
                points[idx] = [b, c]
                points[idx + 1] = [1 - b, c]
                idx += 2
            return points
            
    except Exception:
        pass

    # Fallback: Random points
    rng = np.random.RandomState(7)
    points = []
    while len(points) < N:
        u, v = rng.random(), rng.random()
        if u + v < 0.98 and u > 0.01 and v > 0.01:
            points.append([u + 0.5 * v, (3 ** 0.5) / 2 * v])
    return np.array(points)

# EVOLVE-BLOCK-END
