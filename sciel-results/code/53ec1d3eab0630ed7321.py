import numpy as np
import time
import scipy.optimize as opt

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # 5s margin for safety
    
    # Precompute indices for all triples
    triples = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
    idx_i, idx_j, idx_k = triples[:, 0], triples[:, 1], triples[:, 2]
    
    def get_score(theta):
        x = np.cos(theta)
        y = np.sin(theta)
        
        # Hull Area
        # Sort by angle to ensure correct shoelace order
        angles = np.arctan2(y, x)
        sorted_idx = np.argsort(angles)
        x_s = x[sorted_idx]
        y_s = y[sorted_idx]
        
        x_next = np.roll(x_s, -1)
        y_next = np.roll(y_s, -1)
        area_hull = 0.5 * np.abs(np.sum(x_s * y_next - x_next * y_s))
        
        if area_hull < 1e-9:
            return 0.0
            
        # Triangle Areas (Vectorized)
        v1x = x[idx_j] - x[idx_i]
        v1y = y[idx_j] - y[idx_i]
        v2x = x[idx_k] - x[idx_i]
        v2y = y[idx_k] - y[idx_i]
        
        cross = v1x * v2y - v1y * v2x
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        
        return min_area / area_hull

    def objective(theta):
        return -get_score(theta)

    best_score = -np.inf
    best_theta = None
    
    # Initial guesses
    inits = []
    # 1. Regular polygon
    inits.append(np.linspace(0, 2*np.pi, N, endpoint=False))
    
    # 2. Perturbed regular polygons
    rng = np.random.default_rng(42)
    for _ in range(6):
        # Small noise to break symmetry
        noise = rng.normal(0, 0.05, N)
        inits.append(np.linspace(0, 2*np.pi, N, endpoint=False) + noise)
        
    # 3. Random points on circle (sorted)
    for _ in range(3):
        random_angles = np.sort(rng.uniform(0, 2*np.pi, N))
        inits.append(random_angles)

    # Optimization Loop
    for i, theta0 in enumerate(inits):
        if time.time() - start_time > time_limit - 15:
            break
            
        try:
            res = opt.minimize(objective, theta0, method='Nelder-Mead', 
                               options={'maxiter': 1500, 'adaptive': True, 'xatol': 1e-8, 'fatol': 1e-8})
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_theta = res.x.copy()
                
        except Exception:
            continue

    # Final Polish
    if best_theta is not None and time.time() - start_time < time_limit - 5:
        try:
            res2 = opt.minimize(objective, best_theta, method='Nelder-Mead', 
                                options={'maxiter': 3000, 'adaptive': True, 'xatol': 1e-12, 'fatol': 1e-12})
            score2 = -res2.fun
            if score2 > best_score:
                best_score = score2
                best_theta = res2.x.copy()
        except Exception:
            pass

    if best_theta is None:
        # Fallback to regular polygon
        best_theta = np.linspace(0, 2*np.pi, N, endpoint=False)

    points = np.column_stack((np.cos(best_theta), np.sin(best_theta)))
    return points
