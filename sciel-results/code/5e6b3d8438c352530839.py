import numpy as np
from scipy.optimize import minimize
from itertools import combinations

N = 14

def construct():
    def score(params):
        d_theta = params[:N]
        d_r = params[N:]
        
        base_theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        theta = base_theta + d_theta
        r = 1.0 + d_r
        
        if np.any(r <= 0):
            return -1e9
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Convexity check
        px = np.roll(x, -1) - x
        py = np.roll(y, -1) - y
        cross_z = px * np.roll(py, -1) - py * np.roll(px, -1)
        
        if np.any(cross_z <= 1e-6):
            return -1e9
        
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        area_hull = 0.5 * np.abs(np.sum(x * y_next - x_next * y))
        
        if area_hull < 1e-6:
            return -1e9
            
        min_area = np.inf
        pts = np.column_stack((x, y))
        for i, j, k in combinations(range(N), 3):
            xi, yi = pts[i]
            xj, yj = pts[j]
            xk, yk = pts[k]
            area = 0.5 * abs((xj - xi) * (yk - yi) - (xk - xi) * (yj - yi))
            if area < min_area:
                min_area = area
        
        if min_area == np.inf:
            return -1e9
            
        return min_area / area_hull

    def objective(params):
        return -score(params)

    # Initial guess: Regular polygon
    x0 = np.zeros(2 * N)
    
    best_x = x0
    best_score = score(x0)
    
    # Try a few random perturbations to escape potential local minima
    # and the regular polygon itself
    rng = np.random.RandomState(42)
    
    candidates = [x0]
    for _ in range(5):
        pert = rng.uniform(-0.1, 0.1, 2 * N)
        candidates.append(pert)
        
    for start in candidates:
        res = minimize(objective, start, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < -best_score: # Since we minimize -score
            best_score = -res.fun
            best_x = res.x
            
    # If all failed, return regular polygon
    if best_score < 0.1:
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        r = np.ones(N)
        return np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        
    d_theta = best_x[:N]
    d_r = best_x[N:]
    base_theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    theta = base_theta + d_theta
    r = 1.0 + d_r
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return np.column_stack((x, y))
