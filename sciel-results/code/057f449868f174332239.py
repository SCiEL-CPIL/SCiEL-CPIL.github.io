import numpy as np
from scipy.optimize import minimize

N = 13

def construct():
    def score(theta):
        theta_sorted = np.sort(theta)
        x = np.cos(theta_sorted)
        y = np.sin(theta_sorted)
        pts = np.column_stack((x, y))
        
        x0 = np.roll(x, -1)
        y0 = np.roll(y, -1)
        area_hull = 0.5 * np.abs(np.sum(x * y0 - x0 * y))
        
        if area_hull == 0:
            return 1.0
            
        min_area = np.inf
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    ax = x[j] - x[i]
                    ay = y[j] - y[i]
                    bx = x[k] - x[i]
                    by = y[k] - y[i]
                    area = 0.5 * np.abs(ax * by - ay * bx)
                    if area < min_area:
                        min_area = area
        
        return -min_area / area_hull
    
    # Try multiple initializations to avoid local minima
    best_score = np.inf
    best_theta = None
    
    np.random.seed(42)
    
    # Start with regular polygon
    theta0 = np.linspace(0, 2*np.pi, N, endpoint=False)
    res = minimize(score, theta0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
    if res.fun < best_score:
        best_score = res.fun
        best_theta = res.x
    
    # Try random perturbations of regular polygon
    for trial in range(5):
        theta_rand = theta0 + np.random.randn(N) * 0.1
        res = minimize(score, theta_rand, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_theta = res.x
    
    theta_opt = best_theta
    theta_sorted = np.sort(theta_opt)
    x = np.cos(theta_sorted)
    y = np.sin(theta_sorted)
    pts = np.column_stack((x, y))
    
    return pts
