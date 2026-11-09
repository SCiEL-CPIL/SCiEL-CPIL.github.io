import numpy as np
from scipy.optimize import basinhopping

N = 11
H = np.sqrt(3) / 2

def construct():
    def params_to_cartesian(params):
        pts = np.zeros((N, 2))
        for i in range(N):
            s = params[2*i]
            t = params[2*i+1]
            u = s
            v = t * (1.0 - s)
            w = 1.0 - u - v
            
            x = 0.5 + 0.5 * v - 0.5 * u
            y = H * w
            pts[i] = [x, y]
        return pts

    def min_triangle_area(pts):
        min_area = np.inf
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    a = pts[i]
                    b = pts[j]
                    c = pts[k]
                    area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                    if area < min_area:
                        min_area = area
        return min_area

    def objective(params):
        pts = params_to_cartesian(params)
        area = min_triangle_area(pts)
        return -area

    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_val = np.inf
    best_params = None
    
    starts = []
    rng = np.random.default_rng(42)
    for _ in range(5):
        starts.append(rng.uniform(0, 1, 2*N))
        
    for i, x0 in enumerate(starts):
        try:
            res = basinhopping(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                niter=50, 
                stepsize=0.5, 
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'maxiter': 100, 'ftol': 1e-10}},
                seed=42+i
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        best_params = np.zeros(2*N)
        
    best_pts = params_to_cartesian(best_params)
    
    return best_pts
