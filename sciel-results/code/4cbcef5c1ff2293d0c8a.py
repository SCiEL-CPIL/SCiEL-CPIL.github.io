import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14

def construct():
    start_time = time.time()
    time_limit = 120.0  # Safety margin against 150s kill
    
    # Parameterization: 7 vectors v_i, points are v_i and -v_i
    # Total 14 points.
    k = N // 2
    
    # Initial guess: Regular 14-gon
    # Angles for 7 vectors, spaced by pi/7
    angles = np.linspace(0, np.pi, k, endpoint=False)
    v_init = np.column_stack([np.cos(angles), np.sin(angles)])
    x0 = v_init.flatten()
    
    def get_points(params):
        v = params.reshape(k, 2)
        # Create full set of points
        P = np.vstack([v, -v])
        return P
    
    def score(points):
        # Compute convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return -1e10 # Penalize degenerate hulls
            
        if hull_area < 1e-12:
            return -1e10
            
        # Compute min triangle area
        n = points.shape[0]
        min_area = float('inf')
        
        # Loop over all triples
        # Optimization: Vectorize the cross product calculation for all pairs first?
        # With N=14, 364 triangles is small. A loop is acceptable and clear.
        # We can speed it up by pre-computing differences.
        
        for i in range(n):
            for j in range(i+1, n):
                # Vector from i to j
                dx1 = points[j, 0] - points[i, 0]
                dy1 = points[j, 1] - points[i, 1]
                for k_idx in range(j+1, n):
                    # Vector from i to k
                    dx2 = points[k_idx, 0] - points[i, 0]
                    dy2 = points[k_idx, 1] - points[i, 1]
                    area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                    if area < min_area:
                        min_area = area
                        
        return min_area / hull_area

    def objective(params):
        P = get_points(params)
        s = score(P)
        # We want to maximize s, so minimize -s
        return -s

    # Run optimization
    best_params = x0
    best_score = -objective(x0)
    
    # Powell method is robust for non-smooth functions
    # We limit the number of function evaluations to prevent infinite loops if it gets stuck
    # But we also have a time guard.
    
    def time_guard():
        return time.time() - start_time > time_limit

    try:
        # Use Powell with a reasonable maxiter
        # We check time inside the callback if possible, but scipy minimize doesn't easily support interrupting mid-iteration.
        # So we rely on the fact that each eval is fast and maxiter bounds the total time.
        # Let's estimate: 1 eval ~ 1ms. 10000 evals ~ 10s.
        # Powell might need more. Let's try Nelder-Mead as well or just run Powell with a high maxiter and hope it converges or times out.
        
        # Actually, let's use a wrapper that checks time and raises an exception if time is up.
        # But minimize catches exceptions? No, it propagates them.
        
        # Better approach: Run minimize in a loop of smaller chunks? Or just set a generous maxiter and rely on the outer time check after.
        # If minimize takes 140s, we are dead.
        # Let's set maxiter based on time.
        
        # Let's try Nelder-Mead first, it's often faster to converge on these shapes.
        # Or use a simple local search?
        
        # Let's stick to Powell but monitor.
        # To be safe, we can run minimize with a timeout-like mechanism by checking time after each function evaluation?
        # SciPy's minimize does not support a callback that can abort the optimization.
        
        # Alternative: Use a manual loop for a simple gradient-free method?
        # Or just trust that Powell on 14 dims from a good start converges fast.
        
        res = minimize(objective, x0, method='Powell', options={'maxiter': 5000, 'xtol': 1e-8, 'ftol': 1e-8})
        
        if res.fun < -best_score:
            best_params = res.x
            best_score = -res.fun
            
    except Exception as e:
        pass

    # If we have time left, try another method or polish
    if time.time() - start_time < time_limit - 10:
        try:
            # Try Nelder-Mead from the best point found
            res2 = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            if res2.fun < -best_score:
                best_params = res2.x
                best_score = -res2.fun
        except Exception:
            pass

    # Final Polish: High precision local search
    if time.time() - start_time < time_limit - 5:
        try:
            # Use COBYLA or BFGS with numerical gradients?
            # BFGS might struggle with non-smooth min.
            # Let's just do another Powell run with tighter tolerance.
            res3 = minimize(objective, best_params, method='Powell', options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-10})
            if res3.fun < -best_score:
                best_params = res3.x
                best_score = -res3.fun
        except Exception:
            pass

    final_points = get_points(best_params)
    return final_points
