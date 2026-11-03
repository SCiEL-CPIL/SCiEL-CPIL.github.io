import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 14
k = N // 2  # 7

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin against 150s kill

    def get_points(params):
        v = params.reshape(k, 2)
        P = np.vstack([v, -v])
        return P

    def score(points):
        # Compute convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume
        except Exception:
            return -1e10

        if hull_area < 1e-12:
            return -1e10

        # Compute min triangle area
        n = points.shape[0]
        min_area = float('inf')

        # Vectorized approach for speed
        # For each pair (i, j), compute cross product with all k > j
        # But for n=14, even O(n^3) is fast enough (364 triangles)
        # Let's stick to a simple loop but optimized
        
        for i in range(n):
            xi, yi = points[i, 0], points[i, 1]
            for j in range(i + 1, n):
                dx1 = points[j, 0] - xi
                dy1 = points[j, 1] - yi
                for l in range(j + 1, n):
                    dx2 = points[l, 0] - xi
                    dy2 = points[l, 1] - yi
                    area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                    if area < min_area:
                        min_area = area
                        if min_area < 1e-15:
                            return -1e10

        return min_area / hull_area

    def objective(params):
        P = get_points(params)
        s = score(P)
        return -s

    def objective_soft(params, beta=100.0):
        P = get_points(params)
        try:
            hull = ConvexHull(P)
            hull_area = hull.volume
        except Exception:
            return 1e10

        if hull_area < 1e-12:
            return 1e10

        n = P.shape[0]
        areas = []
        for i in range(n):
            xi, yi = P[i, 0], P[i, 1]
            for j in range(i + 1, n):
                dx1 = P[j, 0] - xi
                dy1 = P[j, 1] - yi
                for l in range(j + 1, n):
                    dx2 = P[l, 0] - xi
                    dy2 = P[l, 1] - yi
                    area = 0.5 * abs(dx1 * dy2 - dy1 * dx2)
                    areas.append(area)
        
        areas = np.array(areas)
        # Soft min: -log(sum(exp(-beta * area))) / beta
        # To avoid overflow, subtract max
        max_a = np.max(areas)
        soft_min = -np.log(np.sum(np.exp(-beta * (areas - max_a)))) / beta + max_a
        
        return -soft_min / hull_area

    # Initial guess: Regular 14-gon
    angles = np.linspace(0, np.pi, k, endpoint=False)
    v_init = np.column_stack([np.cos(angles), np.sin(angles)])
    x0 = v_init.flatten()
    
    best_params = x0.copy()
    best_score = -objective(x0)

    # Phase 1: Soft objective optimization (smooth)
    # This helps navigate the landscape before the hard min becomes an issue
    if time.time() - start_time < time_limit - 20:
        try:
            # Try a few different betas
            for beta in [10.0, 50.0, 100.0]:
                if time.time() - start_time > time_limit - 10:
                    break
                res = minimize(objective_soft, best_params, method='Powell', 
                               args=(beta,),
                               options={'maxiter': 2000, 'xtol': 1e-8, 'ftol': 1e-8})
                if res.fun < -best_score:
                    best_params = res.x
                    best_score = -res.fun
        except Exception:
            pass

    # Phase 2: Hard objective optimization (accurate)
    # Use Powell on the actual objective
    if time.time() - start_time < time_limit - 10:
        try:
            res = minimize(objective, best_params, method='Powell', 
                           options={'maxiter': 3000, 'xtol': 1e-10, 'ftol': 1e-10})
            if res.fun < -best_score:
                best_params = res.x
                best_score = -res.fun
        except Exception:
            pass

    # Phase 3: Multi-start local refinement
    # Generate a few perturbations of the best solution and refine
    if time.time() - start_time < time_limit - 5:
        # Create a few random perturbations
        rng = np.random.RandomState(42)
        for _ in range(5):
            if time.time() - start_time > time_limit - 5:
                break
            # Perturb the best params
            perturbation = rng.normal(0, 0.1, size=best_params.shape)
            x_pert = best_params + perturbation
            
            try:
                res = minimize(objective, x_pert, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < -best_score:
                    best_params = res.x
                    best_score = -res.fun
            except Exception:
                pass

    # Phase 4: Final high-precision polish
    if time.time() - start_time < time_limit - 2:
        try:
            res = minimize(objective, best_params, method='Powell', 
                           options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-12})
            if res.fun < -best_score:
                best_params = res.x
                best_score = -res.fun
        except Exception:
            pass

    final_points = get_points(best_params)
    return final_points
