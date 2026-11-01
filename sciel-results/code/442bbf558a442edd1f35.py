import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin

    def get_score(points):
        """
        points: array of shape (N, 2)
        Returns: min_triangle_area / convex_hull_area
        """
        # Calculate areas of all triangles
        # Using the determinant formula: 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
        # Vectorized calculation:
        # i, j, k are indices
        # We can loop over combinations or use broadcasting.
        # N=14, C(14,3)=364. Looping is fine and clear.
        min_area = float('inf')
        
        # Precompute convex hull area
        try:
            hull = ConvexHull(points)
            hull_area = hull.volume # For 2D, volume is area
            if hull_area <= 1e-10:
                return 0.0
        except Exception:
            return 0.0

        # Calculate min triangle area
        # To speed up, we can use a simple loop
        for i in range(N):
            pi = points[i]
            for j in range(i + 1, N):
                pj = points[j]
                for k in range(j + 1, N):
                    pk = points[k]
                    # Area = 0.5 * |(pj - pi) x (pk - pi)|
                    v1 = pj - pi
                    v2 = pk - pi
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    area = 0.5 * abs(cross)
                    if area < min_area:
                        min_area = area
                        
        return min_area / hull_area

    def objective(x):
        """
        x: flat array of length 2*N
        Returns: -score
        """
        points = x.reshape((N, 2))
        s = get_score(points)
        return -s

    # Initialize with regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    init_points = np.column_stack((np.cos(angles), np.sin(angles)))
    init_x = init_points.flatten()

    best_x = init_x
    best_score = get_score(init_points)

    # Try local optimization
    # Nelder-Mead is robust for non-smooth functions
    try:
        # Set options to respect time limit roughly
        # maxiter is a proxy, but we check time inside if we could.
        # We'll just run it and catch if it takes too long? No, minimize doesn't support timeout directly.
        # We can run a limited number of iterations.
        res = minimize(objective, init_x, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.success or res.fun < -best_score:
            best_x = res.x
            best_score = -res.fun
            
    except Exception as e:
        pass

    # If we have time, try a second phase or different start?
    # Let's check time
    if time.time() - start_time < time_limit:
        # Try Powell method which might find a slightly better local opt
        try:
            res2 = minimize(objective, best_x, method='Powell', options={'maxiter': 500, 'xtol': 1e-6, 'ftol': 1e-6})
            if res2.fun < -best_score:
                best_x = res2.x
                best_score = -res2.fun
        except Exception:
            pass

    final_points = best_x.reshape((N, 2))
    return final_points

# EVOLVE-BLOCK-END
