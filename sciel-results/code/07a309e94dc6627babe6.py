import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time

N = 14

def construct():
    # Pre-compute indices for all triangles
    indices = np.array(list(itertools.combinations(range(N), 3)))
    I_idx = indices[:, 0]
    J_idx = indices[:, 1]
    K_idx = indices[:, 2]
    
    def get_points(x):
        pts = np.zeros((N, 2))
        pts[0] = [0.0, 0.0]
        pts[1] = [1.0, 0.0]
        for i in range(N - 2):
            pts[i + 2, 0] = x[2 * i]
            pts[i + 2, 1] = x[2 * i + 1]
        return pts
    
    def compute_areas(pts):
        Pi = pts[I_idx]
        Pj = pts[J_idx]
        Pk = pts[K_idx]
        cross = (Pj[:, 0] - Pi[:, 0]) * (Pk[:, 1] - Pi[:, 1]) - (Pj[:, 1] - Pi[:, 1]) * (Pk[:, 0] - Pi[:, 0])
        return 0.5 * np.abs(cross)

    def compute_hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.volume
        except Exception:
            return 1e-10 # Small area to penalize degenerate configs

    def raw_objective(x):
        pts = get_points(x)
        areas = compute_areas(pts)
        min_area = np.min(areas)
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
        return -(min_area / hull_area)

    def smoothed_objective(x, beta):
        pts = get_points(x)
        areas = compute_areas(pts)
        
        # Log-sum-exp for smooth min
        # min(a) ≈ -1/beta * log(sum(exp(-beta * a)))
        # To avoid overflow/underflow, shift by max
        max_a = np.max(areas)
        log_sum_exp = max_a + np.log(np.sum(np.exp(-beta * (areas - max_a))))
        smooth_min = -log_sum_exp / beta
        
        hull_area = compute_hull_area(pts)
        if hull_area < 1e-10:
            return 1.0
            
        return -(smooth_min / hull_area)

    # Initial guess: Regular polygon
    def initial_guess():
        x = np.zeros(2 * (N - 2))
        # Place points on a circle centered at (0.5, 0.5) with radius 0.5?
        # Or just a regular polygon including (0,0) and (1,0) is hard to fit exactly.
        # Let's place N-2 points on a circle that passes through (0,0) and (1,0).
        # Center at (0.5, y_c). Radius R.
        # Simple heuristic: place remaining points on a circle centered at (0.5, 0.5) radius 0.5
        # This puts (0,0) and (1,0) on the circle? No, dist from (0.5,0.5) to (0,0) is sqrt(0.5) ~ 0.707.
        # Let's just use a regular N-gon and shift/scale it to match P0, P1.
        
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts_circ = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # We want pts[0] = (0,0) and pts[1] = (1,0).
        # The regular polygon has some symmetry.
        # Let's just initialize the free variables to form a rough convex shape.
        # A good start is a regular polygon.
        # Let's define the first two points of the regular polygon to be (0,0) and (1,0) by rotation/scaling?
        # Easier: Just place points 2..N-1 on a circle that fits nicely.
        # Center (0.5, 0.5), Radius 0.5.
        # Angles for points 2..N-1 should avoid 0 and pi/2 etc if possible, but let's just spread them.
        # Points 0 and 1 are fixed.
        # Let's place the other N-2 points on the upper semicircle of a circle centered at (0.5, 0) with radius 0.5?
        # That would make them collinear with 0 and 1? No.
        
        # Let's use a standard regular N-gon initialization.
        # Generate N points on a circle.
        # Rotate and scale so that min x is 0 and max x is 1?
        # Or just set x[0] = -0.5, y[0]=0 etc.
        
        # Simple robust init:
        # Points on a circle centered at (0.5, 0.5) with radius 0.4
        # This keeps them within [0.1, 0.9] x [0.1, 0.9]
        # But P0=(0,0) and P1=(1,0) are fixed.
        # So the hull will likely include P0, P1 and the arc.
        
        # Let's place points 2..N-1 on an arc from angle pi to 0 (upper half)
        # Center (0.5, 0), Radius 0.5?
        # If center is (0.5, 0), P0=(0,0) and P1=(1,0) are on the circle.
        # So let's place remaining points on the upper semi-circle.
        angles = np.linspace(np.pi, 0, N-2) # From left to right along top
        # Add a bit of randomness or specific spacing
        x_init = np.zeros(2 * (N - 2))
        for i in range(N - 2):
            theta = np.linspace(np.pi, 0, N-2)[i]
            # Circle centered at (0.5, 0) radius 0.5
            x_init[2*i] = 0.5 + 0.5 * np.cos(theta)
            x_init[2*i+1] = 0.5 * np.sin(theta)
            
        return x_init

    best_x = None
    best_score = -np.inf
    start_time = time.time()
    time_limit = 110.0

    # Annealing schedule for beta
    betas = [10, 50, 100, 200, 500, 1000]
    
    # Multiple restarts
    num_restarts = 5
    
    for r in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb initial guess for restarts > 0
        if r == 0:
            x0 = initial_guess()
        else:
            x0 = initial_guess() + np.random.randn(len(initial_guess())) * 0.1
            # Ensure bounds? L-BFGS-B handles bounds.
            
        x_curr = x0
        
        for beta in betas:
            if time.time() - start_time > time_limit:
                break
                
            try:
                res = minimize(
                    smoothed_objective,
                    x_curr,
                    args=(beta,),
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * (N - 2), # Generous bounds
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if res.fun < -best_score:
                    best_score = -res.fun
                    best_x = res.x
                    x_curr = res.x # Warm start for next beta
            except Exception:
                pass
        
        # Final local refinement with raw objective
        if time.time() - start_time < time_limit:
            try:
                res_raw = minimize(
                    raw_objective,
                    x_curr,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res_raw.fun < -best_score:
                    best_score = -res_raw.fun
                    best_x = res_raw.x
            except Exception:
                pass

    if best_x is None:
        return get_points(initial_guess())
        
    return get_points(best_x)

# EVOLVE-BLOCK-END
