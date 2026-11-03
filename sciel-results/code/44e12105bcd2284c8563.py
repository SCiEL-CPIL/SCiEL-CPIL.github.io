"""Heilbronn triangles: place 11 points to maximize the smallest triangle area."""
N = 11
import numpy as np
from scipy.optimize import minimize, basinhopping
import time

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

def point_in_triangle(p, a=V0, b=V1, c=V2, eps=1e-9):
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1 = sign(p, a, b)
    d2 = sign(p, b, c)
    d3 = sign(p, c, a)
    has_neg = (d1 < -eps) or (d2 < -eps) or (d3 < -eps)
    has_pos = (d1 > eps) or (d2 > eps) or (d3 > eps)
    return not (has_neg and has_pos)

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by the points.
    points: array of shape (N, 2)
    """
    n = len(points)
    min_area = float('inf')
    # Vectorized calculation for all triplets is possible but loops are fine for N=11
    # 11 choose 3 = 165 triangles
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Area = 0.5 * |x1(y2 - y3) + x2(y3 - y1) + x3(y1 - y2)|
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
                if area < min_area:
                    min_area = area
    return min_area

def objective(x):
    """
    Objective function to minimize.
    We want to maximize min_triangle_area, so we minimize -min_triangle_area.
    We add a penalty for points outside the triangle to guide the optimizer.
    """
    points = x.reshape(-1, 2)
    
    # Penalty for points outside the triangle
    # Triangle boundaries:
    # 1. y >= 0
    # 2. y <= sqrt(3) * x
    # 3. y <= -sqrt(3) * x + sqrt(3)
    
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Distances to boundaries (positive if outside)
    d1 = np.maximum(0.0, -y_coords)
    d2 = np.maximum(0.0, y_coords - np.sqrt(3) * x_coords)
    d3 = np.maximum(0.0, y_coords + np.sqrt(3) * x_coords - np.sqrt(3))
    
    # Quadratic penalty
    penalty = 100.0 * np.sum(d1**2 + d2**2 + d3**2)
    
    # Calculate min area
    try:
        min_area = min_triangle_area(points)
    except Exception:
        return 1e10
    
    return -min_area + penalty

def local_min(x):
    """
    Local minimizer using Nelder-Mead.
    """
    result = minimize(
        objective,
        x,
        method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
    )
    return {
        'x': result.x,
        'fval': result.fun,
        'nit': result.nit,
        'nfev': result.nfev,
        'success': result.success
    }

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for the 150s limit
    
    rng = np.random.RandomState(42)
    
    best_points = None
    best_score = -float('inf')
    
    # Strategy: Use Basin Hopping which combines local optimization (Nelder-Mead)
    # with global exploration (random jumps). This is structurally different from
    # simple multi-start local optimization.
    
    # We will run Basin Hopping for a few iterations, then check time.
    # If time remains, we can run another instance or polish the best result.
    
    num_bh_runs = 3
    steps_per_bh = 50
    
    for run_idx in range(num_bh_runs):
        if time.time() > deadline - 30.0:
            break
            
        # Generate initial points
        # A good initialization is important.
        # We try a mix of random and structured.
        if run_idx == 0:
            # Structured initialization: vertices and centroid-ish
            pts = []
            # Inset vertices slightly
            pts.append(V0 + 0.1 * (V1 - V0) + 0.1 * (V2 - V0)) # Near V0
            pts.append(V1 + 0.1 * (V0 - V1) + 0.1 * (V2 - V1)) # Near V1
            pts.append(V2 + 0.1 * (V0 - V2) + 0.1 * (V1 - V2)) # Near V2
            # Centroid
            centroid = (V0 + V1 + V2) / 3.0
            pts.append(centroid)
            # Fill rest randomly
            for _ in range(N - 4):
                r1 = rng.random()
                r2 = rng.random()
                if r1 + r2 > 1.0:
                    r1 = 1.0 - r1
                    r2 = 1.0 - r2
                p = r1 * V1 + r2 * V2
                pts.append(p)
            x0 = np.array(pts).flatten()
        else:
            # Random initialization
            pts = []
            for _ in range(N):
                r1 = rng.random()
                r2 = rng.random()
                if r1 + r2 > 1.0:
                    r1 = 1.0 - r1
                    r2 = 1.0 - r2
                p = r1 * V1 + r2 * V2
                # Add some noise to avoid exact duplicates
                p += rng.normal(0, 0.01, 2)
                pts.append(p)
            x0 = np.array(pts).flatten()
        
        try:
            # Run Basin Hopping
            bh_result = basinhopping(
                objective,
                x0,
                minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 500, 'adaptive': True}},
                niter=steps_per_bh,
                stepsize=0.5, # Initial step size for random jumps
                T=100.0,       # Initial temperature for Metropolis acceptance
                accept='opt',  # Accept only if objective improves (or equal)
                minimizer=local_min # Use our custom local min wrapper? 
                                   # Actually, basinhopping uses minimizer_kwargs if minimizer is not provided?
                                   # No, if we provide minimizer, it uses that.
                                   # But local_min calls minimize inside. This might be redundant or slow.
                                   # Let's let basinhopping call minimize directly via minimizer_kwargs.
            )
        except Exception:
            # If basin hopping fails, try a simple local optimization
            try:
                result = minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                bh_result = result
            except Exception:
                continue
        
        if time.time() > deadline - 15.0:
            break
            
        # Get the best point found by basin hopping
        x_opt = bh_result.x
        points_opt = x_opt.reshape(-1, 2)
        
        # Project points to be strictly inside the triangle
        # We use a simple projection: if outside, move towards centroid
        centroid = (V0 + V1 + V2) / 3.0
        for i in range(N):
            p = points_opt[i]
            if not point_in_triangle(p):
                # Move towards centroid until inside
                # Simple linear interpolation
                t = 0.5
                while t < 1.0:
                    p_new = p + t * (centroid - p)
                    if point_in_triangle(p_new):
                        points_opt[i] = p_new
                        break
                    t += 0.1
                else:
                    # If still outside, just use centroid (should not happen with t=1)
                    points_opt[i] = centroid
        
        # Calculate score
        score = min_triangle_area(points_opt)
        
        if score > best_score:
            best_score = score
            best_points = points_opt.copy()
            
        # If we have time, we can try to polish the best_points found so far
        # by running a final local optimization on it
        if best_points is not None and time.time() < deadline - 20.0:
            try:
                result = minimize(
                    objective,
                    best_points.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
                )
                x_polished = result.x.reshape(-1, 2)
                # Re-project
                for i in range(N):
                    p = x_polished[i]
                    if not point_in_triangle(p):
                        t = 0.5
                        while t < 1.0:
                            p_new = p + t * (centroid - p)
                            if point_in_triangle(p_new):
                                x_polished[i] = p_new
                                break
                            t += 0.1
                        else:
                            x_polished[i] = centroid
                
                polished_score = min_triangle_area(x_polished)
                if polished_score > best_score:
                    best_score = polished_score
                    best_points = x_polished.copy()
            except Exception:
                pass

    if best_points is None:
        # Fallback: Random points
        rng = np.random.RandomState(0)
        pts = []
        for _ in range(N):
            r1 = rng.random()
            r2 = rng.random()
            if r1 + r2 > 1.0:
                r1 = 1.0 - r1
                r2 = 1.0 - r2
            p = r1 * V1 + r2 * V2
            pts.append(p)
        best_points = np.array(pts)
        
    return best_points
