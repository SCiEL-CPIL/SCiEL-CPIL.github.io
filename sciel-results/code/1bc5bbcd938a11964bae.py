import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay
import itertools

def get_score(points):
    """
    Calculate the score: minimum triangle area divided by convex hull area.
    """
    N = len(points)
    if N < 3:
        return 0.0

    # Check for degeneracy
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # Vectorized triangle area calculation
    # For N=13, M=286 triangles.
    idx = np.indices((N, N, N))
    i, j, k = idx
    mask = (i < j) & (j < k)
    i, j, k = i[mask], j[mask], k[mask]
    
    p_i = points[i]
    p_j = points[j]
    p_k = points[k]
    
    dx_ji = p_j[:, 0] - p_i[:, 0]
    dy_ji = p_j[:, 1] - p_i[:, 1]
    dx_ki = p_k[:, 0] - p_i[:, 0]
    dy_ki = p_k[:, 1] - p_i[:, 1]
    
    cross = dx_ji * dy_ki - dy_ji * dx_ki
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    
    return min_area / hull_area

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    N = 13
    best_score = 0.0
    best_points = None
    
    # 1. Baseline: Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = get_score(points_reg)
    best_points = points_reg.copy()
    
    # 2. Topology-Guided Optimization
    # Strategy: 
    # - Define a fixed Delaunay-like topology for 13 points.
    # - We will use a "core-shell" or "mesh" structure.
    # - Instead of optimizing 26 absolute coords, we optimize:
    #   - 1 center point (fixed at 0,0 for symmetry breaking? No, let it move slightly or fix to 0,0 to save DOF)
    #   - Let's fix point 0 at (0,0).
    #   - Points 1-4 form a central quadrilateral.
    #   - Points 5-12 form an outer ring.
    #   - Actually, a better approach for "non-regular":
    #     Use a predefined set of angles and radii, and optimize them.
    #     But angles/radii relative to what?
    #     
    # Let's try a specific parameterization:
    # - Point 0: (0, 0)
    # - Points 1-12: Polar coordinates (r_i, theta_i).
    # - We optimize r_i and theta_i.
    # - To avoid the "regular polygon" trap, we initialize with a perturbed, asymmetric configuration.
    # - We also add a penalty if the Delaunay topology changes drastically? 
    #   No, we just optimize the coords. The key is the INITIAL CONFIGURATION 
    #   and the PARAMETERIZATION which allows interior points easily.
    
    # Let's create a diverse set of initial configurations and optimize each.
    
    def polar_to_cartesian(params):
        # params[0] is dummy or center x, params[1] center y?
        # Let's fix center at 0,0 to reduce DOF and enforce symmetry breaking via radii/angles.
        # Actually, allowing the center to move might help if the optimal is not centered.
        # But for a convex region normalized to unit area, centering is usually good.
        # Let's fix point 0 at (0,0).
        # params[0:N-1] are radii, params[N-1:2*N-2] are angles?
        # Total 12 points in polar. 24 params.
        
        r = params[:12]
        theta = params[12:]
        
        pts = np.zeros((13, 2))
        pts[0] = [0.0, 0.0]
        pts[1:, 0] = r * np.cos(theta)
        pts[1:, 1] = r * np.sin(theta)
        
        return pts

    def neg_score_polar(params):
        pts = polar_to_cartesian(params)
        # Check for degeneracy early
        if np.any(np.abs(pts) < 1e-6) and not np.allclose(pts[0], 0):
             return 1.0
        s = get_score(pts)
        if s == 0.0:
            return 1.0
        return -s

    # Initial configurations
    configs = []
    
    # Config 1: Regular 12-gon around center, with small noise
    angles_init = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    radii_init = np.ones(12)
    # Add asymmetry
    radii_init = radii_init * (1 + 0.2 * np.random.normal(0, 1, 12))
    radii_init = np.clip(radii_init, 0.5, 1.5)
    angles_init = angles_init + 0.1 * np.random.normal(0, 1, 12)
    configs.append(np.concatenate([radii_init, angles_init]))
    
    # Config 2: Two concentric rings (6 inner, 6 outer)
    # Inner ring radius 0.5, Outer 1.0
    r_inner = np.full(6, 0.5)
    r_outer = np.full(6, 1.0)
    th_inner = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    th_outer = np.linspace(0, 2 * np.pi, 6, endpoint=False) + np.pi/6 # Offset
    # Perturb
    r_inner += 0.1 * np.random.normal(0, 1, 6)
    r_outer += 0.1 * np.random.normal(0, 1, 6)
    th_inner += 0.1 * np.random.normal(0, 1, 6)
    th_outer += 0.1 * np.random.normal(0, 1, 6)
    configs.append(np.concatenate([np.concatenate([r_inner, r_outer]), np.concatenate([th_inner, th_outer])]))
    
    # Config 3: Random "blob"
    r_rand = np.random.uniform(0.3, 1.2, 12)
    th_rand = np.random.uniform(0, 2 * np.pi, 12)
    configs.append(np.concatenate([r_rand, th_rand]))

    # Config 4: Based on best found so far (Regular 13-gon)
    # Convert regular 13-gon to this parameterization?
    # The regular 13-gon doesn't have a center point.
    # We can project it onto the center+12 polar model by finding the best fit?
    # Or just ignore it and rely on the other configs.
    # Actually, let's try to map the regular 13-gon points to this model.
    # It's hard because one point is at center in this model, but regular polygon has none.
    # So this model is structurally different. That's the point!

    for cfg in configs:
        if time.time() > deadline - 10:
            break
            
        try:
            # Use L-BFGS-B for speed, with bounds
            # Radii must be positive
            bounds = [(1e-4, 2.0)] * 12 + [(-np.pi, np.pi)] * 12
            res = minimize(neg_score_polar, cfg, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            current_score = -res.fun
            if current_score > best_score:
                best_score = current_score
                best_points = polar_to_cartesian(res.x).copy()
                
        except Exception as e:
            pass

    # 3. Local Refinement of the Best Point
    # Now that we have a better starting point (likely with interior points),
    # do a final Nelder-Mead on the full 2D coordinates to fine-tune.
    
    if best_points is not None:
        def neg_score_2d(x):
            pts = x.reshape(N, 2)
            return -get_score(pts)
        
        x0 = best_points.flatten()
        
        # Run a few quick Nelder-Mead restarts from this improved base
        for i in range(5):
            if time.time() > deadline - 5:
                break
            try:
                noise = np.random.normal(0, 0.01, N * 2)
                x_start = x0 + noise
                res = minimize(neg_score_2d, x_start, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
                
                current_score = -res.fun
                if current_score > best_score:
                    best_score = current_score
                    best_points = res.x.reshape(N, 2).copy()
                    x0 = best_points.flatten()
            except Exception:
                pass

    return best_points
