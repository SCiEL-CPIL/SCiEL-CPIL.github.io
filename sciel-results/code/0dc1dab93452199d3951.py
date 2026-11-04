import numpy as np
import time

N = 11
MIN_DIST = np.sqrt(3)

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def cost_func(params):
        # params is a 1D array of 22 elements: [x1, y1, x2, y2, ..., x11, y11]
        points = params.reshape(-1, 2)
        
        # Center the points at the origin to assume the outer hexagon is centered there
        # This is a heuristic; the optimal outer hexagon might be shifted, 
        # but for a symmetric container and cluster, centering is often optimal.
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Calculate the side length L of the smallest enclosing regular hexagon (centered at 0,0)
        # Distance from center to side is L * sqrt(3) / 2
        # Projection of points onto normals must be <= L * sqrt(3) / 2
        # L = max( max_i |p_i . n_k| ) / (sqrt(3)/2)
        
        projections = np.abs(points_centered @ normals.T)
        max_proj = np.max(projections)
        L_enc = max_proj / (np.sqrt(3) / 2)
        
        # Calculate overlap penalty
        # We want distance >= MIN_DIST
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Combine costs
        # We want to minimize L_enc, but heavily penalize overlaps
        # If there are overlaps, the solution is invalid, so we add a large penalty
        return L_enc + 100.0 * penalty

    # Random restarts to find a good local minimum
    best_L = float('inf')
    best_params = None
    
    # Initial guess: place points in a hexagonal grid pattern roughly
    # Side length of cluster ~ 3 or 4?
    # 11 points. A hexagon of side 2 (radius 2) has 1+6+12=19 points? No.
    # Side 1: 1 point. Side 2: 7 points. Side 3: 19 points.
    # So 11 points fit in a hexagon of side ~2.something.
    # Let's start with random points in a circle of radius 3.
    
    for restart in range(50):
        # Random initialization
        if restart == 0:
            # Hexagonal lattice initialization
            # Generate points in a hexagonal grid
            pts = []
            for q in range(-3, 4):
                for r in range(-3, 4):
                    # Convert axial to cartesian
                    x = 1.5 * q
                    y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                    pts.append([x, y])
            # Select 11 closest to center
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            indices = np.argsort(dists)[:N]
            init_pts = pts[indices]
            # Add some noise
            init_pts += np.random.normal(0, 0.1, init_pts.shape)
        else:
            init_pts = np.random.uniform(-3, 3, (N, 2))
            
        initial_params = init_pts.flatten()
        
        # Use Nelder-Mead for optimization
        from scipy.optimize import minimize
        res = minimize(cost_func, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.success:
            final_cost = res.fun
            # Extract L_enc from the best params
            final_pts = res.x.reshape(-1, 2)
            centroid = np.mean(final_pts, axis=0)
            final_pts_centered = final_pts - centroid
            projections = np.abs(final_pts_centered @ normals.T)
            max_proj = np.max(projections)
            L_enc = max_proj / (np.sqrt(3) / 2)
            
            # Check for overlaps
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(final_pts_centered[i] - final_pts_centered[j])
                    if d < MIN_DIST - 1e-5:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and L_enc < best_L:
                best_L = L_enc
                best_params = res.x
                # Optional: break early if we find a very good solution
                if best_L < 3.0: # Theoretical min is likely around 3-4?
                    # Side 2 hexagon has 7 points. Side 3 has 19.
                    # 11 points should fit in side ~2.5?
                    pass

    # If no valid solution found, fallback to a safe large solution
    if best_params is None:
        # Return the seed solution or a large one
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct the output
    # The inner hexagons are at best_params positions (centered at 0,0)
    # The outer hexagon side length is best_L + 1
    # The outer hexagon center is (0,0)
    # The inner hexagons have angle 0.0
    
    inner_pts = best_params.reshape(-1, 2)
    inner = [[float(x), float(y), 0.0] for x, y in inner_pts]
    
    outer_side = best_L + 1.0
    
    return inner, (0.0, 0.0), outer_side, 0.0
