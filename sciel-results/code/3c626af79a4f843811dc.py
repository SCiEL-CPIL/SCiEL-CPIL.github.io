import numpy as np
import time
from scipy.optimize import minimize

N = 11
MIN_DIST = np.sqrt(3)

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def compute_min_enclosing_hexagon(points, normals):
        """
        Given points, find the minimum side length L of a regular hexagon
        centered at the centroid of points that encloses all points.
        """
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # For a regular hexagon with side length L, the distance from center to each side is L * sqrt(3) / 2
        # The projection of each point onto each normal must be <= L * sqrt(3) / 2
        # So L = max( max_i |p_i . n_k| ) / (sqrt(3)/2)
        
        projections = np.abs(points_centered @ normals.T)
        max_proj = np.max(projections)
        L_enc = max_proj / (np.sqrt(3) / 2)
        
        return L_enc, points_centered, centroid
    
    def cost_func(params):
        # params is a 1D array of 22 elements: [x1, y1, x2, y2, ..., x11, y11]
        points = params.reshape(-1, 2)
        
        # Compute minimum enclosing hexagon
        L_enc, points_centered, centroid = compute_min_enclosing_hexagon(points, normals)
        
        # Calculate overlap penalty
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST:
                    penalty += (MIN_DIST - d) ** 2
        
        # Combine costs
        return L_enc + 100.0 * penalty
    
    def get_best_solution(params):
        """Validate and extract solution from params."""
        points = params.reshape(-1, 2)
        L_enc, points_centered, centroid = compute_min_enclosing_hexagon(points, normals)
        
        # Check for overlaps
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(points_centered[i] - points_centered[j])
                if d < MIN_DIST - 1e-5:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            return L_enc, points_centered, centroid
        return None
    
    # Random restarts to find a good local minimum
    best_L = float('inf')
    best_points_centered = None
    best_centroid = None
    
    # Hexagonal lattice initialization
    def hex_lattice_init():
        pts = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                x = 1.5 * q
                y = np.sqrt(3)/2 * q + np.sqrt(3) * r
                pts.append([x, y])
        pts = np.array(pts)
        dists = np.linalg.norm(pts, axis=1)
        indices = np.argsort(dists)[:N]
        init_pts = pts[indices]
        return init_pts
    
    for restart in range(100):
        # Random initialization
        if restart == 0:
            init_pts = hex_lattice_init()
            init_pts += np.random.normal(0, 0.1, init_pts.shape)
        elif restart % 3 == 1:
            init_pts = hex_lattice_init()
            init_pts += np.random.normal(0, 0.3, init_pts.shape)
        else:
            init_pts = np.random.uniform(-3, 3, (N, 2))
            
        initial_params = init_pts.flatten()
        
        # Use Nelder-Mead for optimization
        res = minimize(cost_func, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        solution = get_best_solution(res.x)
        if solution is not None:
            L_enc, points_centered, centroid = solution
            if L_enc < best_L:
                best_L = L_enc
                best_points_centered = points_centered
                best_centroid = centroid

    # If no valid solution found, fallback to a safe large solution
    if best_points_centered is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Construct the output
    inner = [[float(x), float(y), 0.0] for x, y in best_points_centered]
    
    # The outer hexagon side length is best_L (it's the enclosing hexagon)
    # But we need to ensure the inner hexagons fit. The centers are at best_points_centered,
    # and the enclosing hexagon has side length best_L.
    # Since the inner hexagons have side length 1, we need to check if they fit.
    # The distance from center to side of inner hexagon is sqrt(3)/2.
    # So the effective "radius" of each inner hexagon in the direction of the outer hexagon's sides is sqrt(3)/2.
    # We need to add this to the enclosing hexagon's side length? No, the enclosing hexagon is defined by the centers.
    # Actually, the enclosing hexagon must contain the entire inner hexagons, not just their centers.
    # So we need to inflate the enclosing hexagon by the inradius of the inner hexagons.
    # Inradius of unit hexagon = sqrt(3)/2
    # So the side length of the outer hexagon should be best_L + 2 * (sqrt(3)/2) / (sqrt(3)/2) = best_L + 2?
    # No, let's think again.
    
    # The distance from the center of the outer hexagon to its sides is L_outer * sqrt(3) / 2.
    # The centers of the inner hexagons are at distance at most L_enc * sqrt(3) / 2 from the center.
    # Each inner hexagon extends by sqrt(3)/2 in all directions from its center.
    # So the maximum distance from the outer hexagon center to any point on an inner hexagon is:
    # L_enc * sqrt(3) / 2 + sqrt(3) / 2 = (L_enc + 1) * sqrt(3) / 2
    # This must be <= L_outer * sqrt(3) / 2
    # So L_outer >= L_enc + 1
    
    outer_side = best_L + 1.0
    
    return inner, (float(best_centroid[0]), float(best_centroid[1])), outer_side, 0.0
