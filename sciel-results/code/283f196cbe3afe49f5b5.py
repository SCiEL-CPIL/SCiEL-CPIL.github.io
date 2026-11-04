import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 11
MIN_DIST = np.sqrt(3)
SQRT3 = np.sqrt(3)

def construct():
    # Normal vectors for the 6 sides of a regular hexagon (flat-top orientation)
    # Angles: 0, 60, 120, 180, 240, 300 degrees
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    def min_enclosing_hexagon_side(points):
        """
        Compute the minimum side length of a regular hexagon (with fixed orientation 
        matching 'normals') that encloses the given points.
        The hexagon is allowed to be translated but not rotated.
        
        For a regular hexagon with side L and center C, the constraints are:
        For each normal n_k:
            max_i (p_i . n_k) - min_i (p_i . n_k) <= L * SQRT3
            (The width of the point set in direction n_k must be <= width of hexagon)
        
        Wait, the width of a regular hexagon of side L in any of its 6 normal directions 
        is L * SQRT3.
        So if we fix the orientation of the hexagon (normals fixed), the minimum 
        translation that encloses the points is such that the center C is the midpoint 
        of the projections on each normal.
        
        Let proj_k = p_i . n_k.
        min_k = min_i(proj_k), max_k = max_i(proj_k).
        Width w_k = max_k - min_k.
        For the hexagon to enclose the points, we need w_k <= L * SQRT3 for all k.
        So L >= max_k(w_k) / SQRT3.
        
        The center C can be chosen freely to center the projections.
        Specifically, for each normal k, the hexagon's projection interval is 
        [C . n_k - L*SQRT3/2, C . n_k + L*SQRT3/2].
        We need min_k <= C . n_k - L*SQRT3/2 and max_k >= C . n_k + L*SQRT3/2.
        This implies C . n_k = (min_k + max_k) / 2 is optimal for centering, 
        and the constraint is L >= w_k / SQRT3.
        
        So the minimum L is max_k(w_k) / SQRT3.
        """
        if len(points) < 3:
            return 0.0
        projections = points @ normals.T
        min_proj = np.min(projections, axis=0)
        max_proj = np.max(projections, axis=0)
        widths = max_proj - min_proj
        max_width = np.max(widths)
        L = max_width / SQRT3
        return L

    def cost_func(params):
        # params is a 1D array of 22 elements: [x1, y1, x2, y2, ..., x11, y11]
        points = params.reshape(-1, 2)
        
        # Center the points at the origin to remove translational invariance
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Calculate the side length L of the smallest enclosing regular hexagon
        L_enc = min_enclosing_hexagon_side(points_centered)
        
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
    for restart in range(100):
        # Random initialization
        if restart == 0:
            # Hexagonal lattice initialization
            pts = []
            for q in range(-3, 4):
                for r in range(-3, 4):
                    x = 1.5 * q
                    y = SQRT3/2 * q + SQRT3 * r
                    pts.append([x, y])
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            indices = np.argsort(dists)[:N]
            init_pts = pts[indices]
            init_pts += np.random.normal(0, 0.1, init_pts.shape)
        else:
            # Try different initialization strategies
            if restart % 3 == 1:
                # Random in circle
                r = np.random.uniform(0, 2.5, N)
                theta = np.random.uniform(0, 2*np.pi, N)
                init_pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
            else:
                # Random in square
                init_pts = np.random.uniform(-2, 2, (N, 2))
            
        initial_params = init_pts.flatten()
        
        # Use Nelder-Mead for optimization
        res = minimize(cost_func, initial_params, method='Nelder-Mead', 
                       options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5})
        
        if res.success:
            final_pts = res.x.reshape(-1, 2)
            centroid = np.mean(final_pts, axis=0)
            final_pts_centered = final_pts - centroid
            L_enc = min_enclosing_hexagon_side(final_pts_centered)
            
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

    # If no valid solution found, fallback to a safe large solution
    if best_params is None:
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
