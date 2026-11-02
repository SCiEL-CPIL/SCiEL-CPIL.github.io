"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    # For N=12, a 3x4 grid of unit hexagons is a strong candidate.
    # Unit hexagon (flat-top):
    #   width = 2 * side = 2.0
    #   height = sqrt(3) * side = sqrt(3)
    #   distance between centers in same row = 2 * side = 2.0
    #   distance between centers in adjacent rows = sqrt(3) * side = sqrt(3)
    #   offset in x between adjacent rows = side = 1.0
    
    # Let's create a 3x4 grid (3 rows, 4 columns).
    # Row 0 (bottom): 4 hexagons
    # Row 1 (middle): 4 hexagons, offset by 1.0 in x
    # Row 2 (top): 4 hexagons, same x as row 0
    
    sqrt3 = np.sqrt(3)
    
    # Row 0 centers (y=0)
    # To center the grid, let's place 4 centers symmetrically around x=0.
    # x values: -3, -1, 1, 3 (step 2) -> width span 6, centered at 0
    row0_x = np.array([-3.0, -1.0, 1.0, 3.0])
    row0_y = 0.0
    
    # Row 1 centers (y=sqrt3)
    # Offset by 1.0 in x. To keep symmetry, offset the whole row.
    # If row0 is -3,-1,1,3, and we offset by 1, we get -2,0,2,4.
    # This shifts the grid to the right.
    # Let's center the entire cluster.
    # Total x-range of row0: [-3, 3]
    # Total x-range of row1 (unshifted): [-2, 4]
    # Combined x-range: [-3, 4] -> center at 0.5
    # Let's just build the points and re-center them.
    
    points = []
    # Row 0
    for x in row0_x:
        points.append([x, row0_y])
    # Row 1
    row1_x = row0_x + 1.0 # -2, 0, 2, 4
    row1_y = sqrt3
    for x in row1_x:
        points.append([x, row1_y])
    # Row 2
    row2_x = row0_x
    row2_y = 2 * sqrt3
    for x in row2_x:
        points.append([x, row2_y])
        
    points = np.array(points)
    
    # Center the cluster at the origin
    centroid = points.mean(axis=0)
    points -= centroid
    
    inner = []
    for x, y in points:
        inner.append([float(x), float(y), 0.0])
        
    # Now, find the minimum side length of a regular hexagon centered at (0,0)
    # that contains all 12 unit hexagons.
    # A regular hexagon with side length S, centered at origin, rotated by angle theta.
    # The boundary is defined by 6 lines.
    # For a flat-top hexagon (theta=0), the supporting lines are:
    #   x = S
    #   x = -S
    #   x/2 + y*sqrt(3)/2 = S
    #   x/2 + y*sqrt(3)/2 = -S
    #   -x/2 + y*sqrt(3)/2 = S
    #   -x/2 + y*sqrt(3)/2 = -S
    # Wait, the distance from center to each edge is the apothem a = S * sqrt(3)/2.
    # The 6 normals for a flat-top hexagon are at angles 0, 60, 120, 180, 240, 300 degrees.
    # Actually, for a "flat-top" hexagon, the vertices are at 0, 60, ..., 300.
    # The edges are perpendicular to the directions 30, 90, 150, 210, 270, 330.
    # Let's define the hexagon by its 6 outward normals.
    # If the hexagon is rotated by alpha, the normals are at alpha + k*60 for k=0..5.
    # The condition for a point p to be inside is:
    #   dot(p, n_k) <= a  for all k, where a is the apothem.
    #   a = S * sqrt(3)/2  =>  S = 2a / sqrt(3)
    
    # We need to find S and alpha that minimize S such that all vertices of all 12 inner hexagons are inside.
    # Actually, we just need to ensure all 12 unit hexagons are inside.
    # It's sufficient to check the vertices of the inner hexagons.
    # Each inner hexagon has 6 vertices. 12 * 6 = 72 vertices total.
    
    # Generate all 72 vertices of the inner hexagons.
    all_vertices = []
    for x, y, _ in inner:
        # Vertices of a unit flat-top hexagon centered at (x,y)
        # Angles: 0, 60, 120, 180, 240, 300
        for k in range(6):
            angle = np.deg2rad(60 * k)
            vx = x + np.cos(angle)
            vy = y + np.sin(angle)
            all_vertices.append([vx, vy])
            
    all_vertices = np.array(all_vertices)
    
    def objective(params):
        a, alpha_rad = params
        if a < 0:
            return 1e10
        alpha = alpha_rad
        # Normals
        normals = []
        for k in range(6):
            angle = alpha + np.deg2rad(60 * k)
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        
        # Max dot product of any vertex with any normal
        # This is the maximum "distance" in the direction of each normal.
        # We need max over all vertices and all normals of dot(v, n) <= a
        # Actually, for each normal, we find the max dot product over all vertices.
        # The required apothem is the max of these.
        # But we are optimizing a and alpha. We want to minimize a.
        # For a fixed alpha, the minimal a is max over all normals n_k of max over all vertices v of dot(v, n_k).
        # So for a given alpha, a_min(alpha) = max_k ( max_v dot(v, n_k(alpha)) )
        # We want to minimize a_min(alpha) over alpha.
        
        # Compute max dot product for each normal
        # all_vertices is (72, 2), normals is (6, 2)
        # dots = all_vertices @ normals.T  -> (72, 6)
        dots = all_vertices @ normals.T
        max_dots = dots.max(axis=0)  # (6,)
        required_a = max_dots.max()
        
        # If required_a > a, we have a violation.
        # We want to minimize a, so we can just return required_a.
        # But wait, the objective function in scipy.optimize.minimize finds the minimum.
        # So if we return required_a, it will find the alpha that minimizes required_a.
        # Then the optimal a is that value.
        
        return required_a
    
    # Optimize over alpha in [0, pi/3) due to 60 degree symmetry.
    # Let's try a few initial guesses for alpha.
    best_a = 1e10
    best_alpha = 0.0
    
    # Since the function is periodic and relatively smooth, we can use a simple grid search
    # followed by local optimization, or just use scipy.optimize.minimize with multiple starts.
    
    # Let's do a coarse grid search first.
    for alpha_deg in np.arange(0, 60, 1):
        alpha_rad = np.deg2rad(alpha_deg)
        normals = []
        for k in range(6):
            angle = alpha_rad + np.deg2rad(60 * k)
            normals.append([np.cos(angle), np.sin(angle)])
        normals = np.array(normals)
        dots = all_vertices @ normals.T
        max_dots = dots.max(axis=0)
        a_val = max_dots.max()
        if a_val < best_a:
            best_a = a_val
            best_alpha = alpha_rad
            
    # Now refine with local optimization around the best found alpha.
    try:
        res = minimize(lambda x: objective(x), x0=[best_a, best_alpha], method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 2000})
        if res.success and res.x[0] < best_a:
            best_a = res.x[0]
            best_alpha = res.x[1]
    except Exception:
        pass
        
    # Convert apothem a to side length S
    S = 2 * best_a / sqrt3
    
    return inner, (0.0, 0.0), float(S), float(np.rad2deg(best_alpha))
# EVOLVE-BLOCK-END
