"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 12
import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Get vertices of a regular hexagon centered at (cx,cy) with given rotation."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _convex_hull(points):
    """Compute the convex hull of a set of points using monotone chain."""
    points = sorted(set(map(tuple, points)))
    if len(points) <= 1:
        return points
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
        
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
        
    return lower[:-1] + upper[:-1]

def _min_hexagon_side(polygon, outer_angle):
    """
    Calculate the minimum side length of a regular hexagon with given angle 
    that contains the given polygon.
    The hexagon is centered at the origin for calculation purposes.
    """
    angle_rad = np.radians(outer_angle)
    # Face normals of the hexagon
    normals = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3 + np.pi / 6
        normals.append(np.array([np.cos(a), np.sin(a)]))
    
    max_proj = 0
    for p in polygon:
        for n in normals:
            proj = np.dot(p, n)
            if proj > max_proj:
                max_proj = proj
                
    # The inradius is max_proj. Side length s = inradius / (sqrt(3)/2) = 2*inradius/sqrt(3)
    return 2.0 * max_proj / np.sqrt(3)

def _objective(params):
    """
    Objective: Minimize outer hexagon side length.
    params: [cx, cy, outer_angle]
    Inner hexagons are fixed in relative position (hexagonal cluster).
    """
    cx, cy, oa = params
    
    # Generate inner hexagon positions (hexagonal cluster)
    # Axial coordinates for a compact cluster of 12
    # Center at (0,0)
    # We want the 12 closest points in hex grid to origin.
    # Distances in hex grid:
    # 0: (0,0) - 1
    # 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) - 6
    # Total 7. Need 5 more.
    # Next ring distance 2:
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Let's just take the first 5 of the second ring in a symmetric way?
    # Or better: Use a 3x4 rectangle? No, hex cluster is better.
    # Let's use the 12 points:
    # Ring 0: (0,0)
    # Ring 1: all 6
    # Ring 2: 5 of them. To keep symmetry, maybe take (2,0), (1,1), (0,2), (-1,2), (-2,1)?
    # This breaks symmetry. 
    # Alternative: 4-4-4 rows?
    # Let's stick to the "closest 12 points" heuristic.
    
    axial = [(0,0)]
    for r in range(1, 10):
        # Generate ring r
        ring = []
        # 6 sides
        for side_idx in range(6):
            # Start point for this side
            # (q, r) axial
            # Side 0: q=r, r=0 -> (r, 0)
            # Side 1: q=0, r=r -> (0, r)
            # Side 2: q=-r, r=r -> (-r, r)
            # Side 3: q=-r, r=0 -> (-r, 0)
            # Side 4: q=0, r=-r -> (0, -r)
            # Side 5: q=r, r=-r -> (r, -r)
            # Actually, let's just iterate all points in bounding box and sort by distance
            pass
        
    # Simpler: Just hardcode a good cluster of 12.
    # A 3-4-3-2 arrangement is not symmetric.
    # A hexagon of side 2 is 7. Add 5 more.
    # Let's use a 4-4-4 strip? 
    # No, let's use the convex hull of the 12 closest lattice points.
    
    # Generate candidate points
    candidates = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            # Hex distance from origin: (|q| + |r| + |q+r|) / 2
            dist = (abs(q) + abs(r) + abs(q+r)) / 2.0
            candidates.append((dist, q, r))
            
    candidates.sort()
    # Take first 12
    top_12 = candidates[:12]
    
    # Convert to cartesian
    sqrt3 = np.sqrt(3)
    centers = []
    for _, q, r in top_12:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        centers.append((x, y))
        
    # Center the cluster at (0,0) for calculation, then shift by (cx, cy)
    # Actually, the objective function allows cx, cy to move the cluster.
    # So we can just use the raw centers and let cx, cy do the centering.
    
    # Collect all vertices of the 12 hexagons
    all_verts = []
    for cx_c, cy_c in centers:
        # Inner hexagons are aligned with outer (angle 0) for simplicity, 
        # or we can assume they are optimally oriented. 
        # For hexagons, rotation by 30 degrees swaps "flat top" vs "pointy top".
        # The lattice is defined by the centers. The orientation of individual hexagons 
        # doesn't change the convex hull of the union if they are identical and 
        # we consider the union. Wait, it does.
        # However, if we pack them on a lattice, they are usually aligned.
        # Let's assume inner angle = 0.
        verts = _hex_vertices(cx_c, cy_c, 0.0, 1.0)
        all_verts.extend(verts.tolist())
        
    # Compute convex hull of all vertices
    hull = _convex_hull(all_verts)
    
    # Shift hull by (cx, cy)
    hull_shifted = [np.array(p) + np.array([cx, cy]) for p in hull]
    
    # Find min hexagon side containing this hull
    # The hexagon is centered at (0,0) in the objective's local frame?
    # No, the outer hexagon center is also a variable? 
    # In the current interface, we return outer center.
    # But we can assume WLOG that the outer hexagon is centered at (0,0) 
    # by shifting the inner cluster.
    # So, we fix outer center at (0,0) and optimize inner cluster position (cx, cy) and outer angle.
    
    side = _min_hexagon_side(hull_shifted, oa)
    
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial guess: center of cluster at (0,0), angle 0
    x0 = np.array([0.0, 0.0, 0.0])
    
    best_side = np.inf
    best_params = x0
    
    # Try a few initial angles
    for angle in [0, 30, 60, 90, 120, 150]:
        x0_trial = np.array([0.0, 0.0, angle])
        s = _objective(x0_trial)
        if s < best_side:
            best_side = s
            best_params = x0_trial
            
    # Optimize
    # We can use Nelder-Mead on the 3 parameters
    # Or even better, since the problem is convex (min enclosing hexagon of a convex set),
    # we can solve it more directly.
    # But let's stick to a fast optimizer.
    
    for attempt in range(5):
        if time.time() > deadline - 10:
            break
            
        x0 = best_params.copy()
        if attempt > 0:
            x0 += np.random.normal(0, 0.1, size=3)
            
        try:
            result = minimize(_objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
            if result.fun < best_side:
                best_side = result.fun
                best_params = result.x
        except Exception:
            pass
            
    # Construct final answer
    cx, cy, oa = best_params
    
    # Generate inner positions
    candidates = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            dist = (abs(q) + abs(r) + abs(q+r)) / 2.0
            candidates.append((dist, q, r))
            
    candidates.sort()
    top_12 = candidates[:12]
    
    sqrt3 = np.sqrt(3)
    inner = []
    for _, q, r in top_12:
        x = 1.5 * q + cx
        y = sqrt3 * (r + q / 2.0) + cy
        # Angle is 0
        inner.append([float(x), float(y), 0.0])
        
    # Outer center is (0,0) in our optimization frame?
    # Wait, _objective assumed outer center at (0,0) and shifted inner by (cx, cy).
    # So the outer hexagon center is (0,0).
    # But we can return any center. Let's return (0,0).
    outer_center = (0.0, 0.0)
    
    side = float(best_side)
    angle = float(oa)
    
    return inner, outer_center, side, angle

# EVOLVE-BLOCK-END
