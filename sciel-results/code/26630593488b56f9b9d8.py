import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9
SQRT3 = math.sqrt(3)

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Axial coordinates for 11 unit hexagons
    # We generate a set of candidate clusters.
    # A standard hexagonal number is 1, 7, 19... 
    # 11 is not a hex number, so we need to arrange 11 hexagons.
    # Common dense packings for 11:
    # 1. 3-4-3-1 (rows)
    # 2. 2-3-4-2 (rows)
    # 3. 1-3-4-3 (rows)
    # 4. 4-3-4 (rows)
    # 5. 2-4-3-2
    # Let's generate axial coordinates for these row-based structures.
    
    def generate_rows(row_lengths):
        """
        Generate axial coordinates for a hexagonal cluster defined by row lengths.
        Row i has length L_i. 
        Axial coordinates (q, r). 
        r corresponds to the row index (offset).
        q is the horizontal index.
        To center the cluster, we will shift coordinates later.
        """
        pts = []
        current_r = 0
        # We need to determine the horizontal offset for each row to keep it "centered" relative to the previous row.
        # In axial coords, moving from (q, r) to (q, r+1) shifts the row.
        # A standard hex grid:
        # Row 0: q=0
        # Row 1: q=-1, 0, 1 ...
        # Let's just generate a block and then center it.
        
        # A simpler way: generate points in a bounding box and filter/shift.
        # Let's use offset coordinates (col, row) where row is the axial r.
        # For a "brick wall" or hex grid, if we use axial (q, r):
        # y = 1.5 * r
        # x = sqrt(3) * (q + r/2)
        
        # Let's define the structure by listing (q, r) pairs directly for known good 11-hex packings.
        # Structure 1: "Hexagon with 4 corners removed" -> 19-4=15? No.
        # Let's list specific axial coordinates for 11 hexagons.
        
        # Configuration A: 3 rows of 4, 3, 4? No, 4+3+4=11.
        # Row 0 (r=0): q = -1, 0, 1, 2
        # Row 1 (r=1): q = -1, 0, 1
        # Row 2 (r=2): q = -1, 0, 1, 2
        # This is a parallelogram-like shape.
        
        # Configuration B: 2, 3, 4, 2
        # Row 0: q = 0, 1
        # Row 1: q = -1, 0, 1
        # Row 2: q = -1, 0, 1, 2
        # Row 3: q = 0, 1
        
        # Configuration C: 1, 3, 4, 3
        # Row 0: q = 0
        # Row 1: q = -1, 0, 1
        # Row 2: q = -1, 0, 1, 2
        # Row 3: q = -1, 0, 1
        
        # Configuration D: 4, 3, 4
        # Row 0: q = 0, 1, 2, 3
        # Row 1: q = 0, 1, 2
        # Row 2: q = 0, 1, 2, 3
        
        # Configuration E: 2, 4, 3, 2
        # Row 0: q = 0, 1
        # Row 1: q = -1, 0, 1, 2
        # Row 2: q = 0, 1, 2
        # Row 3: q = 0, 1
        
        # Configuration F: 5, 3, 3 (5+3+3=11)
        # Row 0: q = -2, -1, 0, 1, 2
        # Row 1: q = -1, 0, 1
        # Row 2: q = -1, 0, 1
        
        # Configuration G: 3, 5, 3
        # Row 0: q = -1, 0, 1
        # Row 1: q = -2, -1, 0, 1, 2
        # Row 2: q = -1, 0, 1
        
        # Configuration H: 3, 4, 4
        # Row 0: q = 0, 1, 2
        # Row 1: q = 0, 1, 2, 3
        # Row 2: q = 0, 1, 2, 3
        
        # Configuration I: 4, 4, 3
        # Row 0: q = 0, 1, 2, 3
        # Row 1: q = 0, 1, 2, 3
        # Row 2: q = 0, 1, 2
        
        # Configuration J: 2, 3, 3, 3
        # Row 0: q = 0, 1
        # Row 1: q = -1, 0, 1
        # Row 2: q = -1, 0, 1
        # Row 3: q = -1, 0, 1
        
        # Configuration K: 3, 3, 3, 2
        # Row 0: q = -1, 0, 1
        # Row 1: q = -1, 0, 1
        # Row 2: q = -1, 0, 1
        # Row 3: q = 0, 1
        
        # Configuration L: 1, 2, 3, 4, 1 (1+2+3+4+1 = 11)
        # Row 0: q = 0
        # Row 1: q = 0, 1
        # Row 2: q = -1, 0, 1
        # Row 3: q = -1, 0, 1, 2
        # Row 4: q = 0
        
        # Configuration M: 1, 4, 3, 3
        # Row 0: q = 0
        # Row 1: q = -1, 0, 1, 2
        # Row 2: q = -1, 0, 1
        # Row 3: q = -1, 0, 1

        configs = []
        
        # Add the configurations defined above
        # A
        configs.append([(-1,0), (0,0), (1,0), (2,0), (-1,1), (0,1), (1,1), (-1,2), (0,2), (1,2), (2,2)])
        # B
        configs.append([(0,0), (1,0), (-1,1), (0,1), (1,1), (-1,2), (0,2), (1,2), (2,2), (0,3), (1,3)])
        # C
        configs.append([(0,0), (-1,1), (0,1), (1,1), (-1,2), (0,2), (1,2), (2,2), (-1,3), (0,3), (1,3)])
        # D
        configs.append([(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (3,2)])
        # E
        configs.append([(0,0), (1,0), (-1,1), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3)])
        # F
        configs.append([(-2,0), (-1,0), (0,0), (1,0), (2,0), (-1,1), (0,1), (1,1), (-1,2), (0,2), (1,2)])
        # G
        configs.append([(-1,0), (0,0), (1,0), (-2,1), (-1,1), (0,1), (1,1), (2,1), (-1,2), (0,2), (1,2)])
        # H
        configs.append([(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (3,2)])
        # I
        configs.append([(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2)])
        # J
        configs.append([(0,0), (1,0), (-1,1), (0,1), (1,1), (-1,2), (0,2), (1,2), (-1,3), (0,3), (1,3)])
        # K
        configs.append([(-1,0), (0,0), (1,0), (-1,1), (0,1), (1,1), (-1,2), (0,2), (1,2), (0,3), (1,3)])
        # L
        configs.append([(0,0), (0,1), (1,1), (-1,2), (0,2), (1,2), (-1,3), (0,3), (1,3), (2,3), (0,4)])
        # M
        configs.append([(0,0), (-1,1), (0,1), (1,1), (2,1), (-1,2), (0,2), (1,2), (-1,3), (0,3), (1,3)])
        
        # Also try a "compact" 4x3 block minus 1 corner? 12-1=11.
        # 4 cols, 3 rows.
        # Row 0: 0,1,2,3
        # Row 1: 0,1,2,3
        # Row 2: 0,1,2
        # This is Config I.
        
        # Try shifting some to create different shapes?
        # The axial coordinates define the connectivity. The shapes above are connected.
        
        return configs

    candidates = generate_rows(None)
    
    # Function to calculate the minimal enclosing hexagon side length for a set of points
    # assuming the outer hexagon is centered at (cx, cy) and rotated by angle_deg.
    # The inner hexagons are also rotated by angle_deg (aligned).
    # If inner hexagons are rotated by 30 or 60, the geometry is different.
    # However, for a regular hexagon, rotating by 60 is the same. Rotating by 30 is a "pointy-top" vs "flat-top" switch relative to the grid.
    # If we align inner hexagons with the outer one (0 deg relative), the "radius" of the cluster is determined by the max projection.
    # If we rotate inner hexagons by 30 deg, the vertices point differently.
    
    # Let's stick to the 0 degree relative orientation first (aligned).
    # The problem allows rotation. But usually, aligned is optimal for dense packings.
    # We will check 0, 30, 60 degrees rotation for the *set* of inner hexagons relative to the outer.
    # Note: If we rotate the inner hexagons, their positions (x,y) in the lattice must be adjusted or we just rotate the whole cluster?
    # The interface asks for (x, y, angle_degrees) for each inner hexagon.
    # If we place them on a lattice and rotate the whole cluster, the lattice structure is preserved.
    # So we can just rotate the final coordinates and the angle.
    
    def calc_min_side(pts_cart, angle_deg):
        """
        Calculate the minimal side length S of a regular hexagon centered at origin
        (we will re-center later) that contains the given points (centers of unit hexagons)
        assuming the inner hexagons are rotated by angle_deg relative to the outer one?
        Actually, if we rotate the inner hexagons, their "footprint" changes.
        
        Let's assume the inner hexagons are aligned with the outer one (angle_diff = 0).
        The condition for a unit hexagon at center p to be inside a hexagon of side S centered at c with angle alpha:
        The distance from c to the boundary in direction theta is S * cos(theta - alpha) / cos(pi/6) ? No.
        
        The apothem of the outer hexagon is A_out = S * sqrt(3)/2.
        The apothem of the inner hexagon is A_in = 1 * sqrt(3)/2.
        
        If they are aligned, the condition is that the projection of the inner center p onto the normal of each side of the outer hexagon
        must be <= A_out - A_in.
        
        Let normals n_k for k=0..5 be the outward normals of the outer hexagon sides.
        n_k = (cos(alpha + k*pi/3), sin(alpha + k*pi/3)).
        
        Constraint: n_k . (p - c) <= A_out - A_in
        => n_k . p - n_k . c <= A_out - A_in
        
        We want to minimize S. S is proportional to A_out.
        Let D = A_out - A_in.
        We need to find c and D such that for all p, all k:
        n_k . p - n_k . c <= D
        
        This is equivalent to finding the minimal D such that the set of points {p} is contained in the hexagon defined by D.
        The minimal D for a given orientation alpha is:
        D_min = max_k ( max_p (n_k . p) - min_p (n_k . p) ) / 2 ?
        No, we can choose c.
        The width of the set of points in direction n_k is W_k = max_p (n_k . p) - min_p (n_k . p).
        If we center c optimally, the max deviation from center is W_k / 2.
        So we need D >= W_k / 2 for all k.
        Thus, D_min(alpha) = max_k ( W_k(alpha) / 2 ).
        
        Then A_out = D_min + A_in.
        S = A_out / (sqrt(3)/2) = 2 * A_out / sqrt(3).
        
        This assumes the inner hexagons are aligned with the outer one.
        If we rotate the inner hexagons by 30 degrees, their apothem is still A_in, but the "effective" projection might change?
        No, the apothem is the distance from center to side. If the hexagons are rotated, the condition is still that the entire inner hexagon is inside.
        For two convex polygons, one inside the other, the condition is more complex if rotated.
        However, for regular hexagons, if we rotate the inner one by 30 degrees, the vertices of the inner hexagon stick out further in the directions of the outer hexagon's sides?
        Actually, the "radius" to the side is the apothem. The "radius" to the vertex is the side length.
        If aligned: max projection of inner hexagon is A_in.
        If rotated 30 deg: the vertices align with the normals. The projection of a vertex is S_in = 1.0.
        So the effective apothem becomes 1.0 instead of sqrt(3)/2.
        So D_min would need to be larger.
        
        Therefore, aligned (0 deg) is likely better because A_in = 0.866 < 1.0.
        So we will only consider the case where inner hexagons are aligned with the outer one.
        """
        # Convert angle to radians
        alpha = math.radians(angle_deg)
        
        # Normals for the 6 sides of the outer hexagon
        # The sides are perpendicular to the normals.
        # For a flat-top hexagon (0 deg), normals are at 0, 60, 120...
        # Let's use the standard definition.
        normals = np.array([
            [math.cos(alpha + k * math.pi / 3.0), math.sin(alpha + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        # Project points onto normals
        # projs shape (n_pts, 6)
        projs = pts_cart @ normals.T
        
        # Width in each direction
        widths = np.max(projs, axis=0) - np.min(projs, axis=0)
        
        # Minimal D (half-width)
        D_min = np.max(widths) / 2.0
        
        # A_out = D_min + A_in
        A_in = math.sqrt(3) / 2.0
        A_out = D_min + A_in
        
        # S = 2 * A_out / sqrt(3)
        S = 2.0 * A_out / math.sqrt(3)
        
        # Center is the midpoint of the min/max projections
        # c = (max_p + min_p) / 2 for the "bounding" box in each direction?
        # Actually, the optimal center is the intersection of the midlines.
        # For a regular hexagon, the center that minimizes the max deviation is the centroid of the "bounding hexagon".
        # A good approximation is the average of the points, or the center of the bounding box in Cartesian.
        # Let's use the center of the bounding box in Cartesian for simplicity, or better, the center of the hexagon that contains them.
        # The center c should be such that c is the midpoint of the interval [min_p, max_p] for each normal?
        # No, that's only true if the widths are compatible.
        # For a minimal enclosing hexagon of a fixed orientation, the center is unique.
        # We can find it by solving for c such that max_k (max_p n_k.p - n_k.c) = min_k (min_p n_k.p - n_k.c) ... no.
        # Let's just use the centroid of the points as the center. It's a good heuristic.
        # Or, we can compute the center as the average of the midpoints of the projections?
        # Let's refine the center:
        # The constraint is n_k . c >= min_p (n_k . p) - D_min
        # and n_k . c <= max_p (n_k . p) - D_min
        # So for each k, n_k . c is in [min_k - D_min, max_k - D_min].
        # Let L_k = min_p (n_k . p) - D_min
        # Let U_k = max_p (n_k . p) - D_min
        # We need to find c such that L_k <= n_k . c <= U_k for all k.
        # This is a system of linear inequalities.
        # Since it's a regular hexagon, the intervals should overlap.
        # A simple solution is to take the average of the midpoints?
        # Midpoint M_k = (L_k + U_k) / 2 = (min_p + max_p) / 2.
        # Let's compute the center as the least squares solution to n_k . c = M_k.
        
        # Build system A c = b
        # A is 6x2, b is 6x1
        A_mat = normals
        b_vec = (np.min(projs, axis=0) + np.max(projs, axis=0)) / 2.0
        
        # Solve A c = b (overdetermined)
        c, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        center = c
        
        return S, center

    best_S = float('inf')
    best_config = None
    best_angle = 0.0
    best_center = None
    best_pts_cart = None

    # We will test the 3 orientations: 0, 30, 60 degrees for the outer hexagon.
    # Since the inner hexagons are aligned with the outer, rotating the outer hexagon
    # is equivalent to rotating the entire coordinate system.
    # The value of S is invariant under global rotation.
    # However, the "lattice" of inner hexagons is fixed in our candidate generation (axial coords).
    # If we rotate the outer hexagon, the inner hexagons (which are on the lattice) are no longer aligned with it?
    # No, the problem says "each inner hexagon may translate and rotate".
    # If we rotate the outer hexagon by 30 degrees, and keep the inner hexagons aligned with the *lattice* (0 degrees),
    # then the inner hexagons are rotated by -30 degrees relative to the outer.
    # As established, rotating inner hexagons by 30 degrees increases the required size (apothem 1.0 vs 0.866).
    # So we should keep the inner hexagons aligned with the outer hexagon.
    # This means if we rotate the outer hexagon, we must rotate the inner hexagons by the same amount.
    # Since the lattice is symmetric under 60 degree rotation, rotating the whole system by 0, 30, 60...
    # effectively changes the "orientation" of the lattice relative to the "axes" of the hexagon.
    # But since the hexagon is regular, the minimal S for a set of points is independent of the global rotation.
    # So we only need to calculate S for one orientation (e.g., 0 degrees) and then rotate the final coordinates to match the desired output angle?
    # The output angle is arbitrary as long as the geometry is correct.
    # So we can just output angle 0.0 and the corresponding coordinates.
    
    # Wait, the lattice points are generated in axial coordinates which map to Cartesian with a specific orientation (flat-top or pointy-top?).
    # My axial_to_cartesian uses:
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    # This corresponds to a flat-top hexagon grid?
    # Let's check.
    # (0,0) -> (0,0)
    # (1,0) -> (sqrt(3), 0)
    # (0,1) -> (sqrt(3)/2, 1.5)
    # The vector (1,0) is horizontal. The vector (0,1) is at 60 degrees?
    # tan(theta) = 1.5 / (sqrt(3)/2) = 3/sqrt(3) = sqrt(3) -> theta = 60.
    # So the grid is aligned with the standard flat-top hexagon.
    # The outer hexagon at 0 degrees (flat-top) is aligned with this grid.
    # So the "aligned" case is the 0 degree case.
    
    for config in candidates:
        # Convert axial to Cartesian
        pts = []
        for q, r in config:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        pts_cart = np.array(pts)
        
        # Calculate minimal side for 0 degree orientation
        S, center = calc_min_side(pts_cart, 0.0)
        
        if S < best_S:
            best_S = S
            best_config = config
            best_angle = 0.0
            best_center = center
            best_pts_cart = pts_cart

    # If we found a valid configuration
    if best_config is not None:
        cx, cy = best_center
        # Shift points so that the center of the outer hexagon is at (cx, cy)?
        # The calc_min_side returns the center of the enclosing hexagon.
        # The points are in the original lattice coordinates.
        # We need to return the positions of the inner hexagons.
        # The positions are best_pts_cart.
        # The outer hexagon center is best_center.
        # We should translate the inner points so that the outer hexagon is centered at the origin?
        # The interface returns (inner, center, side, angle).
        # "center" is the outer hexagon center.
        # So we can keep the points as they are, and set center to best_center.
        # Or we can shift everything so center is (0,0). It doesn't matter.
        # Let's shift to origin for cleanliness.
        
        inner_pts = best_pts_cart - np.array(best_center)
        
        inner = []
        for x, y in inner_pts:
            inner.append([float(x), float(y), float(best_angle)])
            
        center = [float(best_center[0]), float(best_center[1])]
        side = float(best_S)
        rotation = float(best_angle)
        
        return inner, center, side, rotation
    else:
        # Fallback
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
