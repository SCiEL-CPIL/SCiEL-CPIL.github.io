import numpy as np
import time

N = 12

def _hex_vertices(cx, cy, s, angle_deg):
    """Get vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + s * np.cos(theta)
        y = cy + s * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _hex_face_normals(angle_deg):
    """Get the 6 outward face normals of a regular hexagon with rotation angle_deg."""
    a = np.radians(angle_deg)
    normals = []
    for i in range(6):
        theta = a + i * np.pi / 3 + np.pi / 6  # face normals are offset by 30 deg
        normals.append([np.cos(theta), np.sin(theta)])
    return np.array(normals)

def _min_enclosing_hexagon(inner_positions, inner_angles, inner_sides):
    """
    Find minimal regular hexagon (center, side, angle) containing all inner hexagons.
    Uses support function: for a regular hexagon with side s, center c, angle phi,
    the apothem is s*sqrt(3)/2. A point p is inside iff for all 6 face normals n_i,
    dot(p - c, n_i) <= apothem.
    
    We search over the outer hexagon angle phi, and for each phi, compute the minimal
    side length needed.
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (cx, cy, ang), s in zip(inner_positions, inner_sides):
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    best_side = float('inf')
    best_center = None
    best_angle = 0.0
    
    # The outer hexagon angle can be restricted to [0, 60) due to symmetry
    # Try many angles
    for phi_deg in np.linspace(0, 60, 600):
        phi = np.radians(phi_deg)
        # Face normals of outer hexagon
        normals = []
        for i in range(6):
            theta = phi + i * np.pi / 3 + np.pi / 6
            normals.append([np.cos(theta), np.sin(theta)])
        normals = np.array(normals)  # shape (6, 2)
        
        # For each face normal, compute max and min projection of all vertices
        # The hexagon must contain all points, so:
        # For normal n_i: min_proj <= dot(c, n_i) <= max_proj - 2*apothem... 
        # Actually: for all points p, dot(p - c, n_i) <= apothem AND dot(p - c, -n_i) <= apothem
        # => dot(c, n_i) >= dot(p, n_i) - apothem for all p
        # => dot(c, n_i) <= dot(p, n_i) + apothem for all p
        # So: max_p dot(p, n_i) - apothem <= dot(c, n_i) <= min_p dot(p, n_i) + apothem
        # Wait, that's not right. Let me redo.
        # 
        # Point p is inside hexagon centered at c with apothem a iff:
        #   dot(p - c, n_i) <= a for all i
        #   dot(p - c, -n_i) <= a for all i  (i.e., dot(c - p, n_i) <= a)
        # So: dot(p, n_i) - dot(c, n_i) <= a  =>  dot(c, n_i) >= dot(p, n_i) - a
        # And: dot(c, n_i) - dot(p, n_i) <= a  =>  dot(c, n_i) <= dot(p, n_i) + a
        # For ALL points p and ALL normals n_i:
        #   dot(c, n_i) >= max_p [dot(p, n_i) - a]  =>  dot(c, n_i) + a >= max_p dot(p, n_i)
        #   dot(c, n_i) <= min_p [dot(p, n_i) + a]  =>  dot(c, n_i) - a <= min_p dot(p, n_i)
        # Combining: max_p dot(p, n_i) - min_p dot(p, n_i) <= 2a
        # So the width in direction n_i must be <= 2a = s*sqrt(3)
        # And c must satisfy: max_p dot(p, n_i) - a <= dot(c, n_i) <= min_p dot(p, n_i) + a
        # For feasibility: max_p dot(p, n_i) - a <= min_p dot(p, n_i) + a
        # => max_p dot(p, n_i) - min_p dot(p, n_i) <= 2a
        
        # Compute projections
        projs = all_verts @ normals.T  # shape (num_verts, 6)
        max_projs = projs.max(axis=0)  # shape (6,)
        min_projs = projs.min(axis=0)  # shape (6,)
        widths = max_projs - min_projs  # shape (6,)
        
        # Required apothem: a >= max(widths) / 2
        required_apothem = np.max(widths) / 2
        s_candidate = required_apothem * 2 / np.sqrt(3)
        
        # Check if there exists a center c satisfying all constraints
        # For each normal n_i: dot(c, n_i) in [max_projs[i] - a, min_projs[i] + a]
        # With a = required_apothem, the interval is [max_projs[i] - a, min_projs[i] + a]
        # Note: max_projs[i] - min_projs[i] <= 2a, so the interval is non-empty
        # But we need a single c that satisfies ALL 6 constraints simultaneously
        
        # The center c is determined by the intersection of 6 strips in 2D.
        # For a regular hexagon, the 6 normals are at 60-degree intervals.
        # The intersection of these strips is a hexagon (or empty).
        # We can find the center as the average of the constraint midpoints if feasible.
        
        # Actually, for a regular hexagon, the center is uniquely determined by the
        # requirement that the hexagon is regular. Let's use a different approach:
        # The center should be at the centroid of the bounding region.
        
        # For each normal, the midpoint of the allowed range for dot(c, n_i) is:
        # mid_i = (max_projs[i] + min_projs[i]) / 2
        # The center c should satisfy dot(c, n_i) = mid_i for all i (if the widths are all equal)
        # In general, we need to solve for c such that dot(c, n_i) is in [lo_i, hi_i] for all i.
        
        lo = max_projs - required_apothem
        hi = min_projs + required_apothem
        
        # Check feasibility: for each i, lo[i] <= hi[i]
        if np.any(lo > hi + 1e-12):
            continue
        
        # Find center c: we want dot(c, n_i) to be in [lo[i], hi[i]] for all i.
        # Since normals are symmetric, the optimal center is where dot(c, n_i) = (lo[i]+hi[i])/2
        # if that's feasible. Let's try the midpoint.
        mid = (lo + hi) / 2
        
        # Solve for c from dot(c, n_i) = mid[i] for i=0..5
        # This is an overdetermined system. Use least squares.
        # normals is 6x2, mid is 6x1
        c, residuals, rank, sv = np.linalg.lstsq(normals, mid, rcond=None)
        
        # Verify the solution satisfies all constraints
        c_projs = normals @ c
        if np.any(c_projs < lo - 1e-9) or np.any(c_projs > hi + 1e-9):
            # Try to find a feasible center by adjusting
            # For now, skip if not feasible (shouldn't happen if widths <= 2a)
            continue
        
        if s_candidate < best_side:
            best_side = s_candidate
            best_center = c
            best_angle = phi_deg
    
    if best_center is None:
        # Fallback: use centroid and a large side
        centroid = all_verts.mean(axis=0)
        return centroid[0], centroid[1], 100.0, 0.0
    
    return best_center[0], best_center[1], best_side, best_angle

def _hexagon_center_distance(p1, p2, s1, s2, a1, a2):
    """
    Approximate check: minimum distance between two hexagons.
    For regular hexagons of sides s1, s2 with centers p1, p2 and rotations a1, a2,
    the minimum distance is approx |p1-p2| - s1 - s2 (conservative).
    For exact check, we'd need polygon distance.
    """
    d = np.linalg.norm(p1 - p2)
    return d - s1 - s2  # negative means overlap

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Place 12 unit hexagons in a honeycomb pattern
    # Honeycomb grid: centers at (i * sqrt(3), j * 1.5) with offsets
    # For pointy-top hexagons, the grid spacing is:
    #   horizontal: sqrt(3) * s
    #   vertical: 1.5 * s
    #   odd rows offset by sqrt(3)/2 * s
    
    s = 1.0  # unit hexagons
    sqrt3 = np.sqrt(3)
    
    # Generate honeycomb positions
    # We want 12 hexagons in a compact arrangement
    # Try a 3-4-3-2 arrangement or similar
    
    # Standard honeycomb: row j has hexagons at x = (i + (j%2)*0.5) * sqrt3, y = j * 1.5
    # Let's try different cluster shapes
    
    best_result = None
    best_score = float('inf')
    
    # Configuration 1: centered hexagonal cluster (7 in inner ring + 5 from outer)
    # Actually, let's try multiple configurations
    
    configurations = []
    
    # Config A: 3-4-3-2 (rows of 3, 4, 3, 2)
    # Config B: 4-3-3-2 
    # Config C: 2-3-4-3
    # Config D: centered 1-6-5 (center + 6 around + 5 more)
    
    # Let me generate honeycomb positions systematically
    # For a hexagonal grid with spacing sqrt3 (horizontal) and 1.5 (vertical):
    
    def honeycomb_positions(count, pattern=None):
        """Generate positions for 'count' hexagons in honeycomb pattern."""
        if pattern is None:
            # Try to make it as compact as possible
            # Use rows
            positions = []
            row = 0
            while len(positions) < count:
                # Max hexagons in this row to keep compact
                # Alternate row lengths
                if row % 2 == 0:
                    max_in_row = 4
                else:
                    max_in_row = 3
                for i in range(max_in_row):
                    if len(positions) >= count:
                        break
                    x = (i + 0.5 * (row % 2)) * sqrt3
                    y = row * 1.5
                    positions.append((x, y, 0.0))
                row += 1
            return positions[:count]
        else:
            # pattern is a list of row lengths
            positions = []
            for j, row_len in enumerate(pattern):
                for i in range(row_len):
                    x = (i + 0.5 * (j % 2)) * sqrt3
                    y = j * 1.5
                    positions.append((x, y, 0.0))
            return positions[:count]
    
    # Try several row patterns that sum to 12
    patterns = [
        [3, 4, 3, 2],
        [4, 3, 3, 2],
        [2, 3, 4, 3],
        [3, 3, 4, 2],
        [4, 4, 2, 2],
        [2, 2, 4, 4],
        [5, 3, 2, 2],
        [2, 2, 3, 5],
        [6, 3, 3],
        [3, 3, 6],
        [4, 4, 4],
        [1, 6, 5],
        [5, 6, 1],
        [2, 4, 4, 2],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [3, 5, 4],
        [4, 5, 3],
        [5, 4, 3],
        [3, 4, 5],
    ]
    
    for pattern in patterns:
        if sum(pattern) != 12:
            continue
        if len(pattern) > 6:  # too many rows
            continue
        
        positions = honeycomb_positions(12, pattern)
        if len(positions) != 12:
            continue
        
        # Also try with some rotation of individual hexagons
        # For honeycomb tiling, all hexagons have the same orientation
        # But we can try rotating all by 30 degrees
        
        for rot in [0.0, 30.0]:
            inner = [(x, y, rot) for (x, y, _) in positions]
            inner_sides = [1.0] * 12
            
            cx, cy, side, angle = _min_enclosing_hexagon(inner, [r[2] for r in inner], inner_sides)
            
            # Add small safety margin for numerical precision
            side_safe = side * 1.0001 + 1e-6
            
            if side_safe < best_score:
                best_score = side_safe
                best_result = (inner, (cx, cy), side_safe, angle)
        
        if time.time() > deadline:
            break
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [(4.0 * k, 0.0, 0.0) for k in range(-6, 6)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
