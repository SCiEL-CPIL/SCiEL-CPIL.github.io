"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _hex_verts(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    theta = np.deg2rad(angle_deg)
    angles = theta + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _outer_contains(outer_verts, inner_verts):
    """Check if all inner vertices are inside or on the outer hexagon using cross products."""
    n = len(outer_verts)
    for j in range(6):
        p1 = outer_verts[j]
        p2 = outer_verts[(j + 1) % n]
        edge = p2 - p1
        # For a convex polygon with CCW ordering, all points should be on the left
        cross = np.cross(edge, inner_verts - p1)
        if np.any(cross < -1e-9):
            return False
    return True

def _pairwise_non_overlap(verts_list, tol=1e-9):
    """Check that no two hexagons overlap using separating axis theorem."""
    n = len(verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separates(verts_list[i], verts_list[j]):
                return False
    return True

def _separates(v1, v2):
    """Check if two convex polygons are separated by any edge normal."""
    for v in (v1, v2):
        for k in range(6):
            p1 = v[k]
            p2 = v[(k + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            # Project both polygons onto this normal
            proj1 = np.dot(v, normal)
            proj2 = np.dot(v2, normal)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
    return False

def _valid_packing(inner, center, side, angle):
    """Full validity check."""
    outer_verts = _hex_verts(center[0], center[1], side, angle)
    inner_verts_list = [_hex_verts(p[0], p[1], 1.0, p[2]) for p in inner]
    
    # Check containment
    for iv in inner_verts_list:
        if not _outer_contains(outer_verts, iv):
            return False
    
    # Check non-overlap
    if not _pairwise_non_overlap(inner_verts_list):
        return False
    
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    best_side = 40.0
    best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try structured hexagonal lattice arrangements with various parameters
    # The key insight: arrange unit hexagons in a hexagonal lattice pattern
    # inside a larger hexagon
    
    # For a hexagonal packing of unit hexagons, the optimal arrangement
    # places them on a triangular/hexagonal lattice
    
    # Generate candidate configurations using hexagonal lattice positions
    # A regular hexagon of side S can contain hexagons arranged in a hex pattern
    
    # Strategy: try different lattice spacings and orientations
    # For unit hexagons, the center-to-center distance in a hex lattice
    # is 2 * sqrt(3) / 2 * (1 + epsilon) for some small epsilon
    
    # Actually, for regular hexagons of side 1, the distance between centers
    # in a tight hex packing is 2 * sqrt(3) / 2 * 2 = 2*sqrt(3)
    # Wait, let me think more carefully.
    
    # A regular hexagon of side 1 has "radius" (center to vertex) = 1
    # and "apothem" (center to edge midpoint) = sqrt(3)/2
    # In a hexagonal tiling, adjacent hexagons share edges, so center distance = 2 * apothem = sqrt(3)
    # But we need them to be disjoint, so we need a small gap.
    
    # Let's use a parameterized approach: place hexagons on a hex lattice
    # with a small expansion factor
    
    import itertools
    
    # Hex lattice basis vectors
    # For hexagons, the natural lattice is triangular with spacing d
    # where d = 2 * (sqrt(3)/2) * (1 + gap) = sqrt(3) * (1 + gap)
    
    # Let's try various arrangements
    candidates = []
    
    # Simple row-based arrangements
    for gap in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]:
        d = np.sqrt(3) * (1 + gap)  # center-to-center distance
        
        # Create hexagonal lattice points
        # Rows offset by d/2 horizontally, spaced by d * sqrt(3)/2 vertically
        row_spacing = d * np.sqrt(3) / 2
        
        # Generate points in a hexagonal region
        max_r = 3  # radius in lattice units
        lattice_points = []
        for q in range(-max_r, max_r + 1):
            for r in range(-max_r, max_r + 1):
                # Axial to Cartesian for hex grid
                x = d * (q + r / 2)
                y = d * r * np.sqrt(3) / 2
                # Only keep points within a hexagonal region
                # Check if (q, r) is within hex of radius max_r
                if max(abs(q), abs(r), abs(q + r)) <= max_r:
                    lattice_points.append((x, y))
        
        if len(lattice_points) == 0:
            continue
        
        # Try different subset sizes and arrangements
        # For N=11, we want exactly 11 hexagons
        
        # Try the first N points sorted by distance from center
        lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        for n_pts in range(6, min(len(lattice_points), 15)):
            pts = lattice_points[:n_pts]
            if len(pts) < N:
                continue
            
            # Take exactly N points - the N closest to center
            inner = [[pts[i][0], pts[i][1], 0.0] for i in range(N)]
            center = (0.0, 0.0)
            
            # Estimate required outer side length
            # Find the maximum extent of the inner hexagons
            max_x = max(p[0] for p in inner)
            min_x = min(p[0] for p in inner)
            max_y = max(p[1] for p in inner)
            min_y = min(p[1] for p in inner)
            
            # The outer hexagon needs to contain all inner hexagons
            # A rough estimate: side length should be at least half the max extent
            # plus some margin for the hexagon shape
            extent = max(max_x - min_x, max_y - min_y) / 2
            side_est = extent + 1.5  # rough estimate
            
            candidates.append((side_est, inner, center, 0.0, gap))
    
    # Also try simple linear arrangements
    for spacing in [2.0, 2.1, 2.2, 2.5, 3.0]:
        inner = [[spacing * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        center = (0.0, 0.0)
        max_x = max(p[0] for p in inner)
        side_est = max_x + 1.5
        candidates.append((side_est, inner, center, 0.0, 0.0))
    
    # Sort candidates by estimated side length
    candidates.sort(key=lambda c: c[0])
    
    # Try each candidate and validate
    for est_side, inner, center, angle, gap in candidates:
        if time.time() - start_time > TIME_LIMIT:
            break
        
        # Try a range of outer side lengths around the estimate
        for side in [est_side - 0.5, est_side, est_side + 0.5, est_side + 1.0]:
            if side <= 0:
                continue
            
            if _valid_packing(inner, center, side, angle):
                if side < best_side:
                    best_side = side
                    best_inner = inner
                    best_center = center
                    best_angle = angle
                break  # Found a valid packing with this side, try next candidate
    
    # If no valid packing found, return the seed
    if best_side >= 40.0:
        return best_inner, best_center, 40.0, 0.0
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
