"""Program for packing unit regular hexagons inside a regular hexagon."""
import numpy as np
from scipy.optimize import minimize

N = 11

def _hexagon_vertices(cx, cy, side, angle_rad):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angles = np.deg2rad(angle_rad) + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _point_in_hexagon(px, py, cx, cy, side, angle_rad):
    """Check if point (px, py) is inside or on boundary of hexagon."""
    # Transform to local coordinates
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # In local coords, hexagon has vertices at side * (cos(k*pi/3), sin(k*pi/3)) for k=0..5
    # The hexagon can be described by |x| <= side and |x/2 + sqrt(3)*y/2| <= side and |-x/2 + sqrt(3)*y/2| <= side
    # Actually for a regular hexagon with "flat top" or "pointy top" orientation...
    # Standard regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300 degrees (pointy top)
    # The inequalities are: |x| <= side, |y| <= side * sqrt(3)/2 * 2 / sqrt(3) ... let me think again.
    
    # For a regular hexagon with circumradius = side, vertices at angles 0, 60, ..., 300:
    # The hexagon is the intersection of 6 half-planes.
    # For vertex at angle theta, the normal to the opposite edge is at theta + 180.
    # Actually, let's use the standard form: a regular hexagon with "radius" (center to vertex) = R = side
    # has the property that a point (x, y) is inside if and only if:
    # |x| <= R and |y| <= R * sqrt(3)/2 * 2/sqrt(3)... 
    
    # Let me use a simpler approach: check against all 6 edges.
    verts = _hexagon_vertices(0, 0, side, 0.0)  # local hexagon, no rotation, no center
    # For each edge, check that the point is on the correct side
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        # Edge from v1 to v2, normal pointing inward
        ex = v2[0] - v1[0]
        ey = v2[1] - v1[1]
        # Normal (rotate edge by -90 degrees for inward normal, need to determine sign)
        nx = -ey
        ny = ex
        # Check if origin is on the positive side
        dot_origin = nx * v1[0] + ny * v1[1]
        dot_point = nx * lx + ny * ly
        if dot_origin > 0:
            if dot_point < 0:
                return False
        else:
            if dot_point > 0:
                return False
    return True

def _separation_distance(hex1, hex2):
    """
    hex1, hex2: tuples (cx, cy, angle_rad)
    Returns minimum distance between the two hexagons (0 if overlapping).
    For now, approximate by checking if centers are far enough apart.
    """
    # This is complex; for optimization, we'll use a penalty based on center distance
    # and a simple check.
    pass

def construct():
    """
    Construct N unit hexagons inside a regular hexagon, minimizing the outer side length.
    Returns: (inner, center, side, angle)
      inner: list of [x, y, angle_degrees] for each inner hexagon
      center: (x, y) of outer hexagon
      side: side length of outer hexagon
      angle: rotation of outer hexagon in degrees
    """
    # Start with a grid-like arrangement
    # Place hexagons in a hexagonal grid pattern
    positions = []
    angles = []
    
    # Hexagonal grid: for a hex grid with spacing 2 (since side=1, distance between adjacent centers = 2)
    # Actually, for regular hexagons of side 1, the distance between centers of adjacent hexagons
    # in a hexagonal packing is 2 (since each hex has inradius sqrt(3)/2, and they tile with gap 0,
    # center-to-center distance is 2 * inradius = sqrt(3)... wait no.
    
    # Regular hexagon with side s: 
    # - Circumradius (center to vertex) = s
    # - Inradius (center to edge midpoint) = s * sqrt(3)/2
    # - Distance between centers of two adjacent hexagons in a tiling = 2 * inradius = s * sqrt(3)
    
    # So for s=1, center-to-center distance = sqrt(3) ≈ 1.732
    # But wait, that's for edge-to-edge contact. Let me verify:
    # Two hexagons sharing an edge: the distance between their centers is twice the inradius = 2 * sqrt(3)/2 = sqrt(3). Yes.
    
    # Hexagonal grid offsets:
    # Row 0: (0, 0)
    # Row 1: (sqrt(3)/2, 1.5), (sqrt(3), 1.5), ... wait, let me use the standard hex grid.
    
    # For a hex grid with "pointy top" hexagons:
    # Horizontal spacing: sqrt(3)
    # Vertical spacing: 1.5
    # Odd rows offset by sqrt(3)/2
    
    # Actually, let me just place them in a simple grid and optimize.
    
    # Initial placement: hexagonal grid
    grid_positions = []
    # Arrange in rows
    rows = 4  # 4 rows of about 3 hexagons each = 12, we need 11
    cols = 3
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N:
                break
            x = c * np.sqrt(3) + (r % 2) * np.sqrt(3) / 2
            y = r * 1.5
            grid_positions.append([x, y, 0.0])
            count += 1
        if count >= N:
            break
    
    inner = grid_positions
    
    # Outer hexagon: start with a reasonable estimate
    # The bounding hexagon needs to contain all inner hexagons.
    # Initial guess: center at centroid of inner hexagons, side = 4
    centers = np.array([[p[0], p[1]] for p in inner])
    outer_center = [float(np.mean(centers[:, 0])), float(np.mean(centers[:, 1]))]
    outer_side = 4.0
    outer_angle = 0.0
    
    # Optimize the outer hexagon to minimize side length while containing all inner hexagons
    # We'll use a simple iterative approach: for a given set of inner hexagons,
    # find the minimum enclosing regular hexagon.
    
    def _min_enclosing_hexagon(points, initial_center=None, initial_side=None):
        """Find minimum regular hexagon enclosing given points (centers of unit hexagons)."""
        # The inner hexagons have side 1, so their "extent" adds to the required outer size.
        # A point (center of inner hex) requires that the inner hex fits, which means
        # the distance from outer center to inner center plus the "radius" of inner hex
        # in that direction must be <= the inradius of outer hex in that direction.
        
        # For simplicity, let's approximate: the outer hexagon must contain all vertices
        # of all inner hexagons.
        
        # Generate all vertices of all inner hexagons
        all_verts = []
        for p in points:
            verts = _hexagon_vertices(p[0], p[1], 1.0, np.deg2rad(p[2]))
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Now find min enclosing regular hexagon for all_verts
        # This is a non-trivial optimization. Let's use a simple approach:
        # Binary search on side length, and for each side, check if there exists a center and angle
        # such that all points are inside.
        
        # For a regular hexagon with center (cx, cy), side s, angle a:
        # A point (px, py) is inside if and only if it satisfies 6 linear inequalities.
        
        # Let's use scipy.optimize to minimize side length.
        
        def _is_inside(px, py, cx, cy, s, a):
            """Check if point is inside regular hexagon."""
            dx = px - cx
            dy = py - cy
            ca = np.cos(-a)
            sa = np.sin(-a)
            lx = ca * dx - sa * dy
            ly = sa * dx + ca * dy
            # Regular hexagon with circumradius s, vertices at angles 0, 60, ..., 300
            # The 6 edge normals are at angles 30, 90, 150, 210, 270, 330 degrees
            # Edge i (0-indexed) has normal at angle (30 + 60*i) degrees
            # Inequality: lx * cos(theta_i) + ly * sin(theta_i) <= s * cos(30) = s * sqrt(3)/2
            for i in range(6):
                theta = np.deg2rad(30 + 60 * i)
                if lx * np.cos(theta) + ly * np.sin(theta) > s * np.sqrt(3) / 2 + 1e-9:
                    return False
            return True
        
        def _objective(params):
            cx, cy, a, s = params
            # s is the side length (circumradius)
            # Check all points
            for vx, vy in all_verts:
                if not _is_inside(vx, vy, cx, cy, s, a):
                    # Penalty: return a large number
                    return 1e6
            return s
        
        # Initial guess
        if initial_center is None:
            initial_center = [float(np.mean(all_verts[:, 0])), float(np.mean(all_verts[:, 1]))]
        if initial_side is None:
            # Estimate: max distance from center to any vertex
            dx = all_verts[:, 0] - initial_center[0]
            dy = all_verts[:, 1] - initial_center[1]
            dists = np.sqrt(dx**2 + dy**2)
            initial_side = float(np.max(dists)) + 1.0
        
        x0 = [initial_center[0], initial_center[1], 0.0, initial_side]
        
        # Try a few different initial angles
        best_result = None
        for init_angle in [0, 30, 60, 90, 120, 150]:
            x0_try = [initial_center[0], initial_center[1], np.deg2rad(init_angle), initial_side]
            try:
                result = minimize(_objective, x0_try, method='Nelder-Mead', 
                                  options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
                if result.fun < 1e5:
                    if best_result is None or result.fun < best_result.fun:
                        best_result = result
            except Exception:
                pass
        
        if best_result is not None and best_result.fun < 1e5:
            cx, cy, a, s = best_result.x
            return [float(cx), float(cy)], float(s), float(np.rad2deg(a))
        else:
            # Fallback: use a large hexagon
            return [float(np.mean(all_verts[:, 0])), float(np.mean(all_verts[:, 1]))], float(initial_side * 1.5), 0.0
    
    # Find minimum enclosing hexagon for the current inner arrangement
    outer_center, outer_side, outer_angle = _min_enclosing_hexagon(inner)
    
    # Now try to optimize the inner positions to reduce the outer side length
    # This is a complex optimization; let's do a simple local search.
    
    def _total_side_length(inner_pos):
        """Compute the outer side length needed for given inner positions."""
        oc, os_, oa = _min_enclosing_hexagon(inner_pos)
        return os_
    
    # Simple local search: perturb each inner hexagon slightly
    current_inner = [list(p) for p in inner]
    current_side = outer_side
    
    for iteration in range(50):
        improved = False
        for i in range(N):
            for dx in [-0.1, 0, 0.1]:
                for dy in [-0.1, 0, 0.1]:
                    for da in [-5, 0, 5]:
                        new_inner = [list(p) for p in current_inner]
                        new_inner[i][0] += dx
                        new_inner[i][1] += dy
                        new_inner[i][2] += da
                        try:
                            new_side = _total_side_length(new_inner)
                            if new_side < current_side - 1e-4:
                                current_inner = new_inner
                                current_side = new_side
                                improved = True
                        except Exception:
                            pass
        if not improved:
            break
    
    inner = current_inner
    outer_center, outer_side, outer_angle = _min_enclosing_hexagon(inner)
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
