import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_distance_to_boundary(px, py, cx, cy, side, angle_rad):
    """
    Compute signed distance from point (px,py) to the boundary of a regular hexagon.
    Positive = outside, negative = inside.
    Uses the 6 edge normals for a regular hexagon.
    """
    dx = px - cx
    dy = py - cy
    # Rotate into hexagon frame
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    # For a regular hexagon with side s, the distance from center to each edge is s*sqrt(3)/2
    # The 6 edge normals are at angles 0, 60, 120, 180, 240, 300 degrees (for flat-top)
    # Actually, for a hexagon with vertices at angles 0,60,...,300, the edges are perpendicular to directions 30,90,...,330
    # Let's use the standard: for a hexagon with "radius" = side, the inradius = side*sqrt(3)/2
    # The 6 supporting lines have normals at angles pi/6, pi/2, 5pi/6, 7pi/6, 3pi/2, 11pi/6
    
    inradius = side * np.sqrt(3) / 2.0
    angles = np.pi/6 + np.arange(6) * np.pi/3
    nx = np.cos(angles)
    ny = np.sin(angles)
    
    dots = lx * nx + ly * ny
    return np.max(dots) - inradius

def _compute_outer_side_for_config(inner_positions, inner_angles, outer_angle_rad):
    """
    Given inner hexagon centers and angles, compute the minimum outer hexagon side length
    that contains all of them, assuming outer hexagon is centered at origin.
    
    inner_positions: (N, 2) array of centers
    inner_angles: (N,) array of angles in radians
    outer_angle_rad: angle of outer hexagon
    
    Returns: minimum side length of outer hexagon
    """
    max_dist = 0.0
    
    for i in range(N):
        cx, cy = inner_positions[i]
        angle = inner_angles[i]
        
        # Compute the 6 vertices of this inner hexagon
        # Vertices at angles: angle + 0, 60, 120, 180, 240, 300 degrees (for flat-top)
        # Actually, let's define: for a hexagon with "angle" parameter, vertices are at:
        # angle + k*60 degrees for k=0..5, at distance = side (circumradius = side for regular hex)
        # Wait, for a regular hexagon, circumradius = side, inradius = side*sqrt(3)/2
        
        # The farthest point of this hexagon in any direction is at distance = side (circumradius)
        # from its center, in the direction of one of its vertices
        
        # To find the minimum enclosing hexagon, we need to check each vertex of each inner hex
        # and see how far it extends in the directions perpendicular to the outer hex's edges
        
        # For each of the 6 edge normals of the outer hexagon, find the maximum projection
        # of any inner hexagon vertex onto that normal, plus the inradius of the inner hex
        
        # Actually, simpler: for each inner hex, the maximum extent in direction theta is:
        # cx*cos(theta) + cy*sin(theta) + side * max over vertices of cos(vertex_angle - theta)
        # For a regular hex with side 1, the support function in direction theta is:
        # cos(min over k of |theta - (angle + k*pi/3)| mod pi/3) * 1
        # Actually, the support function of a regular hexagon with circumradius R in direction theta is:
        # R * cos(delta) where delta = min over the 6 vertex angles of the angular distance to theta
        
        # Let's just compute all 6 vertices and find the max projection for each outer edge normal
        
        vertex_angles = angle + np.arange(6) * np.pi/3
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)
        
        # Outer hex edge normals (for flat-top outer hex at outer_angle_rad)
        outer_normal_angles = outer_angle_rad + np.pi/6 + np.arange(6) * np.pi/3
        
        for on in outer_normal_angles:
            onx = np.cos(on)
            ony = np.sin(on)
            # Project all 6 vertices onto this normal
            projections = vx * onx + vy * ony
            max_proj = np.max(projections)
            # The distance from outer center (origin) to the outer edge in this direction is:
            # outer_side * sqrt(3) / 2 (inradius of outer hex)
            # So we need: outer_inradius >= max_proj
            # outer_side >= 2 * max_proj / sqrt(3)
            required_side = 2.0 * max_proj / np.sqrt(3)
            if required_side > max_dist:
                max_dist = required_side
    
    return max_dist

def _check_validity(inner_positions, inner_angles, outer_side, outer_angle_rad, tolerance=1e-6):
    """
    Check if the configuration is valid:
    1. No two inner hexagons overlap
    2. All inner hexagons are contained in the outer hexagon
    """
    # Check overlap between inner hexagons
    for i in range(N):
        for j in range(i+1, N):
            cx1, cy1 = inner_positions[i]
            cx2, cy2 = inner_positions[j]
            a1, a2 = inner_angles[i], inner_angles[j]
            
            # Check if any vertex of hex i is inside hex j
            v1_angles = a1 + np.arange(6) * np.pi/3
            v1x = cx1 + np.cos(v1_angles)
            v1y = cy1 + np.sin(v1_angles)
            
            for k in range(6):
                d = _hex_distance_to_boundary(v1x[k], v1y[k], cx2, cy2, 1.0, a2)
                if d > -tolerance:  # vertex is outside or on boundary
                    # Check if it's actually inside (d < 0 means inside)
                    if d < 0:
                        return False  # overlap!
            
            # Check if any vertex of hex j is inside hex i
            v2_angles = a2 + np.arange(6) * np.pi/3
            v2x = cx2 + np.cos(v2_angles)
            v2y = cy2 + np.sin(v2_angles)
            
            for k in range(6):
                d = _hex_distance_to_boundary(v2x[k], v2y[k], cx1, cy1, 1.0, a1)
                if d < 0:
                    return False  # overlap!
    
    # Check containment in outer hexagon
    for i in range(N):
        cx, cy = inner_positions[i]
        angle = inner_angles[i]
        
        v_angles = angle + np.arange(6) * np.pi/3
        vx = cx + np.cos(v_angles)
        vy = cy + np.sin(v_angles)
        
        for k in range(6):
            d = _hex_distance_to_boundary(vx[k], vy[k], 0.0, 0.0, outer_side, outer_angle_rad)
            if d > tolerance:
                return False  # vertex outside outer hex
    
    return True

def _build_hexagonal_lattice_config(scale_factor, offset_x=0.0, offset_y=0.0, rotation=0.0):
    """
    Build a hexagonal lattice configuration of 12 hexagons.
    
    For a hexagonal lattice with spacing d between centers:
    - Row 0: 4 hexagons
    - Row 1: 4 hexagons (offset by d/2 in x)
    - Row 2: 4 hexagons
    
    Actually, let's use a more symmetric arrangement:
    Center hexagon + 5 around it + 6 more = 12
    
    Better: use a hexagonal number pattern. 12 is not a hexagonal number,
    so we'll use a rectangular-ish hex lattice.
    
    Let's use: 3 rows of 4, with staggered arrangement.
    """
    # Hexagonal lattice spacing: for non-overlapping unit hexagons,
    # center-to-center distance must be at least 2*sqrt(3)/2 = sqrt(3) for adjacent in same row
    # Actually, for two unit hexagons to be disjoint, their centers must be at least
    # 2*inradius = sqrt(3) apart if aligned, but it depends on relative orientation.
    # For identical orientation, minimum center distance is 2*side = 2 for flat-top aligned.
    # Wait, let me recalculate: two unit hexagons with same orientation, the minimum
    # center distance for no overlap is 2*inradius = sqrt(3) if they're aligned edge-to-edge.
    # Actually, for two regular hexagons of side 1 with the same orientation,
    # the minimum distance between centers for no overlap is 2 * (sqrt(3)/2) = sqrt(3)
    # if they share an edge. But if they're vertex-to-vertex, it could be different.
    # For safety, let's use center distance >= 2.0 for same orientation.
    
    # Let's use a hexagonal lattice with spacing s
    s = scale_factor  # center-to-center distance
    
    positions = []
    angles = []
    
    # 3 rows of 4 hexagons in a hexagonal lattice
    # Row 0: 4 hexagons at y = 0
    # Row 1: 4 hexagons at y = s*sqrt(3)/2, offset by s/2 in x
    # Row 2: 4 hexagons at y = s*sqrt(3)
    
    for row in range(3):
        for col in range(4):
            x = (col - 1.5) * s
            if row % 2 == 1:
                x += s / 2.0
            y = row * s * np.sqrt(3) / 2.0
            # Center the whole configuration
            x += offset_x
            y += offset_y
            positions.append([x, y])
            angles.append(rotation)
    
    return np.array(positions), np.array(angles)

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # 5s safety margin before 150s limit
    
    best_side = np.inf
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0
    
    # Try different lattice configurations and outer hexagon orientations
    # The outer hexagon can be rotated, and the inner lattice can be scaled
    
    # For a hexagonal lattice of 12 unit hexagons, the minimum enclosing hexagon
    # depends on the lattice spacing and orientation.
    
    # Strategy: try different scale factors and outer hexagon rotations
    # Start with scale = sqrt(3) (minimum for non-overlapping) and increase
    
    # First, let's try a few promising configurations
    
    # Configuration 1: Hexagonal lattice, scale = sqrt(3) (tight packing)
    # The lattice has 3 rows of 4. The extent:
    # x: from -1.5*s to 1.5*s + s/2 = 2s for s=sqrt(3), so width = 2*sqrt(3)
    # y: from 0 to 2*sqrt(3)/2 * 2 = 2*sqrt(3)
    # This is a roughly square region, but we need a hexagon.
    
    # Let's try different outer hexagon angles to find the best fit
    
    scale_factors = [np.sqrt(3), 1.7, 1.8, 1.9, 2.0, 2.1]
    outer_angles = [0.0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6]
    
    for scale in scale_factors:
        if time.time() > deadline - 10:
            break
        
        positions, angles = _build_hexagonal_lattice_config(scale)
        
        for outer_angle in outer_angles:
            if time.time() > deadline - 5:
                break
            
            # Compute required outer side
            side = _compute_outer_side_for_config(positions, angles, outer_angle)
            
            # Check validity
            if _check_validity(positions, angles, side, outer_angle):
                if side < best_side:
                    best_side = side
                    best_positions = positions.copy()
                    best_angles = angles.copy()
                    best_outer_angle = outer_angle
    
    # If no valid configuration found, use a fallback
    if best_positions is None:
        # Fallback: use a larger scale that definitely works
        scale = 2.5
        positions, angles = _build_hexagonal_lattice_config(scale)
        outer_angle = 0.0
        side = _compute_outer_side_for_config(positions, angles, outer_angle)
        best_side = side
        best_positions = positions
        best_angles = angles
        best_outer_angle = outer_angle
    
    # Convert angles from radians to degrees for output
    inner = []
    for i in range(N):
        inner.append([best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])])
    
    return inner, (0.0, 0.0), best_side, np.degrees(best_outer_angle)

# EVOLVE-BLOCK-END
