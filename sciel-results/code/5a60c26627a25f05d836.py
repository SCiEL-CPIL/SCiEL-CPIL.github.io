"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START

def _hexagon_vertices(cx, cy, side, angle_deg):
    """
    Return vertices of a regular hexagon centered at (cx, cy) with side length 'side'
    and rotation 'angle_deg' (degrees).
    Vertices are ordered counter-clockwise starting from the rightmost point (angle 0).
    For a regular hexagon, the distance from center to vertex is 'side'.
    Angles for vertices: 0, 60, 120, 180, 240, 300 degrees + angle_deg.
    """
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack((x, y))

def _hexagon_support(cx, cy, side, angle_deg, theta):
    """
    Compute the support function of a regular hexagon in direction theta.
    h(theta) = cx*cos(theta) + cy*sin(theta) + side * max_k (cos(theta - angle_k))
    where angle_k are the vertex angles relative to the hexagon's local frame.
    Actually, the support function of a regular hexagon of side 's' centered at origin
    with one vertex at angle alpha is:
    h(theta) = s * max( cos(theta - alpha - k*pi/3) for k in 0..5 )
    Wait, the vertices are at angles alpha + k*pi/3.
    The support function is max over vertices v_i of (theta . v_i).
    v_i = s * (cos(alpha + i*pi/3), sin(alpha + i*pi/3))
    theta . v_i = s * (cos(theta)cos(alpha+i*pi/3) + sin(theta)sin(alpha+i*pi/3))
    = s * cos(theta - alpha - i*pi/3)
    So h(theta) = s * max_i cos(theta - alpha - i*pi/3)
    Note that cos(x) has period 2pi and symmetry. The max of cos over a set of angles
    spaced by pi/3 is determined by the closest angle to 0 (or 2pi).
    Essentially, h(theta) = s * cos( delta ) where delta is the angular distance
    from (theta - alpha) to the nearest multiple of pi/3.
    Since cos is even and periodic, we can compute:
    delta = (theta - alpha) % (pi/3)
    if delta > pi/6: delta = pi/3 - delta
    h = s * cos(delta)
    """
    # alpha is the rotation of the hexagon (angle_deg in radians)
    alpha = np.deg2rad(angle_deg)
    # We need max over i of cos(theta - alpha - i*pi/3)
    # This is equivalent to s * cos( min_i | (theta - alpha - i*pi/3) % 2pi - pi | + pi ) ? No.
    # Simpler: the function g(phi) = max_i cos(phi - i*pi/3) has period pi/3 and is even around 0, pi/6, etc.
    # Specifically, g(phi) = cos( phi_mod ) where phi_mod = (phi + pi/6) % (pi/3) - pi/6
    # Let's verify:
    # If phi = 0, phi_mod = (pi/6)%(pi/3) - pi/6 = pi/6 - pi/6 = 0. cos(0)=1. Correct.
    # If phi = pi/6, phi_mod = (pi/3)%(pi/3) - pi/6 = 0 - pi/6 = -pi/6. cos(-pi/6)=sqrt(3)/2.
    # Max cos(k*pi/3 - pi/6) for k=0..5:
    # k=0: cos(-pi/6)=0.866
    # k=1: cos(pi/6)=0.866
    # k=2: cos(pi/2)=0
    # k=3: cos(5pi/6)=-0.866
    # k=4: cos(7pi/6)=-0.866
    # k=5: cos(3pi/2)=0
    # Max is 0.866. Correct.
    # If phi = pi/3, phi_mod = (pi/2)%(pi/3) - pi/6 = pi/6 - pi/6 = 0. cos(0)=1.
    # Max cos(k*pi/3 - pi/3):
    # k=0: cos(-pi/3)=0.5
    # k=1: cos(0)=1
    # k=2: cos(pi/3)=0.5
    # Max is 1. Correct.
    
    phi = theta - alpha
    phi_mod = (phi + np.pi/6) % (np.pi/3) - np.pi/6
    return cx * np.cos(theta) + cy * np.sin(theta) + side * np.cos(phi_mod)

def _minimal_enclosing_hexagon(inner_positions, inner_angles, inner_side=1.0):
    """
    Find the minimal enclosing regular hexagon for a set of unit hexagons.
    We assume the outer hexagon is centered at the origin for now, and we optimize
    its rotation and side length.
    
    For a fixed outer rotation 'rot', the minimal side length 'S' is determined by
    the maximum of the support functions of all inner hexagons in 6 directions.
    Specifically, for the outer hexagon to contain all inner hexagons, its support
    function h_outer(theta) must be >= h_inner_i(theta) for all theta and all i.
    It suffices to check the 6 vertex directions of the outer hexagon? No.
    The condition for one convex polygon P to be inside another Q is that
    h_P(theta) <= h_Q(theta) for all theta.
    For regular hexagons, it suffices to check the 6 normal directions of the
    outer hexagon? Not necessarily, but due to symmetry, checking the 6 vertex
    angles of the outer hexagon might be sufficient if the inner shapes are also
    hexagons?
    
    Actually, a simpler approach:
    For a fixed outer rotation 'rot', the outer hexagon (centered at origin) has
    vertices at angles rot + k*pi/3.
    The support function of the outer hexagon is S * max_k cos(theta - rot - k*pi/3).
    We need S * max_k cos(theta - rot - k*pi/3) >= h_inner_i(theta) for all theta, i.
    This is equivalent to:
    S >= max_i max_theta [ h_inner_i(theta) / max_k cos(theta - rot - k*pi/3) ]
    
    However, this ratio is tricky to compute analytically.
    Alternative: The minimal enclosing hexagon of a convex polygon (the convex hull
    of all inner hexagon vertices) can be found by rotating calipers or similar.
    But since we are optimizing rotation, let's use a numerical approach.
    
    Let's define the "width" of the cluster in direction theta as W(theta) = max_i h_inner_i(theta) - min_i h_inner_i(theta + pi).
    Wait, if the outer hexagon is centered at the origin, its support function is symmetric: h_outer(theta) = h_outer(theta+pi).
    So we need h_outer(theta) >= max( h_inner_i(theta), -h_inner_i(theta+pi) )? No.
    h_outer(theta) >= h_inner_i(theta) for all theta.
    Since h_outer is symmetric, h_outer(theta) = h_outer(theta+pi).
    So we need h_outer(theta) >= max_i h_inner_i(theta).
    Let M(theta) = max_i h_inner_i(theta).
    We need S * g(theta - rot) >= M(theta) for all theta, where g(phi) = max_k cos(phi - k*pi/3).
    So S >= max_theta [ M(theta) / g(theta - rot) ].
    
    We can sample theta in [0, 2pi) and compute this maximum.
    """
    # Precompute inner hexagon parameters
    # We will sample theta
    thetas = np.linspace(0, 2*np.pi, 720, endpoint=False)
    
    def compute_min_side(rot):
        # rot is in radians
        # g(theta - rot) = max_k cos(theta - rot - k*pi/3)
        # This is the same formula as before
        phi = thetas - rot
        phi_mod = (phi + np.pi/6) % (np.pi/3) - np.pi/6
        g = np.cos(phi_mod)
        
        # Compute M(theta) = max_i h_inner_i(theta)
        # h_inner_i(theta) = cx_i cos(theta) + cy_i sin(theta) + 1.0 * cos( (theta - angle_i) mod pi/3 ... )
        # Let's vectorize this.
        # inner_positions: (N, 2)
        # inner_angles: (N,) in radians
        
        # cx, cy
        cx = inner_positions[:, 0]
        cy = inner_positions[:, 1]
        angles = inner_angles
        
        # term1: cx*cos(theta) + cy*sin(theta)
        # shape: (N, 1) * (1, 720) -> (N, 720)
        c = np.cos(thetas)
        s = np.sin(thetas)
        linear = cx[:, np.newaxis] * c[np.newaxis, :] + cy[:, np.newaxis] * s[np.newaxis, :]
        
        # term2: cos( (theta - angle_i) mod pi/3 ... )
        # phi_i = theta - angle_i
        # shape: (720, 1) - (1, N) -> (720, N)
        phi = thetas[:, np.newaxis] - angles[np.newaxis, :]
        phi_mod = (phi + np.pi/6) % (np.pi/3) - np.pi/6
        radial = np.cos(phi_mod)
        
        # h_inner_i(theta) = linear.T + radial
        # shape: (N, 720)
        h_inner = linear + radial.T
        
        M = np.max(h_inner, axis=0)
        
        # ratio = M / g
        # g should be positive?
        # g(phi) = max cos(...). The min value of g is cos(pi/6) = sqrt(3)/2 > 0.
        # So no division by zero.
        ratio = M / g
        return np.max(ratio)

    # Optimize rot
    # The function is periodic with period pi/3 (60 degrees).
    # We can search rot in [0, pi/6] (30 degrees) due to symmetry?
    # g(theta - rot) has period pi/3.
    # M(theta) is fixed.
    # The maximum of M(theta)/g(theta-rot) might have period pi/3.
    # Let's just search rot in [0, 2*pi] or [0, pi/3].
    
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds
    
    best_side = np.inf
    best_rot = 0.0
    
    # Coarse search
    rots = np.linspace(0, np.pi/3, 100, endpoint=False)
    for rot in rots:
        if time.time() > deadline:
            break
        side = compute_min_side(rot)
        if side < best_side:
            best_side = side
            best_rot = rot
    
    # Fine search around best_rot
    if time.time() < deadline - 10:
        # Golden section search or simple grid
        range_size = np.pi/1000
        rots_fine = np.linspace(best_rot - range_size, best_rot + range_size, 100)
        for rot in rots_fine:
            if time.time() > deadline:
                break
            side = compute_min_side(rot)
            if side < best_side:
                best_side = side
                best_rot = rot
        
        # Even finer
        if time.time() < deadline - 2:
            range_size = np.pi/10000
            rots_finer = np.linspace(best_rot - range_size, best_rot + range_size, 100)
            for rot in rots_finer:
                if time.time() > deadline:
                    break
                side = compute_min_side(rot)
                if side < best_side:
                    best_side = side
                    best_rot = rot

    return best_side, np.degrees(best_rot)

def construct():
    # Define the 11 unit hexagons in a hexagonal lattice arrangement
    # Center at (0,0), surrounded by 6, then another ring?
    # For N=11, a common packing is a central hexagon, 6 around it, and 4 more?
    # Or a 3x3 grid?
    # Let's use a compact hexagonal cluster.
    # Positions for a hexagonal number arrangement:
    # Layer 0: (0,0)
    # Layer 1: 6 hexagons around
    # Layer 2: 12 hexagons around (total 19)
    # We have 11. So Layer 0 + Layer 1 + 4 from Layer 2?
    # Or just a linear chain? No, that's inefficient.
    # Let's use a 3-4-3 arrangement or similar.
    # Hexagonal lattice coordinates:
    # Axial coordinates (q, r).
    # Distance between centers is sqrt(3) for side 1?
    # Side length 1. Distance between centers of touching hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    # Let's generate positions.
    
    positions = []
    angles = []
    
    # Center
    positions.append([0.0, 0.0])
    angles.append(0.0)
    
    # 6 neighbors
    for k in range(6):
        angle = k * 60.0
        rad = np.deg2rad(angle)
        x = np.sqrt(3) * np.cos(rad)
        y = np.sqrt(3) * np.sin(rad)
        positions.append([x, y])
        angles.append(0.0) # Rotation doesn't matter for identical hexagons in terms of shape, only position matters for the union.
    
    # 4 more to make 11?
    # Let's add 4 from the next ring.
    # Next ring positions are at distance 2*sqrt(3) * cos/sin?
    # Or simply extend the lattice.
    # Let's try a different configuration: a 2x2x2 block?
    # Actually, let's just use a simple grid for now and let the optimizer handle it?
    # No, the positions are fixed in this design.
    # Let's use a compact cluster.
    # (0,0)
    # (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), etc.
    
    # Let's stick to the 7 central ones and add 4 more in a way that minimizes the bounding box.
    # Maybe a 3x3 grid of hexagons?
    # Hexagonal grid points:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in axial coords?
    # Let's use a simple rectangular grid of hexagons, which is a valid packing.
    # But a hexagonal cluster is better.
    
    # Let's use the 7 central, and add 4 more at the "corners" of the hexagon.
    # Positions of layer 2:
    # (2*sqrt(3), 0), (sqrt(3), 3), ...
    # This is getting complicated. Let's just use a 3x3 grid of hexagons.
    # 9 hexagons. Add 2 more?
    # Let's use a linear chain of 11? No.
    
    # Let's use a 4-3-4 arrangement?
    # Row 1: 4 hexagons
    # Row 2: 3 hexagons
    # Row 3: 4 hexagons
    # Total 11.
    
    # Spacing:
    # Horizontal distance: sqrt(3)
    # Vertical distance: 1.5
    # Offset: sqrt(3)/2
    
    pos_list = []
    ang_list = []
    
    # Row 1 (y = 1.5)
    for i in range(4):
        x = (i - 1.5) * np.sqrt(3)
        y = 1.5
        pos_list.append([x, y])
        ang_list.append(0.0)
        
    # Row 2 (y = 0)
    for i in range(3):
        x = (i - 1.0) * np.sqrt(3)
        y = 0.0
        pos_list.append([x, y])
        ang_list.append(0.0)
        
    # Row 3 (y = -1.5)
    for i in range(4):
        x = (i - 1.5) * np.sqrt(3)
        y = -1.5
        pos_list.append([x, y])
        ang_list.append(0.0)
        
    inner_positions = np.array(pos_list)
    inner_angles = np.array(ang_list)
    
    # Optimize outer hexagon
    side, rot_deg = _minimal_enclosing_hexagon(inner_positions, inner_angles)
    
    # The outer hexagon is centered at (0,0) in our calculation.
    # The inner hexagons are also centered around (0,0).
    # So the outer center is (0,0).
    
    inner_rows = []
    for i in range(len(pos_list)):
        inner_rows.append([pos_list[i][0], pos_list[i][1], inner_angles[i]])
        
    return inner_rows, (0.0, 0.0), side, rot_deg

# EVOLVE-BLOCK-END
