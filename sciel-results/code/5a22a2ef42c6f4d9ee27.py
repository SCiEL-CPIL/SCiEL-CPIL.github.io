import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    Vertices are in CCW order.
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    """
    Returns squared distance from point (px, py) to the hexagon if outside,
    or 0 if inside.
    Uses the intersection of 3 strips definition for a regular hexagon.
    """
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate point relative to center by -a
    cos_a = math.cos(-a)
    sin_a = math.sin(-a)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    
    apothem = s * math.sqrt(3) / 2
    sum_sq_viol = 0.0
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            sum_sq_viol += (proj - apothem)**2
        if proj < -apothem:
            sum_sq_viol += (-apothem - proj)**2
    return sum_sq_viol

def _get_lattice_positions():
    """
    Returns a list of (x, y, angle) for 11 unit hexagons arranged in a compact cluster.
    We use a standard hexagonal grid layout.
    Center (0,0).
    6 neighbors at distance sqrt(3) (since side=1, distance between centers is sqrt(3)).
    Wait, distance between centers of touching unit hexagons is sqrt(3) * side?
    Side = 1. Apothem = sqrt(3)/2. Distance between centers = 2 * apothem = sqrt(3).
    
    Let's verify:
    Hexagon vertices at distance 1 from center.
    Flat top vs pointy top.
    If vertices are at 0, 60, 120...
    Side length is 1.
    Distance between centers of two touching hexagons (sharing a side) is 2 * apothem = sqrt(3).
    
    Lattice vectors:
    u = (sqrt(3), 0)
    v = (sqrt(3)/2, 1.5)  # 30 deg rotation?
    Actually, let's just use polar coordinates for the rings.
    Ring 0: (0,0)
    Ring 1: 6 hexagons at distance sqrt(3), angles 0, 60, 120, 180, 240, 300.
    Ring 2: We need 4 more.
    In a perfect hex lattice, Ring 2 has 12 hexagons.
    We pick 4 from Ring 2 that are closest to the center of mass or symmetric.
    To make it compact, we pick the ones that "fill in" the gaps or are adjacent to Ring 1.
    Actually, for 11 hexagons, a common optimal packing is the "flower" (7) + 4 more.
    The 4 more are usually placed at the "corners" of the flower to maintain symmetry.
    Let's place them at angles 0, 120, 240, 300? No, that's asymmetric.
    Let's try a symmetric placement:
    The 7 form a hexagon. The next 4 can be placed to form a larger hexagon shape if we had 19.
    With 11, we just add 4.
    Let's pick the 4 positions in Ring 2 that are at angles 30, 90, 150, 210?
    Distance for Ring 2 is 2 * sqrt(3).
    Angles: 0, 60, 120...
    Let's just pick 4 consecutive ones or symmetric ones.
    Symmetric set of 4 in Ring 2:
    Angles: 0, 90, 180, 270? No, hex lattice angles are 0, 60, 120...
    Ring 2 positions are at 0, 60, 120, 180, 240, 300 (outer corners) and 30, 90, 150, 210, 270, 330 (edge midpoints of the larger hex).
    Actually, let's just use the 4 "edge midpoint" positions of the Ring 2 hexagon?
    Or simply, let's use a known good configuration.
    A "compact" 11-hexagon cluster often looks like a 3-4-3 or 4-3-4 row arrangement.
    Let's construct it manually for clarity and robustness.
    
    Row 1 (top): 3 hexagons
    Row 2 (mid): 4 hexagons
    Row 3 (bot): 4 hexagons? No, 3-4-4 = 11.
    Or 4-3-4 = 11.
    
    Let's use the polar ring approach, it's easier to generate.
    Ring 0: (0,0)
    Ring 1: 6 at r=sqrt(3), theta = i*60
    Ring 2: We need 4.
    Let's pick the 4 that are at theta = 30, 90, 150, 210?
    Wait, the hexagonal lattice points in Ring 2 are at:
    r = 2*sqrt(3)
    theta = 0, 60, 120, 180, 240, 300 (Corners)
    AND
    r = sqrt(3) * sqrt(7)? No.
    
    Let's stick to the simplest valid disjoint set.
    The 7 in the center are disjoint.
    The 6 in Ring 1 are disjoint from Center and each other (touching).
    The Ring 2 positions must be disjoint from Ring 1.
    Any position in Ring 2 of the hex lattice is disjoint from Ring 1 (touching Ring 1).
    So we can pick any 4 from the 12 available Ring 2 spots.
    To minimize the bounding box, we should pick ones that are "central" or symmetric.
    Let's pick the 4 at angles 0, 120, 240, 300? That's 3-fold symmetry? No, 0, 120, 240 is 3-fold.
    Let's pick 0, 90, 180, 270? Not in lattice.
    
    Let's just pick 4 consecutive ones: 0, 60, 120, 180.
    This breaks symmetry but is valid.
    Alternatively, pick 30, 150, 270, 330? (Symmetric 2-fold?)
    
    Actually, let's just use a simple grid that is known to be tight.
    Let's use the "flower" (7) + 4 at the "corners" of the bounding box of the flower.
    The flower (7 hexagons) has a bounding hexagon of side 2 (if centered at 0).
    Adding 4 more will increase the size.
    
    Let's generate the 11 positions:
    1. (0, 0)
    2-7. Ring 1: r=sqrt(3), a=i*60
    8-11. Ring 2: r=2*sqrt(3), a=0, 120, 240, 300? 
    Let's check disjointness.
    Ring 2 at 0 deg is at (2*sqrt(3), 0).
    Ring 1 at 0 deg is at (sqrt(3), 0).
    Distance = sqrt(3). They touch. Disjoint (interiors disjoint).
    Ring 2 at 0 deg vs Ring 1 at 60 deg:
    P1 = (2s, 0). P2 = (s/2, s*sqrt(3)/2).
    dx = 1.5s, dy = -s*sqrt(3)/2.
    dist^2 = 2.25 s^2 + 0.75 s^2 = 3 s^2.
    dist = sqrt(3) s. They touch.
    So any Ring 2 point touches 2 Ring 1 points and 1 Ring 2 point (if adjacent).
    So they are disjoint.
    
    So, we will use:
    1 center
    6 in Ring 1
    4 in Ring 2 (at angles 0, 120, 240, 300 for 3-fold-ish symmetry? No, 0, 90, 180, 270 is not lattice. 0, 60, 120, 180 is 4 consecutive. 0, 120, 240, 300 is asymmetric. 0, 60, 180, 240?
    Let's just pick 0, 60, 120, 180. It doesn't matter for the outer optimization, as long as they are fixed.
    """
    positions = []
    # Center
    positions.append((0.0, 0.0, 0.0))
    
    # Ring 1
    r1 = math.sqrt(3)
    for i in range(6):
        a = math.radians(60 * i)
        x = r1 * math.cos(a)
        y = r1 * math.sin(a)
        positions.append((x, y, 0.0))
        
    # Ring 2 (4 more)
    # Let's pick 4 that are roughly symmetric.
    # Angles: 0, 90, 180, 270 are not in the lattice.
    # Lattice angles in Ring 2: 0, 60, 120, 180, 240, 300 (Corners)
    # And the "mid-edge" points of the hexagon formed by Ring 1?
    # Actually, the 12 points in Ring 2 are at:
    # 6 at r=2*sqrt(3), angles 0, 60, ...
    # 6 at r=sqrt(3)*sqrt(3)=3? No.
    # The hex lattice is generated by u=(sqrt(3),0) and v=(sqrt(3)/2, 1.5).
    # Ring 2 points:
    # 2u = (2sqrt(3), 0)
    # 2v = (sqrt(3), 3)
    # u+v = (1.5sqrt(3), 1.5)
    # etc.
    
    # Let's just use the 4 "corner" points of the Ring 2 hexagon for simplicity?
    # No, let's use 4 points that are spread out.
    # Let's use angles 0, 90, 180, 270 approximated by nearest lattice points?
    # No, let's just use 0, 60, 120, 180 from the "corner" set.
    r2 = 2 * math.sqrt(3)
    angles = [0, 60, 120, 180]
    for a_deg in angles:
        a = math.radians(a_deg)
        x = r2 * math.cos(a)
        y = r2 * math.sin(a)
        positions.append((x, y, 0.0))
        
    return positions

def _outer_penalty(inner_positions, outer_params):
    """
    inner_positions: list of (cx, cy, angle)
    outer_params: (cx, cy, s, angle)
    Returns penalty if any inner hexagon is not fully inside the outer hexagon.
    """
    ocx, ocy, os, oang = outer_params
    
    penalty = 0.0
    for ip in inner_positions:
        icx, icy, iang = ip
        # Check vertices of inner hexagon
        v_in = _hex_vertices(icx, icy, 1.0, iang)
        for pt in v_in:
            # Distance to outer hexagon
            penalty += _point_in_hex_dist_sq(pt[0], pt[1], ocx, ocy, os, oang)
            
    return penalty

def _objective(x):
    """
    x: [ocx, ocy, os, oang]
    """
    ocx = x[0]
    ocy = x[1]
    os = x[2]
    oang = x[3]
    outer_params = (ocx, ocy, os, oang)
    
    # Containment penalty
    pen_outer = _outer_penalty(INNER_POSITIONS, outer_params)
    
    # Minimize outer side length
    # If penalty is 0, we want os to be small.
    # If penalty > 0, we want to reduce it.
    # Objective = os + lambda * penalty
    return os + 100.0 * pen_outer

# Pre-compute inner positions
INNER_POSITIONS = _get_lattice_positions()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # Initial guess: Center at (0,0), large side
    # The inner cluster is roughly centered at 0.
    # Bounding box of the 11 hexagons:
    # Max x: 2*sqrt(3) + 1 (radius of hex) ~ 3.46 + 1 = 4.46
    # So side length ~ 4.5 is a safe start.
    x0 = [0.0, 0.0, 5.0, 0.0]
    
    # Bounds
    # Center: [-5, 5]
    # Side: [3.0, 10.0] (Min side for 11 hexagons is likely around 3.5-4.0)
    # Angle: [0, 360]
    bounds = [
        (-5.0, 5.0),
        (-5.0, 5.0),
        (3.0, 10.0),
        (0.0, 360.0)
    ]
    
    best_x = x0
    best_val = _objective(x0)
    
    # Run Differential Evolution
    # This is a 4D problem, very fast.
    try:
        res = differential_evolution(
            _objective, 
            bounds, 
            maxiter=100, 
            popsize=10, 
            seed=42, 
            polish=False,
            tol=1e-6
        )
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline:
        return _format_result(best_x)
        
    # Local optimization (Nelder-Mead) for final polish
    try:
        res2 = minimize(
            _objective, 
            best_x, 
            method='Nelder-Mead', 
            options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        if res2.fun < best_val:
            best_val = res2.fun
            best_x = res2.x
    except Exception:
        pass
        
    return _format_result(best_x)

def _format_result(x):
    inner = []
    for i in range(N):
        cx, cy, a = INNER_POSITIONS[i]
        inner.append([cx, cy, a])
        
    ocx = x[0]
    ocy = x[1]
    os = x[2]
    oang = x[3]
    
    return inner, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)
