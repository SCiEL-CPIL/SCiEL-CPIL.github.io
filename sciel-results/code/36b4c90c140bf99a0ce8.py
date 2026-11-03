"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import differential_evolution

N = 12

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    theta = np.deg2rad(angle_deg)
    angles = theta + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform point to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    theta = -np.deg2rad(angle_deg)
    lx = dx * np.cos(theta) - dy * np.sin(theta)
    ly = dx * np.sin(theta) + dy * np.cos(theta)
    
    # In local coords, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon can be described by |x| <= side, |y| <= side*sqrt(3)/2, |x ± y/sqrt(3)| <= side
    # More precisely, for a regular hexagon with "flat" top (vertex at angle 0), 
    # the constraints are based on the three pairs of parallel sides.
    
    # For a hexagon with vertices at angles 0, 60, ..., 300:
    # Side 1-2 (top-right): line from (side, 0) to (side/2, side*sqrt(3)/2)
    # The three pairs of opposite sides define:
    # |x| <= side
    # |y| <= side * sqrt(3) / 2  
    # |x + y/sqrt(3)| <= side  ... wait, let me think more carefully.
    
    # Actually, for a regular hexagon with a vertex at angle 0 (pointing right):
    # Vertices: (s,0), (s/2, s√3/2), (-s/2, s√3/2), (-s, 0), (-s/2, -s√3/2), (s/2, -s√3/2)
    # The six sides are:
    # 1. From (s,0) to (s/2, s√3/2): line x + y/√3 = s
    # 2. From (s/2, s√3/2) to (-s/2, s√3/2): line y = s√3/2
    # 3. From (-s/2, s√3/2) to (-s, 0): line -x + y/√3 = s
    # 4. From (-s, 0) to (-s/2, -s√3/2): line x + y/√3 = -s
    # 5. From (-s/2, -s√3/2) to (s/2, -s√3/2): line y = -s√3/2
    # 6. From (s/2, -s√3/2) to (s, 0): line -x + y/√3 = -s
    
    # So constraints:
    # x + y/√3 <= s,  x + y/√3 >= -s
    # y <= s√3/2,     y >= -s√3/2
    # -x + y/√3 <= s, -x + y/√3 >= -s
    
    sqrt3 = np.sqrt(3.0)
    val1 = lx + ly / sqrt3
    val2 = ly
    val3 = -lx + ly / sqrt3
    
    return (abs(val1) <= side and abs(val2) <= side * sqrt3 / 2 and abs(val3) <= side)

def hexagon_distance_to_center(cx, cy, side, angle_deg):
    """Maximum distance from center to any point in the hexagon (circumradius for regular hexagon is side)."""
    return side

def construct():
    # Strategy: Place 12 unit hexagons in a honeycomb-like pattern inside a larger hexagon.
    # We'll use optimization to find the minimum enclosing hexagon.
    
    # Initial guess: arrange in a hexagonal grid pattern
    # For 12 hexagons, a good arrangement is a central hexagon surrounded by others,
    # or a rectangular-ish arrangement.
    
    # Let's try a hexagonal packing: place them in a pattern that minimizes the bounding hexagon.
    # A hexagon of hexagons with side k has 3k(k-1)+1 hexagons.
    # k=2: 7 hexagons, k=3: 19 hexagons. So 12 doesn't fit a perfect hexagonal number.
    
    # Let's try placing them in a 4x3 grid with hexagonal offsets, then optimize.
    
    # Hexagonal grid coordinates:
    # Use axial coordinates. For hexagons of side 1, the distance between centers in a honeycomb
    # is sqrt(3) for adjacent hexagons (edge-to-edge distance is 1, center-to-center is sqrt(3)).
    
    # Actually, for regular hexagons of side 1 packed edge-to-edge, the center-to-center distance
    # between adjacent hexagons is sqrt(3) * 1 = sqrt(3) ≈ 1.732.
    
    # Let's set up an initial configuration and use differential evolution to optimize.
    
    def objective(params):
        # params: 12 * 3 + 3 = 39 parameters
        # First 36: (x_i, y_i, angle_i) for i in 0..11
        # Last 3: (outer_cx, outer_cy, outer_side)
        # Actually, we want to minimize outer_side, so let's parameterize differently.
        # We'll fix the outer hexagon center at origin and optimize its side and the inner positions.
        
        inner_params = params[:36]
        outer_side = params[36]
        
        # Check if outer_side is reasonable
        if outer_side < 1.0:
            return 100.0
        
        # Check all inner hexagons are inside the outer hexagon
        outer_cx, outer_cy, outer_angle = 0.0, 0.0, 0.0
        
        for i in range(12):
            ix = inner_params[3*i]
            iy = inner_params[3*i+1]
            iangle = inner_params[3*i+2]
            
            # Check all vertices of inner hexagon are inside outer hexagon
            verts = hex_vertices(ix, iy, 1.0, iangle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                    return outer_side + 10.0 * (1.0 / (outer_side - 1.0 + 1e-6))
        
        # Check no overlap between inner hexagons
        for i in range(12):
            for j in range(i+1, 12):
                ix1 = inner_params[3*i]
                iy1 = inner_params[3*i+1]
                ia1 = inner_params[3*i+2]
                ix2 = inner_params[3*j]
                iy2 = inner_params[3*j+1]
                ia2 = inner_params[3*j+2]
                
                # Check if hexagons overlap by checking if any vertex of one is inside the other
                # This is a sufficient but not necessary condition. For regular hexagons,
                # we can use a simpler check: if center-to-center distance < 2*side*something
                # Actually, for non-overlapping hexagons of side 1, the minimum center distance
                # depends on relative orientation. A safe check: if distance < 2*sqrt(3)*1/2... 
                # Let's use vertex containment check.
                
                verts1 = hex_vertices(ix1, iy1, 1.0, ia1)
                verts2 = hex_vertices(ix2, iy2, 1.0, ia2)
                
                # Check if any vertex of hex1 is inside hex2 or vice versa
                for vx, vy in verts1:
                    if point_in_hexagon(vx, vy, ix2, iy2, 1.0, ia2):
                        return outer_side + 10.0
                for vx, vy in verts2:
                    if point_in_hexagon(vx, vy, ix1, iy1, 1.0, ia1):
                        return outer_side + 10.0
        
        return outer_side
    
    # Initial guess: hexagonal arrangement
    # Place 12 hexagons in a roughly hexagonal pattern
    sqrt3 = np.sqrt(3.0)
    
    # Axial coordinates for a hexagonal arrangement
    # Let's try: rows of 3,4,3,2 or similar
    # Better: use a known good initial configuration
    
    # Start with a compact arrangement
    # Center at (0,0), place hexagons in a pattern
    positions = []
    angles = []
    
    # Hexagonal grid: use offset coordinates
    # For a honeycomb, adjacent centers are at distance sqrt(3)
    # Let's place them in a 4x3 grid with hexagonal offset
    
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    
    # Actually, let's just use a simple initial guess and let DE do the work.
    # Start with all hexagons at origin with small perturbations
    
    np.random.seed(42)
    
    # Better initial guess: hexagonal packing
    # Place in a hexagonal pattern
    hex_positions = []
    # Central hexagon at (0,0)
    hex_positions.append((0.0, 0.0))
    # 6 surrounding
    for k in range(6):
        ang = k * np.pi / 3.0
        hex_positions.append((sqrt3 * np.cos(ang), sqrt3 * np.sin(ang)))
    # 5 more to make 12
    for k in range(5):
        ang = k * np.pi / 3.0 + np.pi / 6.0
        hex_positions.append((2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang)))
    
    # All angles 0
    init_params = []
    for (hx, hy) in hex_positions:
        init_params.extend([hx, hy, 0.0])
    init_params.append(6.0)  # initial outer side guess
    
    # Bounds
    bounds = []
    for i in range(12):
        bounds.append((-10, 10))  # x
        bounds.append((-10, 10))  # y
        bounds.append((0, 180))   # angle
    bounds.append((1.0, 20.0))    # outer side
    
    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=100,
        popsize=15,
        tol=1e-4,
        init='latinhypercube',
        polish=False
    )
    
    best_params = result.x
    best_outer_side = best_params[36]
    
    # Extract inner hexagons
    inner = []
    for i in range(12):
        ix = best_params[3*i]
        iy = best_params[3*i+1]
        iangle = best_params[3*i+2]
        inner.append([ix, iy, iangle])
    
    return inner, (0.0, 0.0), best_outer_side, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
