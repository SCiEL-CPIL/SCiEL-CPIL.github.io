import math
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Define unit hexagon vertices (pointy top)
    # Vertices at angles 30, 90, 150, 210, 270, 330
    angles = [math.radians(30 + 60*k) for k in range(6)]
    unit_verts = [(math.cos(a), math.sin(a)) for a in angles]
    
    # Lattice vectors
    sqrt3 = math.sqrt(3)
    
    # Generate candidate configurations
    # Base 7-cluster
    base_centers = [
        (0.0, 0.0),
        (sqrt3, 0.0),
        (-sqrt3, 0.0),
        (sqrt3/2, 1.5),
        (-sqrt3/2, 1.5),
        (sqrt3/2, -1.5),
        (-sqrt3/2, -1.5)
    ]
    
    # Ring 2 candidates
    ring2 = {
        'A': (2*sqrt3, 0.0),
        'B': (-2*sqrt3, 0.0),
        'C': (sqrt3, 3.0),
        'D': (-sqrt3, -3.0),
        'E': (sqrt3, -3.0),
        'F': (-sqrt3, 3.0),
        'G': (1.5*sqrt3, 1.5),
        'H': (-1.5*sqrt3, -1.5),
        'I': (0.0, -3.0),
        'J': (0.0, 3.0),
        'K': (1.5*sqrt3, -1.5),
        'L': (-1.5*sqrt3, 1.5)
    }
    
    # List of 5-point subsets to try
    # We want to cover the space efficiently.
    subsets = [
        ['J', 'I', 'G', 'H', 'L'],
        ['J', 'I', 'G', 'H', 'K'],
        ['J', 'I', 'A', 'B', 'G'],
        ['J', 'I', 'A', 'B', 'H'],
        ['C', 'D', 'E', 'F', 'A'],
        ['C', 'D', 'E', 'F', 'B'],
        ['G', 'H', 'K', 'L', 'A'],
        ['G', 'H', 'K', 'L', 'B'],
        ['J', 'I', 'C', 'D', 'A'],
        ['J', 'I', 'C', 'D', 'B']
    ]
    
    best_L = float('inf')
    best_config = None
    best_center = (0.0, 0.0)
    
    for subset in subsets:
        if time.time() > deadline:
            break
            
        centers = base_centers + [ring2[key] for key in subset]
        
        # Calculate centroid to center the outer hexagon
        cx = sum(c[0] for c in centers) / N
        cy = sum(c[1] for c in centers) / N
        
        # Calculate required L for this configuration centered at (cx, cy)
        # Shift centers to be relative to (cx, cy)
        rel_centers = [(c[0]-cx, c[1]-cy) for c in centers]
        
        min_L_for_config = 0.0
        
        # Check 6 normal directions of the outer hexagon
        # Normals at 0, 60, 120, 180, 240, 300 degrees
        for k in range(6):
            theta = math.radians(60 * k)
            nx = math.cos(theta)
            ny = math.sin(theta)
            
            # Max dot product of inner hexagon vertices with normal
            # h_inner(theta) = max( v . n ) for v in unit_verts
            # For pointy top, this is periodic.
            # We can precompute or just calc.
            h_inner = max(vx*nx + vy*ny for vx, vy in unit_verts)
            
            # Max dot product of centers
            max_center_dot = max(cx_c*nx + cy_c*ny for cx_c, cy_c in rel_centers)
            
            # Total extent
            extent = max_center_dot + h_inner
            
            # Inradius r = L * sqrt(3) / 2
            # extent <= r
            # L >= extent * 2 / sqrt(3)
            L_needed = extent * 2.0 / sqrt3
            
            if L_needed > min_L_for_config:
                min_L_for_config = L_needed
        
        if min_L_for_config < best_L:
            best_L = min_L_for_config
            best_config = centers
            best_center = (cx, cy)
            
    # If no config found (shouldn't happen), fallback
    if best_config is None:
        best_config = base_centers + [ring2['J'], ring2['I'], ring2['G'], ring2['H'], ring2['L']]
        cx = sum(c[0] for c in best_config) / N
        cy = sum(c[1] for c in best_config) / N
        best_center = (cx, cy)
        # Recalc L roughly
        best_L = 5.0 

    # Construct return value
    # inner: list of [x, y, angle_deg]
    inner = []
    for c in best_config:
        inner.append([c[0], c[1], 0.0])
        
    # outer: center, side, angle
    # We aligned outer hexagon to inner (angle 0)
    return inner, best_center, best_L, 0.0
# EVOLVE-BLOCK-END
