<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Define 11 hexagons on a hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian
    # Use a slightly larger scale to ensure initial non-overlap, 
    # but we will optimize to fit tighter.
    # The "gap" between hexagons in a honeycomb is small.
    # We will initialize the outer ring with a 30 degree rotation to mimic optimal packing.
    
    centers = []
    angles = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))
        
        # Check if it's the center hexagon (0,0)
        if q == 0 and r == 0:
            angles.append(0.0)
        else:
            # Outer ring gets 30 degrees rotation
            # This helps them interlock better
            angles.append(30.0)
            
    # Initial layout: Inner hexagons aligned, outer rotated
    initial_inner = [[x, y, a] for (x, y), a in zip(centers, angles)]
    
    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
=======
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # 1. Define 11 hexagons on a hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian
    # Use a slightly larger scale to ensure initial non-overlap, 
    # but we will optimize to fit tighter.
    # The "gap" between hexagons in a honeycomb is small.
    # We will initialize the outer ring with a 30 degree rotation to mimic optimal packing.
    
    centers = []
    angles = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        centers.append((x, y))
        
        # Check if it's the center hexagon (0,0)
        if q == 0 and r == 0:
            angles.append(0.0)
        else:
            # Outer ring gets 30 degrees rotation
            # This helps them interlock better
            angles.append(30.0)
            
    # Initial layout: Inner hexagons aligned, outer rotated
    initial_inner = [[x, y, a] for (x, y), a in zip(centers, angles)]
    
    # Initialize best valid solution with the initial layout
    initial_outer_center = (0.0, 0.0)
    initial_outer_angle = 0.0
    initial_side = compute_required_side(initial_inner, initial_outer_center, initial_outer_angle)
    
    best_valid_side = initial_side
    best_valid_inner = [list(x) for x in initial_inner]
    best_valid_outer_center = initial_outer_center
    best_valid_outer_angle = initial_outer_angle
    
    def update_best_valid(inner_list, outer_center, outer_angle):
        nonlocal best_valid_side, best_valid_inner, best_valid_outer_center, best_valid_outer_angle
        if check_all_disjoint(inner_list, eps=1e-6):
            side = compute_required_side(inner_list, outer_center, outer_angle)
            if side < best_valid_side:
                best_valid_side = side
                best_valid_inner = [list(x) for x in inner_list]
                best_valid_outer_center = outer_center
                best_valid_outer_angle = outer_angle
>>>>>>> REPLACE

