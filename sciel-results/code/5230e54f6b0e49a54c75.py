# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 10.0
    
    # Hexagon geometry constants
    d = math.sqrt(3)
    
    # Generate centers
    centers = [(0.0, 0.0)]
    
    # Ring 1: 6 neighbors
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 12 neighbors
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))
                
    if len(ring2_centers) != 12:
        pass
        
    best_S = float('inf')
    best_config = None
    best_theta = 0.0
    best_alpha = 0.0
    
    all_combinations = list(itertools.combinations(range(12), 4))
    
    # Pre-compute vertices for the 7 central hexagons
    # We will generate vertices on the fly for different alphas
    
    # We will iterate through combinations
    for combo in all_combinations:
        if time.time() > deadline - safety_margin:
            break
            
        # Construct the 11 centers
        current_centers = centers[:7]
        for idx in combo:
            current_centers.append(ring2_centers[idx])
            
        # Try a few alpha values (rotation of inner hexagons)
        # 0, 15, 30, 45, 60 degrees
        # Due to symmetry, 0 and 60 are same. 30 is the other extreme.
        # Let's try 0, 15, 30, 45.
        for alpha_deg in [0, 15, 30, 45]:
            if time.time() > deadline - safety_margin:
                break
                
            alpha = math.radians(alpha_deg)
            
            # Generate vertices for this alpha
            verts_rel = []
            for i in range(6):
                angle = i * math.pi / 3 + alpha
                vx = math.cos(angle)
                vy = math.sin(angle)
                verts_rel.append((vx, vy))
                
            # Get all vertices
            points = []
            for cx, cy in current_centers:
                for vx, vy in verts_rel:
                    points.append((cx + vx, cy + vy))
                    
            # Convert to numpy array for faster dot products
            points_np = np.array(points)
            
            # Function to calculate S for a given theta
            def calc_S(theta):
                max_proj = 0.0
                # Normals at theta + k*pi/3
                for k in range(6):
                    angle_n = theta + k * math.pi / 3
                    nx = math.cos(angle_n)
                    ny = math.sin(angle_n)
                    # Dot product with all points
                    dots = np.abs(points_np[:, 0] * nx + points_np[:, 1] * ny)
                    current_max = np.max(dots)
                    if current_max > max_proj:
                        max_proj = current_max
                return (2.0 / math.sqrt(3)) * max_proj
                
            # Coarse search to find approximate minimum theta
            best_theta_coarse = 0.0
            min_S_coarse = float('inf')
            for i in range(100):
                theta = (i / 100.0) * (math.pi / 3)
                S_val = calc_S(theta)
                if S_val < min_S_coarse:
                    min_S_coarse = S_val
                    best_theta_coarse = theta
                    
            # Fine search using scipy minimize_scalar
            window = (math.pi / 3) / 100.0
            lower_bound = max(0.0, best_theta_coarse - window)
            upper_bound = min(math.pi / 3, best_theta_coarse + window)
            
            if lower_bound >= upper_bound:
                lower_bound = 0.0
                upper_bound = math.pi / 3
                
            res = minimize_scalar(calc_S, bounds=(lower_bound, upper_bound), method='bounded')
            
            if res.success:
                min_S_for_combo = res.fun
                min_theta = res.x
            else:
                min_S_for_combo = min_S_coarse
                min_theta = best_theta_coarse
                
            if min_S_for_combo < best_S:
                best_S = min_S_for_combo
                best_config = current_centers
                best_theta = min_theta
                best_alpha = alpha

    # If no valid config found, fallback
    if best_config is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    inner_rows = []
    for cx, cy in best_config:
        inner_rows.append([cx, cy, math.degrees(best_alpha)])
        
    center = (0.0, 0.0)
    side = best_S
    angle_deg = math.degrees(best_theta)
    
    return inner_rows, center, side, angle_deg
# EVOLVE-BLOCK-END
