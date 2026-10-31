<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit

    # 1. Try multiple hexagonal cluster configurations
    # Each config is a list of axial coordinates (q, r)
    configs = [
        # Config 1: Center + 6 neighbors + 4 more (original)
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2)
        ],
        # Config 2: More radially symmetric
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, -1), (1, 2), (-2, 1), (-1, -2)
        ],
        # Config 3: Compact diamond-like shape
        [
            (0, 0),
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (-1, 1), (1, -1), (-1, -1),
            (2, 0), (0, 2)
        ],
        # Config 4: Hexagonal ring with center
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2)
        ],
        # Config 5: Tighter arrangement
        [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, -1), (1, 1), (-2, 1), (0, 2)
        ],
    ]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # For each config, try to optimize
    for config_idx, axial_coords in enumerate(configs):
        if time.time() > deadline - 40:
            break
        
        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        
        # Convert axial to Cartesian with slight expansion
        epsilon = 1e-4
        scale = 1.0 + epsilon
        
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
            
        initial_inner = [[x, y, 0.0] for x, y in centers]
        
        # Calculate centroid
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        
        # 2. Optimize Outer Hexagon parameters for this config
        def objective_outer_smooth(params, inner=initial_inner):
            cx, cy, angle_deg = params
            return compute_required_side_lse(inner, (cx, cy), angle_deg, temp=0.05)
        
        def objective_outer_exact(params, inner=initial_inner):
            cx, cy, angle_deg = params
            return compute_required_side(inner, (cx, cy), angle_deg)
            
        best_result = None
        best_val_smooth = float('inf')
        
        # Try multiple initial angles
        init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330]
        
        for init_angle in init_angles:
            if time.time() > deadline - 35:
                break
                
            init_params = [cx_avg, cy_avg, float(init_angle)]
            
            try:
                res = minimize(objective_outer_smooth, init_params, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 5000})
                if res.fun < best_val_smooth:
                    best_val_smooth = res.fun
                    best_result = res.x
            except Exception:
                pass
                
        if best_result is None:
            continue
            
        best_outer_center_cfg = (best_result[0], best_result[1])
        best_outer_angle_cfg = best_result[2]
        best_side_cfg = objective_outer_exact(best_result)
        
        # 3. Joint Optimization for this config
        def objective_joint(params, inner_init=initial_inner):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])
            
            ocx = params[N*3]
            ocy = params[N*3+1]
            oangle = params[N*3+2]
            
            side = compute_required_side_lse(inner_list, (ocx, ocy), oangle, temp=0.02)
            penalty = compute_overlap_penalty_smooth(inner_list)
            
            return side + 10.0 * penalty
        
        initial_params = []
        for x, y, a in initial_inner:
            initial_params.extend([x, y, a])
        initial_params.extend([best_outer_center_cfg[0], best_outer_center_cfg[1], best_outer_angle_cfg])
        
        current_params = np.array(initial_params)
        current_val = objective_joint(current_params)
        best_params_cfg = current_params.copy()
        best_val_joint_cfg = current_val
        
        # Adaptive Step Size Local Search with better parameters
        step_size = 0.1
        min_step_size = 0.0005
        decay = 0.985
        max_iter = 3000
        
        iter_count = 0
        while iter_count < max_iter:
            if time.time() > deadline - 20:
                break
                
            # Perturb 2-4 dimensions
            num_perturb = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
            indices = np.random.choice(len(current_params), num_perturb, replace=False)
            
            neighbor = current_params.copy()
            for idx in indices:
                # Different step sizes for position vs angle
                if idx % 3 == 2:  # Angle
                    delta = np.random.normal(0, step_size * 0.5)
                else:  # Position
                    delta = np.random.normal(0, step_size)
                neighbor[idx] += delta
                
            neighbor_val = objective_joint(neighbor)
            
            if neighbor_val < current_val:
                current_params = neighbor
                current_val = neighbor_val
                if current_val < best_val_joint_cfg:
                    best_params_cfg = current_params.copy()
                    best_val_joint_cfg = current_val
            else:
                # Simulated annealing acceptance
                T = 0.005 * max(0.1, 1.0 - iter_count / max_iter)
                if T > 0:
                    delta_val = neighbor_val - current_val
                    if delta_val < 5.0 and np.random.rand() < math.exp(-delta_val / T):
                        current_params = neighbor
                        current_val = neighbor_val
            
            step_size = max(min_step_size, step_size * decay)
            iter_count += 1
            
        # Reconstruct best joint solution for this config
        best_inner_joint_cfg = []
        for i in range(N):
            x = best_params_cfg[i*3]
            y = best_params_cfg[i*3+1]
            a = best_params_cfg[i*3+2]
            best_inner_joint_cfg.append([x, y, a])
        
        best_ocx_cfg = best_params_cfg[N*3]
        best_ocy_cfg = best_params_cfg[N*3+1]
        best_oangle_cfg = best_params_cfg[N*3+2]
        
        # Verify with SAT
        if check_all_disjoint(best_inner_joint_cfg, eps=1e-5):
            # Calculate exact side length
            exact_side = compute_required_side(best_inner_joint_cfg, (best_ocx_cfg, best_ocy_cfg), best_oangle_cfg)
            
            # Update global best if this config is better
            if exact_side < best_side:
                best_side = exact_side
                best_inner = best_inner_joint_cfg
                best_outer_center = (best_ocx_cfg, best_ocy_cfg)
                best_outer_angle = best_oangle_cfg
        else:
            # If joint search broke disjointness, use the rigid body result
            if best_side_cfg < best_side:
                best_side = best_side_cfg
                best_inner = initial_inner
                best_outer_center = best_outer_center_cfg
                best_outer_angle = best_outer_angle_cfg
=======
def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin for 150s limit

    # Generate all 11-cell polyhexes via BFS on axial grid
    def generate_polyhexes(n_cells):
        """Generate all polyhexes of n_cells using BFS."""
        # Directions in axial coordinates
        dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        
        def canonicalize(cells):
            """Return canonical form of a polyhex for deduplication."""
            # Translate so min q is 0, then min r is 0
            min_q = min(q for q, r in cells)
            min_r = min(r for q, r in cells)
            translated = [(q - min_q, r - min_r) for q, r in cells]
            return tuple(sorted(translated))
        
        # Start with single cell
        current = {frozenset([(0, 0)])}
        
        for size in range(1, n_cells):
            next_set = set()
            for cells in current:
                for cell in cells:
                    q, r = cell
                    for dq, dr in dirs:
                        new_cell = (q + dq, r + dr)
                        if new_cell not in cells:
                            new_cells = frozenset(cells | {new_cell})
                            next_set.add(new_cells)
            current = next_set
        
        # Deduplicate using canonical form
        seen = set()
        unique_polyhexes = []
        for cells in current:
            canon = canonicalize(cells)
            if canon not in seen:
                seen.add(canon)
                unique_polyhexes.append(list(canon))
        
        return unique_polyhexes

    # Generate all 11-cell polyhexes
    all_polyhexes = generate_polyhexes(N)
    
    # Score each polyhex by its minimal enclosing hexagon side length
    def score_polyhex(axial_coords):
        """Compute the minimal enclosing hexagon side length for a polyhex."""
        # Convert to Cartesian
        centers = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))
        
        inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        
        # Try a few angles to find best
        best_side = float('inf')
        for angle in [0, 15, 30, 45, 60, 75, 90]:
            side = compute_required_side(inner, (cx_avg, cy_avg), angle)
            if side < best_side:
                best_side = side
        
        return best_side

    # Score all polyhexes and sort
    scored_polyhexes = [(score_polyhex(p), p) for p in all_polyhexes]
    scored_polyhexes.sort(key=lambda x: x[0])
    
    # Take top 20 polyhexes
    top_polyhexes = [p for _, p in scored_polyhexes[:20]]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # For each top polyhex, try to optimize
    for config_idx, axial_coords in enumerate(top_polyhexes):
        if time.time() > deadline - 40:
            break
        
        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        
        # Convert axial to Cartesian with slight expansion
        epsilon = 1e-4
        scale = 1.0 + epsilon
        
        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))
            
        initial_inner = [[x, y, 0.0] for x, y in centers]
        
        # Calculate centroid
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        
        # 2. Optimize Outer Hexagon parameters for this config
        def objective_outer_smooth(params, inner=initial_inner):
            cx, cy, angle_deg = params
            return compute_required_side_lse(inner, (cx, cy), angle_deg, temp=0.05)
        
        def objective_outer_exact(params, inner=initial_inner):
            cx, cy, angle_deg = params
            return compute_required_side(inner, (cx, cy), angle_deg)
            
        best_result = None
        best_val_smooth = float('inf')
        
        # Try multiple initial angles
        init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330]
        
        for init_angle in init_angles:
            if time.time() > deadline - 35:
                break
                
            init_params = [cx_avg, cy_avg, float(init_angle)]
            
            try:
                res = minimize(objective_outer_smooth, init_params, method='Nelder-Mead', 
                              options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 5000})
                if res.fun < best_val_smooth:
                    best_val_smooth = res.fun
                    best_result = res.x
            except Exception:
                pass
                
        if best_result is None:
            continue
            
        best_outer_center_cfg = (best_result[0], best_result[1])
        best_outer_angle_cfg = best_result[2]
        best_side_cfg = objective_outer_exact(best_result)
        
        # 3. Joint Optimization for this config
        def objective_joint(params, inner_init=initial_inner):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])
            
            ocx = params[N*3]
            ocy = params[N*3+1]
            oangle = params[N*3+2]
            
            side = compute_required_side_lse(inner_list, (ocx, ocy), oangle, temp=0.02)
            penalty = compute_overlap_penalty_smooth(inner_list)
            
            return side + 10.0 * penalty
        
        initial_params = []
        for x, y, a in initial_inner:
            initial_params.extend([x, y, a])
        initial_params.extend([best_outer_center_cfg[0], best_outer_center_cfg[1], best_outer_angle_cfg])
        
        current_params = np.array(initial_params)
        current_val = objective_joint(current_params)
        best_params_cfg = current_params.copy()
        best_val_joint_cfg = current_val
        
        # Adaptive Step Size Local Search with better parameters
        step_size = 0.1
        min_step_size = 0.0005
        decay = 0.985
        max_iter = 3000
        
        iter_count = 0
        while iter_count < max_iter:
            if time.time() > deadline - 20:
                break
                
            # Perturb 2-4 dimensions
            num_perturb = np.random.choice([2, 3, 4], p=[0.3, 0.5, 0.2])
            indices = np.random.choice(len(current_params), num_perturb, replace=False)
            
            neighbor = current_params.copy()
            for idx in indices:
                # Different step sizes for position vs angle
                if idx % 3 == 2:  # Angle
                    delta = np.random.normal(0, step_size * 0.5)
                else:  # Position
                    delta = np.random.normal(0, step_size)
                neighbor[idx] += delta
                
            neighbor_val = objective_joint(neighbor)
            
            if neighbor_val < current_val:
                current_params = neighbor
                current_val = neighbor_val
                if current_val < best_val_joint_cfg:
                    best_params_cfg = current_params.copy()
                    best_val_joint_cfg = current_val
            else:
                # Simulated annealing acceptance
                T = 0.005 * max(0.1, 1.0 - iter_count / max_iter)
                if T > 0:
                    delta_val = neighbor_val - current_val
                    if delta_val < 5.0 and np.random.rand() < math.exp(-delta_val / T):
                        current_params = neighbor
                        current_val = neighbor_val
            
            step_size = max(min_step_size, step_size * decay)
            iter_count += 1
            
        # Reconstruct best joint solution for this config
        best_inner_joint_cfg = []
        for i in range(N):
            x = best_params_cfg[i*3]
            y = best_params_cfg[i*3+1]
            a = best_params_cfg[i*3+2]
            best_inner_joint_cfg.append([x, y, a])
        
        best_ocx_cfg = best_params_cfg[N*3]
        best_ocy_cfg = best_params_cfg[N*3+1]
        best_oangle_cfg = best_params_cfg[N*3+2]
        
        # Verify with SAT
        if check_all_disjoint(best_inner_joint_cfg, eps=1e-5):
            # Calculate exact side length
            exact_side = compute_required_side(best_inner_joint_cfg, (best_ocx_cfg, best_ocy_cfg), best_oangle_cfg)
            
            # Update global best if this config is better
            if exact_side < best_side:
                best_side = exact_side
                best_inner = best_inner_joint_cfg
                best_outer_center = (best_ocx_cfg, best_ocy_cfg)
                best_outer_angle = best_oangle_cfg
        else:
            # If joint search broke disjointness, use the rigid body result
            if best_side_cfg < best_side:
                best_side = best_side_cfg
                best_inner = initial_inner
                best_outer_center = best_outer_center_cfg
                best_outer_angle = best_outer_angle_cfg
>>>>>>> REPLACE
