# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Tighter safety margin
    
    SQRT3 = math.sqrt(3)
    
    def get_hex_vertices(cx, cy, angle_deg):
        angles = np.radians(angle_deg) + np.arange(6) * (np.pi / 3.0)
        verts = np.column_stack((np.cos(angles), np.sin(angles)))
        return verts + np.array([cx, cy])

    def get_hex_axes(angle_deg):
        angles = np.radians(angle_deg) + np.pi/6 + np.arange(6) * (np.pi / 3.0)
        axes = np.column_stack((np.cos(angles), np.sin(angles)))
        return axes

    def project_poly(vertices, axes):
        projs = vertices @ axes.T
        return projs.min(axis=0), projs.max(axis=0)

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2):
        min1, max1 = project_poly(hex1_verts, axes1)
        min2, max2 = project_poly(hex2_verts, axes1)
        if np.any(max1 < min2) or np.any(max2 < min1):
            return True
        min1, max1 = project_poly(hex1_verts, axes2)
        min2, max2 = project_poly(hex2_verts, axes2)
        if np.any(max1 < min2) or np.any(max2 < min1):
            return True
        return False

    def calc_outer_side_smooth(hexagons, outer_angle_deg, k=50.0):
        base_angles = np.radians([30, 90, 150, 210, 270, 330])
        axis_angles = base_angles + np.radians(outer_angle_deg)
        axis_dirs = np.column_stack((np.cos(axis_angles), np.sin(axis_angles)))
        
        max_abs_projs = []
        for hex_verts in hexagons:
            projs = hex_verts @ axis_dirs.T
            abs_projs = np.abs(projs)
            max_abs = np.max(abs_projs)
            max_abs_projs.append(max_abs)
            
        # Soft max of the max abs projections across all hexagons?
        # Actually, we need max over all vertices and all axes.
        # Let's collect all max_abs per axis? No, calc_outer_side takes max over all.
        # Let's just compute the raw max and apply soft-max on the set of critical constraints?
        # Simpler: Just use the raw max for now, L-BFGS-B is robust enough for this single kink often.
        # Or, compute soft-max of the absolute projections of all vertices onto all axes? That's 33*6*6 = 1188 values.
        # Let's try a simpler smooth approximation: 
        # S = (1/k) * log( sum_i sum_j exp(k * |proj_ij|) )
        # This is expensive. 
        # Let's stick to the raw max for the outer side, but use smooth penalty for overlaps.
        
        max_proj = 0.0
        for hex_verts in hexagons:
            projs = hex_verts @ axis_dirs.T
            abs_projs = np.abs(projs)
            max_abs = np.max(abs_projs)
            if max_abs > max_proj:
                max_proj = max_abs
        return 2.0 * max_proj / SQRT3

    def calc_smooth_overlap_penalty(hexagons, hex_axes):
        """
        Returns a smooth penalty term for overlaps.
        Penalty = sum over pairs of soft_max(-gaps)
        where gaps are the separations on each SAT axis.
        """
        total_penalty = 0.0
        k = 20.0 # Sharpness of soft-max
        
        for i in range(N):
            for j in range(i+1, N):
                h1, h2 = hexagons[i], hexagons[j]
                a1, a2 = hex_axes[i], hex_axes[j]
                
                # Gaps on axes of h1
                min1, max1 = project_poly(h1, a1)
                min2, max2 = project_poly(h2, a1)
                gaps1 = np.minimum(max1, max2) - np.maximum(min1, min2)
                
                # Gaps on axes of h2
                min1, max1 = project_poly(h1, a2)
                min2, max2 = project_poly(h2, a2)
                gaps2 = np.minimum(max1, max2) - np.maximum(min1, min2)
                
                all_gaps = np.concatenate([gaps1, gaps2])
                
                # If they are disjoint, max(all_gaps) > 0. Penalty should be 0.
                # If they overlap, max(all_gaps) < 0. Penalty should be -max(all_gaps).
                # We want a smooth function f(gaps) that is 0 if max(gaps) > 0 and -max(gaps) if max(gaps) < 0.
                # Actually, the "distance" is max(0, max(gaps)).
                # The "penetration" is max(0, -max(gaps)).
                # We want to penalize penetration.
                # soft_max_gaps = (1/k) * log(sum(exp(k * gaps)))
                # This approximates max(gaps).
                # If max(gaps) is negative, soft_max is negative. We want -soft_max.
                # If max(gaps) is positive, we want 0.
                # So penalty = max(0, -soft_max_gaps).
                # This is still non-smooth at 0.
                # We can use: penalty = (1/k) * log(1 + sum(exp(k * gaps)))? No.
                # Let's just use: penalty = max(0, -soft_max_gaps) and hope L-BFGS-B handles the kink, 
                # or use a smooth ramp: penalty = (soft_max_gaps - soft_max_gaps^2/2) ... no.
                # Let's use a simple smooth approximation: 
                # If we define u = -gaps, we want max(0, soft_max(u)).
                # soft_max(u) is smooth. max(0, x) is not.
                # We can approximate max(0, x) with x/2 + sqrt(x^2 + eps)/2.
                
                soft_max_gaps = (1.0/k) * np.log(np.sum(np.exp(k * all_gaps)))
                # soft_max_gaps approximates max(gaps)
                penetration_approx = -soft_max_gaps
                # Smooth rectification of penetration_approx
                eps = 1e-6
                smooth_penalty = 0.5 * (penetration_approx + np.sqrt(penetration_approx**2 + eps))
                
                total_penalty += smooth_penalty
                
        return total_penalty

    def evaluate(p, outer_angle_deg=0.0):
        centers = p[:2*N].reshape(N, 2)
        angles = p[2*N:]
        
        cx_avg = np.mean(centers[:, 0])
        cy_avg = np.mean(centers[:, 1])
        shifted_centers = centers - np.array([cx_avg, cy_avg])
        
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
        
        overlap_count = 0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                    overlap_count += 1
                    
        if overlap_count > 0:
            return 1000.0 + overlap_count + calc_outer_side_smooth(hexagons, outer_angle_deg)
            
        return calc_outer_side_smooth(hexagons, outer_angle_deg)

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    coords3 = [(0,0), (1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (0,1), (1,1), (2,1), (3,1)]
    
    configs = [coords1, coords2, coords3]
    
    best_params = None
    best_score = float('inf')
    
    for config in configs:
        if time.time() > deadline - 40.0:
            break
            
        params = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        params = np.array(params)
        
        current_score = evaluate(params)
        local_best_params = params.copy()
        local_best_score = current_score
        
        T = 15.0
        T_min = 0.01
        cooling_rate = 0.995
        steps_per_temp = 50
        
        step = 0
        while T > T_min and time.time() < deadline - 20.0:
            new_params = params.copy()
            idx = random.randint(0, N-1)
            
            step_scale = 0.2 + 0.2 * (T / 15.0)
            dx = random.gauss(0, step_scale)
            dy = random.gauss(0, step_scale)
            da = random.gauss(0, 10.0)
            
            new_params[idx*3] += dx
            new_params[idx*3+1] += dy
            new_params[idx*3+2] += da
            
            new_score = evaluate(new_params)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                params = new_params
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_params = params.copy()
            
            step += 1
            if step % steps_per_temp == 0:
                T *= cooling_rate
        
        if local_best_score < best_score:
            best_score = local_best_score
            best_params = local_best_params.copy()
            
    if best_params is None:
        params = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        best_params = np.array(params)

    # Final Polish with L-BFGS-B using smooth penalty
    def objective_lbfgs(p):
        centers = p[:2*N].reshape(N, 2)
        angles = p[2*N:]
        
        cx_avg = np.mean(centers[:, 0])
        cy_avg = np.mean(centers[:, 1])
        shifted_centers = centers - np.array([cx_avg, cy_avg])
        
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
            
        # Smooth overlap penalty
        penalty = calc_smooth_overlap_penalty(hexagons, hex_axes)
        
        # Outer side length (raw max is okay, L-BFGS-B is robust)
        side = calc_outer_side_smooth(hexagons, 0.0)
        
        # We want to minimize side + penalty
        # If penalty is large, it dominates.
        return side + penalty

    try:
        if time.time() < deadline - 10.0:
            res = minimize(objective_lbfgs, best_params, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6})
            
            # Check if the result is valid (no overlaps)
            final_centers = res.x[:2*N].reshape(N, 2)
            final_angles = res.x[2*N:]
            cx_avg = np.mean(final_centers[:, 0])
            cy_avg = np.mean(final_centers[:, 1])
            shifted_centers = final_centers - np.array([cx_avg, cy_avg])
            
            hexagons = []
            hex_axes = []
            for i in range(N):
                cx, cy = shifted_centers[i]
                a = final_angles[i]
                hex_verts = get_hex_vertices(cx, cy, a)
                hexagons.append(hex_verts)
                hex_axes.append(get_hex_axes(a))
            
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Re-evaluate the true score (with optimal outer angle)
                def side_len(oa):
                    return calc_outer_side_smooth(hexagons, oa)
                res_ao = minimize_scalar(side_len, bounds=(0, 60), method='bounded')
                true_score = res_ao.fun
                if true_score < best_score:
                    best_score = true_score
                    best_params = res.x
    except Exception:
        pass

    # Final extraction
    final_params = best_params
    centers = final_params[:2*N].reshape(N, 2)
    angles = final_params[2*N:]
    
    cx_avg = np.mean(centers[:, 0])
    cy_avg = np.mean(centers[:, 1])
    shifted_centers = centers - np.array([cx_avg, cy_avg])
    
    hexagons = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = angles[i]
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
        
    def side_len(oa):
        return calc_outer_side_smooth(hexagons, oa)
        
    res_ao = minimize_scalar(side_len, bounds=(0, 60), method='bounded')
    best_side = res_ao.fun
    best_oa = res_ao.x

    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
