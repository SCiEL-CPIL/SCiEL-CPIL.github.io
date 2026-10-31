<<<<<<< SEARCH
def generate_hexagonal_symmetric_init():
    """Generate hexagonal lattice points, then fit to symmetric structure."""
    best_params = None
    best_val = -np.inf
    
    for scale in [0.18, 0.2, 0.22, 0.25, 0.28, 0.3]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            # Try to select 11 points and fit to symmetric structure
            rng = np.random.default_rng(42)
            for trial in range(20):
                idx = rng.choice(len(pts), N, replace=False)
                selected = np.array(pts)[idx]
                
                # Find best symmetric fit: center = mean, orbits from remaining
                center_est = np.mean(selected, axis=0)
                center_est = np.clip(center_est, 0.01, 0.99)
                
                # Assign remaining 10 points to 3 orbits (3+3+3+1 leftover, or 4+3+3)
                # Simple approach: cluster by angle from center
                remaining = selected.copy()
                remaining[0] = center_est  # Mark center
                
                # Use 3 orbits of 3: pick 9 points, assign to orbits
                orbit_pts = []
                for orbit_i in range(3):
                    # Pick 3 points roughly 120° apart from center
                    angles = rng.uniform(0, 2*np.pi, 3)
                    orbit_rep = center_est + 0.2 * np.array([np.cos(angles[0]), np.sin(angles[0]) * 0.577])
                    orbit_rep = np.clip(orbit_rep, 0.01, 0.99)
                    orbit_pts.append(orbit_rep)
                
                params = np.concatenate([center_est, orbit_pts[0], orbit_pts[1], orbit_pts[2]])
                bary = build_symmetric_bary(params)
                val = -hard_min_obj_bary(bary)
                
                if val > best_val:
                    best_val = val
                    best_params = params.copy()
    
    if best_params is None:
        return generate_symmetric_init(42)
        
    return best_params
=======
def generate_hexagonal_symmetric_init():
    """Generate hexagonal lattice points, then fit to symmetric structure."""
    best_params = None
    best_val = -np.inf
    
    for scale in [0.18, 0.2, 0.22, 0.25, 0.28, 0.3]:
        pts = []
        max_k = int(1.0 / scale) + 1
        for k in range(max_k + 1):
            w_val = k * scale
            if w_val > 1.0:
                break
            num_pts_in_row = int((1 - w_val) / scale) + 1
            for j in range(num_pts_in_row):
                u_val = j * scale
                v_val = (1 - w_val) - u_val
                if u_val >= -1e-9 and v_val >= -1e-9 and u_val + v_val <= 1 + 1e-9:
                    pts.append([u_val, v_val])
        
        if len(pts) >= N:
            pts_arr = np.array(pts)
            rng = np.random.default_rng(42)
            for trial in range(20):
                idx = rng.choice(len(pts_arr), N, replace=False)
                selected = pts_arr[idx]
                
                # Find best symmetric fit: center = mean, orbits from remaining
                center_est = np.mean(selected, axis=0)
                center_est = np.clip(center_est, 0.01, 0.99)
                
                # Use 3 orbits of 3: pick 9 points, assign to orbits
                orbit_pts = []
                for orbit_i in range(3):
                    # Pick a representative point from the selected set, slightly perturbed
                    # to ensure distinct orbits if needed, or just use random points
                    rep_idx = rng.integers(0, N)
                    orbit_rep = selected[rep_idx].copy()
                    # Add small noise to break perfect symmetry if needed, or just use as is
                    orbit_rep = np.clip(orbit_rep, 0.01, 0.99)
                    orbit_pts.append(orbit_rep)
                
                # Ensure we have exactly 8 parameters (2 for center + 3*2 for orbits)
                params = np.concatenate([center_est, orbit_pts[0], orbit_pts[1], orbit_pts[2]])
                bary = build_symmetric_bary(params)
                val = -hard_min_obj_bary(bary)
                
                if val > best_val:
                    best_val = val
                    best_params = params.copy()
    
    if best_params is None:
        return generate_symmetric_init(42)
        
    return best_params
>>>>>>> REPLACE

