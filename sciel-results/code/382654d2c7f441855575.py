"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    best_sum_r = 0.0
    best_config = None
    best_W = 1.0
    best_H = 1.0
    
    def is_valid(circles, W, H, tol=1e-6):
        """Check if packing is valid."""
        n = len(circles)
        for i in range(n):
            xi, yi, ri = circles[i]
            if xi < ri - tol or xi > W - ri + tol:
                return False
            if yi < ri - tol or yi > H - ri + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist_sq = dx*dx + dy*dy
                min_dist = circles[i][2] + circles[j][2]
                if dist_sq < min_dist*min_dist - tol*min_dist:
                    return False
        return True

    def optimize_packing(circles_init, W, H, time_limit):
        """
        Topological graph model optimization.
        1. Identify contact graph (pairs where dist < r_i + r_j + margin).
        2. Apply forces:
           - Spring force for contact edges to enforce d = r_i + r_j.
           - Wall forces to keep circles inside.
           - Global expansion: try to increase all radii slightly.
        3. Iterate until convergence or time limit.
        """
        xs = np.array([c[0] for c in circles_init], dtype=np.float64)
        ys = np.array([c[1] for c in circles_init], dtype=np.float64)
        rs = np.array([c[2] for c in circles_init], dtype=np.float64)
        n = len(xs)
        
        # We will optimize positions and radii.
        # Strategy: 
        # - Maintain a contact graph.
        # - If circles overlap, push apart (repulsion).
        # - If circles are in contact (dist approx r_i+r_j), treat as rigid bar.
        # - To maximize sum(r), we need to "inflate" the packing.
        #   We can do this by adding a small attractive force between all pairs 
        #   (to pull them closer) while maintaining non-overlap, effectively 
        #   compressing the empty space and allowing radii to grow if we 
        #   re-normalize? No, better:
        #   
        #   Alternative: 
        #   Fix positions, solve for max radii? No, positions matter.
        #   
        #   Let's use a "pressure" approach.
        #   Force 1: Repulsion if overlap.
        #   Force 2: Attraction if not overlapping but close (to find tight packing).
        #   Force 3: Wall constraints.
        #   After relaxation, we check if we can uniformly scale radii up 
        #   while maintaining validity.
        
        # Better approach for this specific task:
        # 1. Start with a valid packing.
        # 2. Iteratively:
        #    a. Slightly increase all radii by factor (1+eps).
        #    b. Relax positions to remove overlaps (force-directed).
        #    c. If valid, keep. Else, revert radii.
        
        current_sum_r = np.sum(rs)
        
        # Pre-allocate force arrays
        fx = np.zeros(n)
        fy = np.zeros(n)
        
        # Contact graph management
        # We will re-evaluate contacts every few steps to save time
        
        step = 0
        last_contact_update = 0
        contact_pairs = []
        
        def update_contacts():
            nonlocal contact_pairs
            contact_pairs = []
            for i in range(n):
                for j in range(i+1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    min_dist = rs[i] + rs[j]
                    # Consider contact if close
                    if dist_sq < (min_dist * 1.05)**2:
                        contact_pairs.append((i, j))
                        
        update_contacts()
        
        lr = 0.05
        growth_rate = 0.002
        max_iterations = 5000
        
        for it in range(max_iterations):
            if time.time() > time_limit:
                break
            
            # 1. Try to grow radii
            new_rs = rs * (1.0 + growth_rate)
            
            # 2. Calculate forces with new_rs to see if we can resolve overlaps
            # We simulate with new_rs
            fx_sim = np.zeros(n)
            fy_sim = np.zeros(n)
            
            # Update contacts if needed
            if it % 50 == 0:
                # We need to check contacts with new_rs potentially
                # For simplicity, use current xs, ys and new_rs to find overlaps/contacts
                # But contact graph should be based on topology, not just current dist
                # Let's just use all pairs for force calculation if n is small (21 is small)
                pass
                
            # Pairwise forces
            for i in range(n):
                for j in range(i+1, n):
                    dx = xs[i] - xs[j]
                    dy = ys[i] - ys[j]
                    dist_sq = dx*dx + dy*dy
                    dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-9
                    min_dist = new_rs[i] + new_rs[j]
                    
                    if dist < min_dist:
                        # Overlap: Strong repulsion
                        overlap = min_dist - dist
                        if dist > 1e-6:
                            nx = dx / dist
                            ny = dy / dist
                            fx_sim[i] += nx * overlap
                            fy_sim[i] += ny * overlap
                            fx_sim[j] -= nx * overlap
                            fy_sim[j] -= ny * overlap
                        else:
                            angle = np.random.uniform(0, 2*np.pi)
                            fx_sim[i] += np.cos(angle) * overlap
                            fy_sim[i] += np.sin(angle) * overlap
                            fx_sim[j] -= np.cos(angle) * overlap
                            fy_sim[j] -= np.sin(angle) * overlap
                    else:
                        # No overlap: Weak attraction to encourage tight packing
                        # This helps the system find the "taut" state
                        # Attraction only if reasonably close
                        if dist < min_dist * 2.0:
                            attr = 0.01 * (min_dist - dist) # Linear attraction
                            if dist > 1e-6:
                                nx = dx / dist
                                ny = dy / dist
                                fx_sim[i] += nx * attr
                                fy_sim[i] += ny * attr
                                fx_sim[j] -= nx * attr
                                fy_sim[j] -= ny * attr
            
            # Wall constraints (with new_rs)
            for i in range(n):
                if xs[i] < new_rs[i]:
                    fx_sim[i] += (new_rs[i] - xs[i]) * 5.0
                if xs[i] > W - new_rs[i]:
                    fx_sim[i] -= (xs[i] - (W - new_rs[i])) * 5.0
                if ys[i] < new_rs[i]:
                    fy_sim[i] += (new_rs[i] - ys[i]) * 5.0
                if ys[i] > H - new_rs[i]:
                    fy_sim[i] -= (ys[i] - (H - new_rs[i])) * 5.0
            
            # Update positions
            xs_new = xs + lr * fx_sim
            ys_new = ys + lr * fy_sim
            
            # Project to bounds (hard constraint)
            xs_new = np.clip(xs_new, new_rs, W - new_rs)
            ys_new = np.clip(ys_new, new_rs, H - new_rs)
            
            # Check validity of new configuration
            valid = True
            for i in range(n):
                if xs_new[i] < new_rs[i] - 1e-5 or xs_new[i] > W - new_rs[i] + 1e-5:
                    valid = False
                    break
                if ys_new[i] < new_rs[i] - 1e-5 or ys_new[i] > H - new_rs[i] + 1e-5:
                    valid = False
                    break
            
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        dx = xs_new[i] - xs_new[j]
                        dy = ys_new[i] - ys_new[j]
                        dist_sq = dx*dx + dy*dy
                        min_dist = new_rs[i] + new_rs[j]
                        if dist_sq < min_dist*min_dist - 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                xs = xs_new
                ys = ys_new
                rs = new_rs
                # Reduce growth rate slightly as we converge?
                # growth_rate = max(0.0001, growth_rate * 0.99)
            else:
                # Revert radii, but maybe keep some position changes?
                # To be safe, revert positions too if growth failed
                # Or try smaller growth
                # For now, revert growth, keep positions (they might be better for next try)
                # Actually, if growth failed, positions were updated to accommodate growth.
                # If we revert radii, the old positions might be valid.
                # Let's just revert radii.
                pass # rs remains old
                # Try to relax positions without growing to settle
                # Use old rs for forces
                fx_settle = np.zeros(n)
                fy_settle = np.zeros(n)
                for i in range(n):
                    for j in range(i+1, n):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist_sq = dx*dx + dy*dy
                        dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-9
                        min_dist = rs[i] + rs[j]
                        if dist < min_dist:
                            overlap = min_dist - dist
                            if dist > 1e-6:
                                nx = dx / dist
                                ny = dy / dist
                                fx_settle[i] += nx * overlap
                                fy_settle[i] += ny * overlap
                                fx_settle[j] -= nx * overlap
                                fy_settle[j] -= ny * overlap
                        else:
                            if dist < min_dist * 2.0:
                                attr = 0.01 * (min_dist - dist)
                                if dist > 1e-6:
                                    nx = dx / dist
                                    ny = dy / dist
                                    fx_settle[i] += nx * attr
                                    fy_settle[i] += ny * attr
                                    fx_settle[j] -= nx * attr
                                    fy_settle[j] -= ny * attr
                
                for i in range(n):
                    if xs[i] < rs[i]:
                        fx_settle[i] += (rs[i] - xs[i]) * 5.0
                    if xs[i] > W - rs[i]:
                        fx_settle[i] -= (xs[i] - (W - rs[i])) * 5.0
                    if ys[i] < rs[i]:
                        fy_settle[i] += (rs[i] - ys[i]) * 5.0
                    if ys[i] > H - rs[i]:
                        fy_settle[i] -= (ys[i] - (H - rs[i])) * 5.0
                
                xs = np.clip(xs + lr * fx_settle, rs, W - rs)
                ys = np.clip(ys + lr * fy_settle, rs, H - rs)
            
            # Check if sum_r improved
            current_sum_r = np.sum(rs)
        
        return list(zip(xs, ys, rs))

    def generate_hexagonal_init(W, H, n_circles):
        """
        Generate a hexagonal packing initialization.
        """
        circles = []
        # Estimate radius
        # Area per circle in hex packing is 2*sqrt(3)*r^2
        # Total Area = W*H
        # n * 2*sqrt(3)*r^2 <= W*H
        # r <= sqrt(W*H / (n * 2*sqrt(3)))
        r_est = np.sqrt(W*H / (n_circles * 2 * np.sqrt(3)))
        
        # Hex grid parameters
        # dx = 2*r, dy = sqrt(3)*r
        # Adjust for aspect ratio? 
        # Let's just place on a hex grid centered in the box.
        
        # Determine grid size
        # We need roughly n_circles circles.
        # Try different densities
        best_circles = None
        best_area = 0
        
        for scale in [0.9, 0.95, 1.0, 1.05, 1.1]:
            r = r_est * scale
            dx = 2 * r
            dy = np.sqrt(3) * r
            
            # Generate points
            pts = []
            # Center the grid
            # Number of rows/cols
            # Approx cols = W / dx, rows = H / dy
            n_cols = int(W / dx) + 2
            n_rows = int(H / dy) + 2
            
            # Shift to center
            offset_x = (W - (n_cols-1)*dx) / 2.0
            offset_y = (H - (n_rows-1)*dy) / 2.0
            
            for row in range(n_rows):
                for col in range(n_cols):
                    x = offset_x + col * dx
                    if row % 2 == 1:
                        x += dx / 2.0
                    y = offset_y + row * dy
                    pts.append((x, y))
            
            # Select n_circles closest to center? Or just take first N?
            # Taking first N might bias to corner.
            # Let's take N closest to the center of the rectangle (W/2, H/2)
            pts_arr = np.array(pts)
            center = np.array([W/2, H/2])
            dists = np.sum((pts_arr - center)**2, axis=1)
            idx = np.argsort(dists)[:n_circles]
            selected_pts = pts_arr[idx]
            
            # Check if they fit
            valid = True
            for i in range(len(selected_pts)):
                x, y = selected_pts[i]
                if x < r or x > W - r or y < r or y > H - r:
                    valid = False
                    break
                if not valid:
                    break
            
            if valid:
                # Check overlaps (should be fine for hex grid)
                # Sum of radii
                sum_r = n_circles * r
                if sum_r > best_area:
                    best_area = sum_r
                    best_circles = [(x, y, r) for x, y in selected_pts]
        
        if best_circles is None:
            # Fallback: simple grid
            r = min(W, H) / 6.0
            circles = []
            for i in range(n_circles):
                # Spiral out?
                x = W/2
                y = H/2
                # Just place randomly for now?
                # Better: Grid
                pass
            # Simple grid fallback
            side = int(np.ceil(np.sqrt(n_circles)))
            r = min(W/side, H/side) * 0.45
            for i in range(n_circles):
                cx = i % side
                cy = i // side
                x = (cx + 0.5) * W / side
                y = (cy + 0.5) * H / side
                circles.append((x, y, r))
            return circles
            
        return best_circles

    # Main loop
    np.random.seed(42)
    
    aspect_ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    num_restarts = 8
    
    for W_ratio in aspect_ratios:
        if time.time() > deadline - 30:
            break
        
        W = W_ratio
        H = 2.0 - W_ratio
        if W <= 0 or H <= 0:
            continue
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Jitter the initial aspect ratio slightly?
            # Or just use the fixed W, H for this restart
            
            # Generate init
            init_circles = generate_hexagonal_init(W, H, N)
            
            if len(init_circles) < N:
                continue
                
            # Optimize
            time_limit_for_opt = deadline - 5
            optimized = optimize_packing(init_circles, W, H, time_limit_for_opt)
            
            if not is_valid(optimized, W, H):
                continue
            
            total_r = sum(c[2] for c in optimized)
            if total_r > best_sum_r:
                best_sum_r = total_r
                best_config = np.array(optimized)
                best_W = W
                best_H = H
    
    if best_config is not None:
        return best_config
    
    # Fallback: Grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
