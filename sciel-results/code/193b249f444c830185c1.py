import numpy as np
import time
import random
import math

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def force_directed_pack(W, H, n_circles, time_limit):
        """
        Packs n_circles into a W x H rectangle.
        Returns array of shape (n, 3) with [x, y, r] or None if failed.
        Uses force-directed relaxation with radius growth.
        """
        # 1. Initial layout: Hexagonal grid scaled to fit
        # Estimate spacing
        area_per_circle = (W * H) / n_circles
        init_spacing = math.sqrt(area_per_circle)
        
        # Determine grid dimensions roughly
        cols = max(1, int(np.ceil(n_circles / (H / init_spacing)))) if H > 0 else n_circles
        rows = max(1, int(np.ceil(n_circles / cols)))
        
        # Better init: Just use a grid that covers the area
        # Adjust cols/rows to get close to n_circles
        while cols * rows < n_circles:
            cols += 1
            if cols * rows < n_circles:
                rows += 1
        
        pts = []
        for i in range(rows):
            for j in range(cols):
                if len(pts) >= n_circles:
                    break
                # Hex offset
                x_off = 0.5 * (j % 2)
                x = (j + x_off + 0.5) * W / cols
                y = (i + 0.5) * H / rows
                # Clamp to valid range
                x = np.clip(x, 0.1, W - 0.1)
                y = np.clip(y, 0.1, H - 0.1)
                pts.append([x, y])
        
        # If we have fewer than n_circles (shouldn't happen with the while loop logic above, but just in case)
        while len(pts) < n_circles:
            pts.append([random.uniform(0.5, W-0.5), random.uniform(0.5, H-0.5)])
            
        pos = np.array(pts[:n_circles])
        
        # Initial radii: small
        # Max possible radius is bounded by min(W,H)/2 but also by packing density
        # Start with a fraction of the theoretical max
        max_possible_r = min(W, H) / 4.0
        radii = np.full(n_circles, max_possible_r * 0.5)
        
        # Force-directed relaxation
        # We want to maximize the minimum distance between centers relative to radii sum.
        # Strategy: 
        # 1. Relax positions to remove overlaps for current radii.
        # 2. Increase radii slightly.
        # 3. Repeat.
        
        dt = 0.05
        k_rep = 50.0  # Repulsion strength
        k_wall = 100.0 # Wall repulsion strength
        
        # Pre-allocate forces
        forces = np.zeros_like(pos)
        
        # Adaptive step size
        current_r_scale = 1.0
        target_r_max = min(W, H) / 2.0
        
        # We will iterate for a fixed number of steps or until time limit
        steps_per_iteration = 50
        total_steps = 0
        max_total_steps = 50000 # Cap to avoid infinite loops if stuck
        
        while time.time() < time_limit and total_steps < max_total_steps:
            # --- Phase 1: Relax positions for current radii ---
            for _ in range(steps_per_iteration):
                forces.fill(0.0)
                
                # Pairwise repulsion
                # Only compute for pairs that are close to improve speed? 
                # For N=21, N^2/2 = 210 pairs. 50 steps * 210 pairs is very fast.
                for i in range(n_circles):
                    for j in range(i + 1, n_circles):
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist_sq = dx*dx + dy*dy
                        dist = math.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-9
                        
                        min_dist = radii[i] + radii[j]
                        
                        if dist < min_dist:
                            # Overlap
                            overlap = min_dist - dist
                            # Force is proportional to overlap
                            f_mag = k_rep * overlap
                            fx = f_mag * dx / dist
                            fy = f_mag * dy / dist
                            forces[i, 0] += fx
                            forces[i, 1] += fy
                            forces[j, 0] -= fx
                            forces[j, 1] -= fy
                        else:
                            # Soft repulsion to spread them out slightly
                            # This helps avoid local minima where circles are just touching
                            soft_overlap = min_dist * 0.1 - dist
                            if soft_overlap > 0:
                                f_mag = k_rep * 0.1 * soft_overlap
                                fx = f_mag * dx / dist
                                fy = f_mag * dy / dist
                                forces[i, 0] += fx
                                forces[i, 1] += fy
                                forces[j, 0] -= fx
                                forces[j, 1] -= fy

                # Wall repulsion
                for i in range(n_circles):
                    r = radii[i]
                    x, y = pos[i]
                    
                    # Left
                    if x < r:
                        overlap = r - x
                        forces[i, 0] += k_wall * overlap
                    # Right
                    if x > W - r:
                        overlap = x - (W - r)
                        forces[i, 0] -= k_wall * overlap
                    # Bottom
                    if y < r:
                        overlap = r - y
                        forces[i, 1] += k_wall * overlap
                    # Top
                    if y > H - r:
                        overlap = y - (H - r)
                        forces[i, 1] -= k_wall * overlap

                # Update positions
                pos += dt * forces
                
                # Clamp to hard boundaries (safety)
                for i in range(n_circles):
                    r = radii[i]
                    pos[i, 0] = np.clip(pos[i, 0], r, W - r)
                    pos[i, 1] = np.clip(pos[i, 1], r, H - r)
                
                total_steps += 1
                if time.time() > time_limit:
                    break
            
            # --- Phase 2: Grow radii ---
            # Check if valid (no overlaps, within bounds)
            valid = True
            for i in range(n_circles):
                r = radii[i]
                x, y = pos[i]
                if x < r - 1e-6 or x > W - r + 1e-6 or y < r - 1e-6 or y > H - r + 1e-6:
                    valid = False
                    break
            
            if valid:
                for i in range(n_circles):
                    for j in range(i + 1, n_circles):
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist_sq = dx*dx + dy*dy
                        min_dist = radii[i] + radii[j]
                        if dist_sq < min_dist*min_dist - 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                # Increase radii
                # Scale factor: 1.01
                radii *= 1.01
                # Cap radii
                max_r = min(W, H) / 2.0
                radii = np.clip(radii, 0.0, max_r)
            else:
                # If invalid, shrink slightly and relax more? 
                # Or just relax more with current radii.
                # Let's just relax more.
                pass

        # Final validation and cleanup
        # Ensure strict validity
        for i in range(n_circles):
            r = radii[i]
            x, y = pos[i]
            if x < r: pos[i,0] = r
            if x > W - r: pos[i,0] = W - r
            if y < r: pos[i,1] = r
            if y > H - r: pos[i,1] = H - r
            
        # Check overlaps one last time. If overlap, shrink radii until valid.
        # This is a fallback to ensure we return a valid solution.
        for _ in range(100):
            valid = True
            for i in range(n_circles):
                r = radii[i]
                x, y = pos[i]
                if x < r - 1e-9 or x > W - r + 1e-9 or y < r - 1e-9 or y > H - r + 1e-9:
                    valid = False
                    break
            if valid:
                for i in range(n_circles):
                    for j in range(i + 1, n_circles):
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        dist_sq = dx*dx + dy*dy
                        min_dist = radii[i] + radii[j]
                        if dist_sq < min_dist*min_dist - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                break
            # Shrink radii to resolve violation
            radii *= 0.95
        
        result = np.zeros((n_circles, 3))
        result[:, 0] = pos[:, 0]
        result[:, 1] = pos[:, 1]
        result[:, 2] = radii
        return result

    def is_valid(circles, W, H, tol=1e-5):
        """Check validity."""
        if not circles:
            return False
        for x, y, r in circles:
            if x < r - tol or x > W - r + tol or y < r - tol or y > H - r + tol:
                return False
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist_sq = dx*dx + dy*dy
                min_dist = circles[i][2] + circles[j][2]
                if dist_sq < min_dist*min_dist - tol:
                    return False
        return True

    best_sum_r = 0.0
    best_config = None

    # Sweep aspect ratios
    # W + H = 2. 
    # Symmetric around W=1. 
    # Try a range of W values.
    # The optimal aspect ratio for packing circles in a rectangle is often close to 1:1 or specific ratios.
    # Let's sample W from 0.6 to 1.4.
    
    W_values = np.linspace(0.6, 1.4, 9)
    
    # Sort by distance from 1.0 to try square first (often good for dense packing)
    W_values = sorted(W_values, key=lambda w: abs(w - 1.0))
    
    for W in W_values:
        if time.time() > deadline - 5:
            break
            
        H = 2.0 - W
        if H <= 0.1 or W <= 0.1:
            continue
            
        # Run force-directed packing
        # Give it a time slice. 
        # Total time 145s. 9 values. ~15s each.
        slice_time = min(15.0, deadline - 5 - time.time())
        if slice_time <= 1:
            break
            
        res = force_directed_pack(W, H, N, time.time() + slice_time)
        
        if res is not None:
            # Validate
            if is_valid(res, W, H):
                s = np.sum(res[:, 2])
                if s > best_sum_r:
                    best_sum_r = s
                    best_config = res.copy()
            else:
                # If invalid due to tiny numerical issues, try to fix by shrinking
                # But force_directed_pack already has a shrink loop.
                # If it's still invalid, skip.
                pass

    # If we found a good config, return it
    if best_config is not None:
        return best_config

    # Fallback: Simple grid if everything failed
    side = 5
    cells = []
    W_fallback = 1.0
    H_fallback = 1.0
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side * 0.8
            cells.append((x, y, r))
    return np.array(cells[:N])

# EVOLVE-BLOCK-END
