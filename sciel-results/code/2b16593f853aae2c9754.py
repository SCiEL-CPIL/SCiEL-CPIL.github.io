import numpy as np
import time
import math
from scipy.spatial import cKDTree

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_radius(centers):
        """Compute the maximum radius for given centers."""
        wall_dists = 0.5 - np.abs(centers)
        min_wall = np.min(wall_dists)
        
        if len(centers) < 2:
            return min_wall
            
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1])
        
        r = min(min_wall, min_pair / 2.0)
        return max(0.0, r)

    def hexagonal_initialization(n_rows, n_cols):
        """Create a hexagonal packing layout."""
        d = 1.0 / (n_cols + 0.5)
        row_spacing = d * math.sqrt(3) / 2
        
        total_height = (n_rows - 1) * row_spacing + d
        if total_height > 1.0 + 1e-6:
            return None
        
        pts = []
        for row in range(n_rows):
            y = -0.5 + d/2 + row * row_spacing
            if row % 2 == 0:
                n_in_row = n_cols
                x_start = -0.5 + d/2
            else:
                n_in_row = n_cols - 1
                x_start = -0.5 + d
            
            for col in range(n_in_row):
                x = x_start + col * d
                pts.append([x, y])
        
        pts = np.array(pts)
        if len(pts) >= N:
            pts = pts[:N]
        elif len(pts) < N:
            return None
            
        return pts

    def max_min_expansion(centers, max_time):
        """
        Iterative max-min expansion algorithm.
        At each step, find the bottleneck (min wall or min pair).
        Move the involved centers to maximize the clearance of that specific constraint.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        # Parameters
        step_scale = 0.5 # Fraction of the ideal move to take (damping)
        min_step_scale = 0.05
        
        while time.time() < max_time:
            # 1. Identify Bottleneck
            wall_dists = 0.5 - np.abs(centers)
            min_wall_idx = np.argmin(wall_dists)
            min_wall_val = wall_dists[min_wall_idx]
            
            tree = cKDTree(centers)
            dists, idxs = tree.query(centers, k=2)
            min_pair_idx_i = np.argmin(dists[:, 1])
            min_pair_idx_j = idxs[min_pair_idx_i][1]
            min_pair_dist = dists[min_pair_idx_i][1]
            min_pair_val = min_pair_dist / 2.0
            
            current_r = min(min_wall_val, min_pair_val)
            
            if current_r > best_r:
                best_r = current_r
                best_centers = centers.copy()
                
            # 2. Determine Move Strategy
            # We want to increase the minimum of (wall_dists, pair_dists/2)
            # If wall is bottleneck, move that point away from wall.
            # If pair is bottleneck, move the two points away from each other.
            
            # To avoid oscillation, we alternate or prioritize based on which is smaller.
            # Actually, the standard approach is to move the point(s) that are limiting the radius.
            # If min_wall < min_pair/2, the wall constraint is active.
            # If min_pair/2 < min_wall, the pair constraint is active.
            
            if min_wall_val <= min_pair_val:
                # Wall is bottleneck
                i = min_wall_idx
                # Move away from the nearest wall
                # Determine which wall is closest
                x, y = centers[i]
                # Distance to walls: x+0.5, 0.5-x, y+0.5, 0.5-y
                dists_to_walls = [x + 0.5, 0.5 - x, y + 0.5, 0.5 - y]
                min_wall_dist = min(dists_to_walls)
                
                # Direction to move
                if min_wall_dist == x + 0.5: # Left wall
                    move_dir = np.array([1, 0])
                elif min_wall_dist == 0.5 - x: # Right wall
                    move_dir = np.array([-1, 0])
                elif min_wall_dist == y + 0.5: # Bottom wall
                    move_dir = np.array([0, 1])
                else: # Top wall
                    move_dir = np.array([0, -1])
                
                # Ideal move: move by min_wall_val to make dist 0? No, we want to increase min_wall_val.
                # If we move by d, new dist is min_wall_val + d.
                # But we must not violate pair constraints.
                # Heuristic: move by a fraction of the "slack" or just a fixed small step?
                # Better: Move towards the center of the available space.
                # For now, simple move: move by step_scale * min_wall_val? 
                # Actually, if we move by min_wall_val, the new dist is 2*min_wall_val.
                # But this might push into neighbors.
                # Let's try moving by a step proportional to the current radius.
                step_len = step_scale * current_r
                new_pos = centers[i] + move_dir * step_len
                
                # Clip to bounds
                new_pos = np.clip(new_pos, [-0.5, -0.5], [0.5, 0.5])
                centers[i] = new_pos
            else:
                # Pair is bottleneck
                i = min_pair_idx_i
                j = min_pair_idx_j
                pi = centers[i]
                pj = centers[j]
                vec = pj - pi
                dist = np.linalg.norm(vec)
                
                if dist < 1e-9:
                    # Coincident, move randomly
                    centers[i] += np.random.randn(2) * 0.001
                    centers[j] -= np.random.randn(2) * 0.001
                else:
                    unit_vec = vec / dist
                    
                    # We want to increase dist.
                    # Move i by -unit_vec * d, j by unit_vec * d.
                    # New dist = dist + 2*d.
                    # We want new dist/2 to be larger than current min.
                    # Ideal d: such that new_dist/2 == min_wall_val (if wall is the next bottleneck)
                    # Or just a fixed step.
                    # Let's use a step based on the current radius.
                    step_len = step_scale * current_r
                    
                    new_pi = pi - unit_vec * step_len
                    new_pj = pj + unit_vec * step_len
                    
                    # Clip
                    new_pi = np.clip(new_pi, [-0.5, -0.5], [0.5, 0.5])
                    new_pj = np.clip(new_pj, [-0.5, -0.5], [0.5, 0.5])
                    
                    centers[i] = new_pi
                    centers[j] = new_pj
            
            # Decay step scale slowly to allow fine-tuning
            step_scale = max(min_step_scale, step_scale * 0.999)
            
            # Random perturbation to escape local minima (very small)
            if np.random.rand() < 0.05:
                k = np.random.randint(N)
                centers[k] += np.random.randn(2) * current_r * 0.01
                centers[k] = np.clip(centers[k], [-0.5, -0.5], [0.5, 0.5])

        return best_centers, best_r

    def local_search_polish(centers, max_rounds=10):
        """Final high-precision polish using direct local moves."""
        centers = centers.copy()
        current_r = compute_radius(centers)
        
        for round_num in range(max_rounds):
            if time.time() > deadline - 5:
                break
                
            improved = False
            tree = cKDTree(centers)
            
            for i in range(N):
                if time.time() > deadline - 2:
                    break
                    
                orig = centers[i].copy()
                best_pos = orig
                best_r = current_r
                
                # Find nearest neighbors
                dists, idxs = tree.query(orig, k=5)
                
                # Try small moves in 8 directions + random
                step = 0.001 / (round_num + 1)
                directions = [
                    (1, 0), (-1, 0), (0, 1), (0, -1),
                    (1, 1), (1, -1), (-1, 1), (-1, -1)
                ]
                directions = [np.array(d) / np.linalg.norm(d) for d in directions]
                for _ in range(4):
                    angle = np.random.rand() * 2 * np.pi
                    directions.append(np.array([math.cos(angle), math.sin(angle)]))
                
                for d in directions:
                    for step_mult in [0.5, 1.0, 2.0, 4.0]:
                        new_center = orig + d * step * step_mult
                        if new_center[0] < -0.5 or new_center[0] > 0.5:
                            continue
                        if new_center[1] < -0.5 or new_center[1] > 0.5:
                            continue
                            
                        centers[i] = new_center
                        new_r = compute_radius(centers)
                        
                        if new_r > best_r + 1e-12:
                            best_r = new_r
                            best_pos = new_center.copy()
                            improved = True
                            
                centers[i] = best_pos
                
            current_r = best_r
            
            if not improved:
                break
                
        return centers, current_r

    # Generate diverse initial configurations
    best_centers = None
    best_radius = 0.0
    
    # Try multiple hexagonal layouts
    configs = []
    for n_rows in range(13, 17):
        n_cols = math.ceil(N / n_rows)
        pts = hexagonal_initialization(n_rows, n_cols)
        if pts is not None:
            configs.append(pts)
            
    # Square grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    configs.append(grid[:N])
    
    # Optimize each configuration
    for idx, init_pts in enumerate(configs):
        if time.time() > deadline - 20:
            break
            
        # Evaluate initial
        r = compute_radius(init_pts)
        if r > best_radius:
            best_radius = r
            best_centers = init_pts.copy()
            
        # Max-Min Expansion
        if time.time() < deadline - 15:
            try:
                exp_time_limit = min(deadline - 10, time.time() + 25)
                exp_centers, exp_r = max_min_expansion(init_pts, exp_time_limit)
                if exp_r > best_radius:
                    best_radius = exp_r
                    best_centers = exp_centers.copy()
            except Exception:
                pass
                
        # Final polish on best so far
        if time.time() < deadline - 5:
            polished, r_polished = local_search_polish(best_centers, max_rounds=5)
            if r_polished > best_radius:
                best_radius = r_polished
                best_centers = polished.copy()

    # Final high-precision polish
    if best_centers is not None and time.time() < deadline - 2:
        polished, r_polished = local_search_polish(best_centers, max_rounds=10)
        if r_polished > best_radius:
            best_radius = r_polished
            best_centers = polished.copy()

    if best_centers is None:
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers
