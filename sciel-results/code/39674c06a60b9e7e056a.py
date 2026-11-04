import numpy as np
import time

N = 21

def construct():
    start_time = time.time()
    max_time = 110.0
    
    best_score = 0.0
    best_result = None
    
    def compute_radii_vec(positions, W, H):
        # positions: (N, 2)
        # Compute distances to walls
        # Left: x, Right: W-x, Bottom: y, Top: H-y
        dist_walls = np.stack([
            positions[:, 0], 
            W - positions[:, 0], 
            positions[:, 1], 
            H - positions[:, 1]
        ], axis=1) # (N, 4)
        
        # Compute pairwise distances
        # diff: (N, N, 2)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2)) # (N, N)
        np.fill_diagonal(dists, np.inf)
        
        # Min distance to any neighbor
        min_neighbor = np.min(dists, axis=1) # (N,)
        
        # Min distance to any wall
        min_wall = np.min(dist_walls, axis=1) # (N,)
        
        # Overall min
        min_dist = np.minimum(min_neighbor, min_wall)
        
        radii = 0.5 * min_dist
        return radii

    def get_score_vec(positions, W, H):
        radii = compute_radii_vec(positions, W, H)
        return np.sum(radii)

    def optimize_restart(initial_positions, initial_W):
        H = 2.0 - initial_W
        positions = initial_positions.copy()
        W = initial_W
        
        step_size = 0.05
        patience = 0
        max_patience = 100
        
        current_score = get_score_vec(positions, W, H)
        
        for iter in range(500):
            if time.time() - start_time > max_time:
                break
                
            # Calculate gradients
            # We need to know which constraint is active for each circle
            # This requires finding the argmin for each circle
            
            radii = compute_radii_vec(positions, W, H)
            
            # Re-compute min distances to find active constraints
            # This is a bit redundant but N is small.
            # Let's just use the logic from before but vectorized where possible.
            
            grad_pos = np.zeros_like(positions)
            grad_W = 0.0
            
            # Vectorized distance to walls
            d_left = positions[:, 0]
            d_right = W - positions[:, 0]
            d_bottom = positions[:, 1]
            d_top = H - positions[:, 1]
            
            # Pairwise distances
            diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # For each i, find min dist and type
            # We can check wall distances first
            min_dist = np.minimum.reduce([d_left, d_right, d_bottom, d_top])
            active_wall = np.zeros(N, dtype=int) # 0:none, 1:left, 2:right, 3:bottom, 4:top
            # This logic is tricky to vectorize fully for "active type".
            # Given N=21, a Python loop for gradient calculation is fine.
            
            for i in range(N):
                x, y = positions[i]
                min_d = np.inf
                active_type = -1
                active_j = -1
                
                if x < min_d:
                    min_d = x; active_type = 0
                if W - x < min_d:
                    min_d = W - x; active_type = 1
                if y < min_d:
                    min_d = y; active_type = 2
                if H - y < min_d:
                    min_d = H - y; active_type = 3
                
                for j in range(N):
                    if i == j: continue
                    dx = x - positions[j, 0]
                    dy = y - positions[j, 1]
                    d = np.sqrt(dx*dx + dy*dy)
                    if d < min_d:
                        min_d = d; active_type = 4; active_j = j
                        
                if active_type == 0: # Left
                    grad_pos[i, 0] += 0.5
                elif active_type == 1: # Right
                    grad_pos[i, 0] -= 0.5
                    grad_W += 0.5
                elif active_type == 2: # Bottom
                    grad_pos[i, 1] += 0.5
                elif active_type == 3: # Top
                    grad_pos[i, 1] -= 0.5
                    grad_W += -0.5
                elif active_type == 4: # Neighbor
                    dx = x - positions[active_j, 0]
                    dy = y - positions[active_j, 1]
                    d = min_d
                    if d > 1e-9:
                        grad_pos[i, 0] += 0.5 * (dx / d)
                        grad_pos[i, 1] += 0.5 * (dy / d)
                        grad_pos[active_j, 0] += -0.5 * (dx / d)
                        grad_pos[active_j, 1] += -0.5 * (dy / d)
                        
            positions += step_size * grad_pos
            W += step_size * grad_W
            W = np.clip(W, 0.5, 1.5)
            H = 2.0 - W
            positions[:, 0] = np.clip(positions[:, 0], 0.01, W - 0.01)
            positions[:, 1] = np.clip(positions[:, 1], 0.01, H - 0.01)
            
            new_score = get_score_vec(positions, W, H)
            
            if new_score > current_score:
                current_score = new_score
                patience = 0
                step_size = min(step_size * 1.05, 0.2)
            else:
                patience += 1
                step_size *= 0.9
                if patience > max_patience:
                    break
                    
        return current_score, positions, W, H

    num_restarts = 20
    for r in range(num_restarts):
        if time.time() - start_time > max_time:
            break
            
        W_init = 1.0 + 0.5 * np.random.rand()
        H_init = 2.0 - W_init
        
        rows = int(np.sqrt(N * H_init / W_init))
        cols = int(np.sqrt(N * W_init / H_init))
        
        positions = np.zeros((N, 2))
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= N:
                    break
                x = (j + 0.5) * (W_init / cols) + (0.5 * (W_init / cols) if i % 2 == 1 else 0)
                y = (i + 0.5) * (H_init / rows)
                x += np.random.uniform(-0.05, 0.05)
                y += np.random.uniform(-0.05, 0.05)
                positions[idx, 0] = x
                positions[idx, 1] = y
                idx += 1
        
        if idx < N:
            for k in range(idx, N):
                positions[k, 0] = np.random.uniform(0.1, W_init - 0.1)
                positions[k, 1] = np.random.uniform(0.1, H_init - 0.1)
        
        score, pos, W, H = optimize_restart(positions, W_init)
        
        if score > best_score:
            best_score = score
            radii = compute_radii_vec(pos, W, H)
            best_result = np.column_stack([pos, radii])

    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    else:
        # Ensure radii are positive
        # If any radius is 0, it's a degenerate circle, but allowed? "disjoint interiors"
        # Usually r>0 is implied.
        # Our calculation ensures r >= 0.
        return best_result

# The prompt requires keeping the function name `construct`
# I will remove the duplicate definition and ensure only one exists.
