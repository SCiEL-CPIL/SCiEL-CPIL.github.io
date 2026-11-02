import numpy as np
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_score = -np.inf
    best_config = None
    
    def run_optimization(init_pos, init_rad, max_iters=10000, lr_init=0.01, lr_end=0.0001):
        state = np.copy(init_pos)
        state[:, 2] = np.copy(init_rad)
        v = np.zeros_like(state)
        beta = 0.9
        
        w_overlap = 10.0
        w_boundary = 10.0
        
        best_score_local = -np.inf
        best_config_local = None
        
        current_lr = lr_init
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
                
            x = state[:, 0]
            y = state[:, 1]
            r = state[:, 2]
            
            # Pairwise distances
            dx = x[:, np.newaxis] - x[np.newaxis, :]
            dy = y[:, np.newaxis] - y[np.newaxis, :]
            dist_sq = dx**2 + dy**2
            np.fill_diagonal(dist_sq, 1.0)
            dist = np.sqrt(dist_sq)
            
            # Overlaps
            sum_r = r[:, np.newaxis] + r[np.newaxis, :]
            overlap = np.maximum(0, sum_r - dist)
            
            # Directions
            dir_x = dx / dist
            dir_y = dy / dist
            
            # Forces from Overlap
            fx_ov = np.sum(overlap * dir_x, axis=1)
            fy_ov = np.sum(overlap * dir_y, axis=1)
            fr_ov_force = -2.0 * np.sum(overlap, axis=1)
            
            # Forces from Boundary
            fx_b_force = np.where(x < r, 1.0, 0.0) - np.where(x > 1 - r, 1.0, 0.0)
            fy_b_force = np.where(y < r, 1.0, 0.0) - np.where(y > 1 - r, 1.0, 0.0)
            
            fr_b_force = 0.0
            fr_b_force -= np.where(x < r, 2.0 * (r - x), 0.0)
            fr_b_force -= np.where(x > 1 - r, 2.0 * (r - (1 - x)), 0.0)
            fr_b_force -= np.where(y < r, 2.0 * (r - y), 0.0)
            fr_b_force -= np.where(y > 1 - r, 2.0 * (r - (1 - y)), 0.0)
            
            # Objective Force
            fr_obj_force = np.ones(N)
            
            # Total Forces
            fx_total = w_overlap * fx_ov + w_boundary * fx_b_force
            fy_total = w_overlap * fy_ov + w_boundary * fy_b_force
            fr_total = fr_obj_force + w_overlap * fr_ov_force + w_boundary * fr_b_force
            
            # Update
            v[:, 0] = beta * v[:, 0] + current_lr * fx_total
            v[:, 1] = beta * v[:, 1] + current_lr * fy_total
            v[:, 2] = beta * v[:, 2] + current_lr * fr_total
            
            state[:, 0] += v[:, 0]
            state[:, 1] += v[:, 1]
            state[:, 2] += v[:, 2]
            
            # Constraints
            state[:, 2] = np.maximum(state[:, 2], 1e-6)
            state[:, 0] = np.clip(state[:, 0], 0.0, 1.0)
            state[:, 1] = np.clip(state[:, 1], 0.0, 1.0)
            
            # Anneal
            current_lr = lr_init * (lr_end / lr_init) ** (i / max_iters)
            
            # Validity Check
            # Re-use overlap from above? No, state changed.
            # But we can approximate or just check at the end of the loop.
            # For speed, check every 100 iterations or at the end.
            # Checking every iteration is O(N^2). 10000 * 676 is fine.
            
            dx_v = x[:, np.newaxis] - x[np.newaxis, :]
            dy_v = y[:, np.newaxis] - y[np.newaxis, :]
            dist_v = np.sqrt(dx_v**2 + dy_v**2)
            np.fill_diagonal(dist_v, 1.0)
            sum_r_v = r[:, np.newaxis] + r[np.newaxis, :]
            overlap_v = np.maximum(0, sum_r_v - dist_v)
            
            bnd_ok = np.all(x >= r) and np.all(x <= 1 - r) and np.all(y >= r) and np.all(y <= 1 - r)
            ov_ok = np.all(overlap_v < 1e-9)
            
            if bnd_ok and ov_ok:
                score = np.sum(r)
                if score > best_score_local:
                    best_score_local = score
                    best_config_local = np.copy(state)
                    
        return best_score_local, best_config_local

    num_restarts = 20
    for k in range(num_restarts):
        if time.time() > deadline:
            break
            
        side = int(np.ceil(np.sqrt(N)))
        xs = []
        ys = []
        count = 0
        for i in range(side):
            for j in range(side):
                if count >= N:
                    break
                cx = (i + 0.5) / side
                cy = (j + 0.5) / side
                jitter = 0.1 / side
                cx += np.random.uniform(-jitter, jitter)
                cy += np.random.uniform(-jitter, jitter)
                xs.append(cx)
                ys.append(cy)
                count += 1
            if count >= N:
                break
        
        init_pos = np.column_stack([xs, ys])
        init_rad = np.full(N, 0.45 / side)
        
        score, config = run_optimization(init_pos, init_rad)
        
        if config is not None and score > best_score:
            best_score = score
            best_config = config

    if best_config is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.4 / side) for i in range(side) for j in range(side)]
        best_config = np.array(cells[:N])
        
    return best_config
