import numpy as np
import time

N = 200
DEADLINE = 110.0

def _compute_scores(c):
    """Compute the minimum radius for a configuration c."""
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    wall_dists = np.min(np.stack([
        0.5 - c[:, 0],
        0.5 + c[:, 0],
        0.5 - c[:, 1],
        0.5 + c[:, 1]
    ], axis=1), axis=1)
    
    min_pair_dist = np.min(dists)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _soft_min_gradient_ascent(c0, max_iters=5000, time_limit=100.0):
    c = c0.copy()
    start_time = time.time()
    
    beta = 10.0
    eta = 0.01
    
    best_c = c.copy()
    best_r = _compute_scores(c)
    
    for i in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        
        wall_dists = np.min(np.stack([
            0.5 - c[:, 0],
            0.5 + c[:, 0],
            0.5 - c[:, 1],
            0.5 + c[:, 1]
        ], axis=1), axis=1)
        
        mask = dists > 1e-10
        dists_safe = np.where(mask, dists, 1.0)
        
        v_pairs = dists / 2.0
        w_pairs = np.exp(-beta * v_pairs)
        np.fill_diagonal(w_pairs, 0.0)
        
        A = np.where(mask, w_pairs / (2.0 * dists_safe), 0.0)
        
        sum_A = np.sum(A, axis=1)
        sum_Ac = A @ c
        
        G_pairs = c * sum_A[:, np.newaxis] - sum_Ac
        
        v_w0 = 0.5 - c[:, 0]
        v_w1 = 0.5 + c[:, 0]
        v_w2 = 0.5 - c[:, 1]
        v_w3 = 0.5 + c[:, 1]
        
        v_walls = np.stack([v_w0, v_w1, v_w2, v_w3], axis=1)
        w_walls = np.exp(-beta * v_walls)
        
        G_walls = np.zeros((N, 2))
        G_walls[:, 0] = -w_walls[:, 0] + w_walls[:, 1]
        G_walls[:, 1] = -w_walls[:, 2] + w_walls[:, 3]
        
        sum_w_pairs = np.sum(w_pairs)
        sum_w_walls = np.sum(w_walls)
        total_w = sum_w_pairs + sum_w_walls
        
        if total_w < 1e-15:
            break
            
        G = (G_pairs + G_walls) / total_w
        
        c = c + eta * G
        c = np.clip(c, -0.5, 0.5)
        
        r = _compute_scores(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
            
        beta *= 1.05
        eta *= 0.99
        
        if eta < 1e-6:
            break
            
    return best_c, best_r

def _init_hexagonal():
    s = np.sqrt(2.0 / (N * np.sqrt(3.0)))
    pts = []
    row = 0
    y = -0.5
    while y <= 0.5:
        x_offset = 0.0 if row % 2 == 0 else s * np.sqrt(3.0) / 2.0
        x = -0.5 + x_offset
        while x <= 0.5:
            pts.append([x, y])
            x += s * np.sqrt(3.0)
        y += s * 1.5
        row += 1
        
    pts = np.array(pts)
    
    if len(pts) > N:
        indices = np.random.choice(len(pts), N, replace=False)
        pts = pts[indices]
    elif len(pts) < N:
        missing = N - len(pts)
        new_pts = np.random.uniform(-0.5, 0.5, (missing, 2))
        pts = np.vstack([pts, new_pts])
        
    return pts

def _init_random():
    return np.random.uniform(-0.5, 0.5, (N, 2))

def construct():
    start_time = time.time()
    
    best_c = None
    best_r = -np.inf
    
    inits = [
        _init_hexagonal(),
        _init_random(),
        _init_hexagonal()
    ]
    
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    for idx, init in enumerate(inits):
        if time.time() - start_time > DEADLINE:
            break
            
        time_left = DEADLINE - (time.time() - start_time)
        time_per_init = time_left / (len(inits) - idx)
        
        c_opt, r_opt = _soft_min_gradient_ascent(init, max_iters=2000, time_limit=time_per_init)
        
        if r_opt > best_r:
            best_r = r_opt
            best_c = c_opt
            
    if best_c is None:
        best_c = inits[0]
        
    return best_c
