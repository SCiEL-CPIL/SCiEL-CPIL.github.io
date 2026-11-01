"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 110.0

def _compute_radius(points):
    # points: (N, 2)
    # Compute min distance between any two centers / 2
    # and min distance to boundary
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    # Boundary distances
    min_boundary = np.min([
        np.min(points[:, 0] + 0.5),
        np.min(0.5 - points[:, 0]),
        np.min(points[:, 1] + 0.5),
        np.min(0.5 - points[:, 1])
    ])
    
    return min(min_dist / 2.0, min_boundary)

def _hex_init():
    # Generate hexagonal lattice points
    # Spacing based on r ~ 0.07
    s_x = 0.14
    s_y = 0.1212
    
    points = []
    for j in range(-10, 11):
        y = j * s_y
        if y < -0.5 or y > 0.5:
            continue
        offset = 0 if j % 2 == 0 else s_x / 2.0
        for i in range(-10, 11):
            x = i * s_x + offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) > N:
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        missing = N - len(points)
        new_points = np.random.uniform(-0.45, 0.45, size=(missing, 2))
        points = np.vstack([points, new_points])
        
    return points

def construct():
    start_time = time.time()
    
    # 1. Initialize
    P = _hex_init()
    P = P + np.random.normal(0, 0.01, P.shape)
    P = np.clip(P, -0.5, 0.5)
    
    r_curr = _compute_radius(P)
    P_best = P.copy()
    r_best = r_curr
    
    # 2. Simulated Annealing
    T = 0.01
    T_min = 1e-6
    cooling = 0.9995
    move_size = 0.01
    
    last_improve = 0
    steps = 0
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT:
            break
        
        idx = np.random.randint(N)
        old_pos = P[idx].copy()
        
        delta = np.random.normal(0, move_size, size=2)
        new_pos = old_pos + delta
        new_pos[0] = np.clip(new_pos[0], -0.5, 0.5)
        new_pos[1] = np.clip(new_pos[1], -0.5, 0.5)
        
        P[idx] = new_pos
        r_new = _compute_radius(P)
        
        if r_new >= r_curr:
            r_curr = r_new
            if r_new > r_best:
                P_best = P.copy()
                r_best = r_new
                last_improve = steps
        else:
            p = np.exp((r_new - r_curr) / T)
            if np.random.random() < p:
                r_curr = r_new
            else:
                P[idx] = old_pos
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        # Stagnation detection and kick
        if steps - last_improve > 20000:
            P = P_best + np.random.normal(0, 0.02, P_best.shape)
            P = np.clip(P, -0.5, 0.5)
            r_curr = _compute_radius(P)
            T = 0.01
            last_improve = steps
            
        steps += 1

    # 3. Local Polish (Hill Climbing)
    if time.time() - start_time < TIME_LIMIT - 5:
        eps = 0.001
        improved = True
        while improved and (time.time() - start_time < TIME_LIMIT - 2):
            improved = False
            for i in range(N):
                for dx, dy in [(eps, 0), (-eps, 0), (0, eps), (0, -eps)]:
                    old = P_best[i].copy()
                    new = old + [dx, dy]
                    if new[0] < -0.5 or new[0] > 0.5 or new[1] < -0.5 or new[1] > 0.5:
                        continue
                    P_best[i] = new
                    r_new = _compute_radius(P_best)
                    if r_new > r_best:
                        r_best = r_new
                        improved = True
                    else:
                        P_best[i] = old

    return P_best
# EVOLVE-BLOCK-END
