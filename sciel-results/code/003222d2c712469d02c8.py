import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Initialization: Hexagonal Grid
    s = 0.074
    
    points = []
    max_k = int(np.ceil(1.0 / (s * np.sqrt(3) / 2))) + 2
    max_l = int(np.ceil(1.0 / s)) + 2
    
    for k in range(max_k):
        y = -0.5 + (k + 0.5) * s * np.sqrt(3) / 2
        if y > 0.5 or y < -0.5:
            continue
        offset = 0.5 * s if k % 2 == 1 else 0.0
        for l in range(max_l):
            x = -0.5 + (l + 0.5) * s + offset
            if x > 0.5 or x < -0.5:
                continue
            points.append([x, y])
            
    points = np.array(points)
    
    if len(points) > N:
        indices = np.random.choice(len(points), N, replace=False)
        centers = points[indices]
    elif len(points) < N:
        random_pts = np.random.uniform(-0.5, 0.5, size=(N - len(points), 2))
        centers = np.vstack([points, random_pts])
    else:
        centers = points
        
    # 2. Repulsion Dynamics
    centers = centers.copy()
    eta = 0.01
    eta_decay = 0.999
    
    while time.time() < deadline - 10.0: # Leave 10s for polish
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1.0)
        
        inv_dists_sq = 1.0 / dists_sq
        np.fill_diagonal(inv_dists_sq, 0.0)
        
        weighted_diff = diff * inv_dists_sq[:, :, np.newaxis]
        forces = np.sum(weighted_diff, axis=1)
        
        centers += eta * forces
        centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
        centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
        
        eta *= eta_decay
        if eta < 1e-6:
            break
            
    # 3. Final Polish
    if time.time() < deadline - 2.0:
        for _ in range(2000):
            if time.time() > deadline - 2.0:
                break
            
            # Find closest pair
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dist, (i, j) = _get_min_pair_info(dists)
            
            # Check wall distances
            # dist to left: x + 0.5
            # dist to right: 0.5 - x
            # dist to bottom: y + 0.5
            # dist to top: 0.5 - y
            d_left = centers[:, 0] + 0.5
            d_right = 0.5 - centers[:, 0]
            d_bottom = centers[:, 1] + 0.5
            d_top = 0.5 - centers[:, 1]
            
            min_wall_dist = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0))
            
            if min_dist <= min_wall_dist:
                # Move pair apart
                vec = centers[i] - centers[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    step = 1e-4
                    centers[i] += (vec / norm) * step
                    centers[j] -= (vec / norm) * step
            else:
                # Move point closest to wall
                # Find the point and wall
                wall_dists_per_point = np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0)
                idx = np.argmin(wall_dists_per_point)
                c = centers[idx]
                step = 1e-4
                # Determine direction
                if c[0] <= c[1] * 0: # dummy
                    pass
                # Check which wall is closest for this point
                dl = c[0] + 0.5
                dr = 0.5 - c[0]
                db = c[1] + 0.5
                dt = 0.5 - c[1]
                min_w = min(dl, dr, db, dt)
                if min_w == dl:
                    centers[idx][0] += step
                elif min_w == dr:
                    centers[idx][0] -= step
                elif min_w == db:
                    centers[idx][1] += step
                elif min_w == dt:
                    centers[idx][1] -= step
            
            centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
            centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)

    return centers

def _get_min_pair_info(dists):
    min_val = np.min(dists)
    idx = np.unravel_index(np.argmin(dists), dists.shape)
    return min_val, idx

# EVOLVE-BLOCK-START
# The code above is the full program content.
# EVOLVE-BLOCK-END
