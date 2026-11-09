import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    np.random.seed(42)
    
    # 1. Greedy Best-Fit Initialization
    centers = np.zeros((N, 3))
    centers[0] = [0, 0, 0]
    
    for i in range(1, N):
        samples = np.random.uniform(-0.5, 0.5, size=(500, 3))
        
        diff = samples[:, np.newaxis, :] - centers[:i][np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        min_dists_to_others = np.min(dists, axis=1)
        
        wall_dists = 0.5 - np.abs(samples)
        min_wall_dists = np.min(wall_dists, axis=1)
        
        scores = np.minimum(min_dists_to_others, min_wall_dists)
        
        best_idx = np.argmax(scores)
        centers[i] = samples[best_idx]
        
    best_centers = centers.copy()
    
    def calc_score(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        wall_dists = 0.5 - np.abs(c)
        min_wall_dist = np.min(wall_dists)
        
        return min(min_pair_dist / 2.0, min_wall_dist)
    
    best_score = calc_score(best_centers)
    
    # 2. Force-Directed Relaxation
    current_centers = best_centers.copy()
    lr = 0.05
    cutoff = 0.25
    
    while time.time() < deadline - 30.0:
        diff = current_centers[:, np.newaxis, :] - current_centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        mask = dists < cutoff
        mask[np.diag_indices(N)] = False
        
        i_idx, j_idx = np.where(mask)
        
        if len(i_idx) == 0:
            break
            
        d_vals = dists[i_idx, j_idx]
        d_safe = np.where(d_vals > 1e-10, d_vals, 1.0)
        
        mag = (1.0 - d_vals / cutoff) / d_safe
        dirs = diff[i_idx, j_idx]
        
        contrib = mag[:, np.newaxis] * dirs
        
        forces = np.zeros_like(current_centers)
        np.add.at(forces, i_idx, contrib)
        np.add.at(forces, j_idx, -contrib)
        
        margin = 0.1
        for k in range(3):
            close_to_pos = current_centers[:, k] > 0.5 - margin
            if np.any(close_to_pos):
                dist_to_wall = 0.5 - current_centers[close_to_pos, k]
                forces[close_to_pos, k] -= (margin - dist_to_wall) / margin
            
            close_to_neg = current_centers[:, k] < -0.5 + margin
            if np.any(close_to_neg):
                dist_to_wall = current_centers[close_to_neg, k] + 0.5
                forces[close_to_neg, k] += (margin - dist_to_wall) / margin
        
        current_centers += lr * forces
        current_centers = np.clip(current_centers, -0.5, 0.5)
        
        score = calc_score(current_centers)
        if score > best_score:
            best_score = score
            best_centers = current_centers.copy()
            
        lr *= 0.9995
        
    # 3. Simulated Annealing
    current_centers = best_centers.copy()
    T = 0.01
    cooling = 0.9995
    current_score = calc_score(current_centers)
    
    while time.time() < deadline:
        i = np.random.randint(N)
        old_pos = current_centers[i].copy()
        
        step = np.random.normal(0, 0.02, size=3)
        current_centers[i] += step
        current_centers[i] = np.clip(current_centers[i], -0.5, 0.5)
        
        new_score = calc_score(current_centers)
        delta = new_score - current_score
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_centers = current_centers.copy()
        else:
            current_centers[i] = old_pos
            
        T *= cooling
        if T < 1e-7:
            break
            
    return best_centers
