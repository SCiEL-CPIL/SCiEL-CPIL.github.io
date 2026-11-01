import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # 1. Initialization
    # Hexagonal lattice
    m = 14
    n = 15
    dx = 1.0 / n
    dy = dx * np.sqrt(3) / 2
    
    xs = []
    ys = []
    for i in range(m):
        for j in range(n):
            x = -0.5 + (j + 0.5) * dx + (i % 2) * dx * 0.5
            y = -0.5 + (i + 0.5) * dy
            xs.append(x)
            ys.append(y)
            
    pts = np.array(list(zip(xs, ys)))
    pts -= pts.mean(axis=0)
    
    # If we have more than N, take the first N. If less, pad? 
    # 14*15 = 210. We take 200.
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        # Pad with random points in the square
        extra = np.random.uniform(-0.4, 0.4, (N - len(pts), 2))
        pts = np.vstack([pts, extra])
        
    best_pts = pts.copy()
    best_score = -np.inf
    
    # Precompute pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Objective function
    def objective(x, beta):
        # x is shape (2N,)
        c = x.reshape(N, 2)
        
        # Distances between pairs
        # c[i] - c[j]
        diffs = c[i_idx] - c[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        
        # Wall distances
        # 0.5 - |c_ik|
        wall_dists = 0.5 - np.abs(c).max(axis=1) # Wait, min over k of (0.5 - |c_ik|) is 0.5 - max(|c_x|, |c_y|)
        
        # All distances
        all_dists = np.concatenate([dists, wall_dists])
        
        # Soft min: -1/beta * log(sum(exp(-beta * d)))
        # To avoid overflow, use log-sum-exp trick
        # min_d = np.min(all_dists)
        # val = min_d - 1/beta * np.log(np.sum(np.exp(-beta * (all_dists - min_d))))
        
        # Simpler: just use the softmin formula directly if beta is small enough
        # For stability:
        m_d = np.min(all_dists)
        exps = np.exp(-beta * (all_dists - m_d))
        soft_min = m_d - np.log(np.sum(exps)) / beta
        
        return -soft_min # We minimize negative soft min to maximize soft min

    def gradient(x, beta):
        c = x.reshape(N, 2)
        diffs = c[i_idx] - c[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        
        wall_dists = 0.5 - np.abs(c).max(axis=1)
        
        all_dists = np.concatenate([dists, wall_dists])
        
        m_d = np.min(all_dists)
        exps = np.exp(-beta * (all_dists - m_d))
        weights = exps / np.sum(exps)
        
        # Gradient of soft_min w.r.t d_k is -weights_k
        # So gradient of -soft_min is weights_k * grad(d_k)
        
        grad_dists = np.zeros_like(dists)
        grad_wall = np.zeros_like(wall_dists)
        
        # Pairwise distances gradient
        # d = ||c_i - c_j||
        # grad_{c_i} d = (c_i - c_j) / d
        # grad_{c_j} d = (c_j - c_i) / d = -grad_{c_i} d
        # Avoid division by zero
        mask = dists > 1e-12
        unit_vecs = diffs[mask] / dists[mask, None]
        
        # Accumulate gradient for points
        grad_c = np.zeros_like(c)
        
        # For each pair, add weight * unit_vec to c_i and -weight * unit_vec to c_j
        # weights are for the distance, so grad of distance is unit_vec
        # We want grad of -soft_min = sum w_k * grad(d_k)
        
        # Vectorized accumulation is tricky with indices.
        # Let's do it in a loop for clarity and correctness, N=200 is small enough?
        # 200*199/2 = ~20k pairs. Python loop might be slow.
        # Let's use np.add.at
        
        # grad_dists for each pair is the unit vector
        # But we need to map back to c.
        
        # Let's compute grad for pairwise part
        # grad_pair_c[i] += w * u
        # grad_pair_c[j] -= w * u
        
        # Create arrays of indices and weights
        w_pairs = weights[:len(dists)]
        
        # Only for valid distances
        w_pairs = np.where(mask, w_pairs, 0)
        
        # u is (num_pairs, 2)
        # We need to scatter add
        # This is complex to vectorize perfectly without sparse matrices.
        # Given N=200, a simple loop over pairs might take ~10ms. 
        # 100 iterations * 10ms = 1s. Acceptable.
        
        grad_c_pairs = np.zeros_like(c)
        for k in range(len(dists)):
            if mask[k]:
                w = w_pairs[k]
                u = unit_vecs[k]
                i = i_idx[k]
                j = j_idx[k]
                grad_c_pairs[i] += w * u
                grad_c_pairs[j] -= w * u
                
        # Wall distances gradient
        # d_wall = 0.5 - max(|x|, |y|)
        # grad is -sign(x) or -sign(y) depending on which is max
        # If |x| > |y|, grad_x = -sign(x), grad_y = 0
        # If |y| > |x|, grad_x = 0, grad_y = -sign(y)
        # If equal, split?
        
        w_walls = weights[len(dists):]
        
        abs_c = np.abs(c)
        max_idx = np.argmax(abs_c, axis=1) # 0 for x, 1 for y
        
        grad_c_walls = np.zeros_like(c)
        for i in range(N):
            w = w_walls[i]
            if w == 0: continue
            if max_idx[i] == 0:
                grad_c_walls[i, 0] = -w * np.sign(c[i, 0])
            else:
                grad_c_walls[i, 1] = -w * np.sign(c[i, 1])
                
        grad_c = grad_c_pairs + grad_c_walls
        return grad_c.flatten()

    # Optimization loop
    current_pts = best_pts
    beta = 10.0
    
    while time.time() < deadline:
        # Run minimize
        res = minimize(objective, current_pts.flatten(), args=(beta,), method='L-BFGS-B', 
                       jac=gradient, bounds=[(-0.5, 0.5)] * (2 * N), options={'maxiter': 50, 'ftol': 1e-8})
        
        if res.success or res.status == 0:
            new_pts = res.x.reshape(N, 2)
            # Calculate actual score
            dists = np.linalg.norm(new_pts[i_idx] - new_pts[j_idx], axis=1)
            wall_dists = 0.5 - np.abs(new_pts).max(axis=1)
            r = np.min(np.concatenate([dists / 2.0, wall_dists]))
            
            if r > best_score:
                best_score = r
                best_pts = new_pts
                current_pts = new_pts
            else:
                # If no improvement, maybe increase beta or break
                beta *= 2
                if beta > 1000:
                    break
        else:
            # If minimize failed, try a smaller step or break
            beta *= 0.5
            if beta < 1:
                break

    return best_pts

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # 1. Initialization: Hexagonal lattice
    m = 14
    n = 15
    dx = 1.0 / n
    dy = dx * np.sqrt(3) / 2
    
    xs = []
    ys = []
    for i in range(m):
        for j in range(n):
            x = -0.5 + (j + 0.5) * dx + (i % 2) * dx * 0.5
            y = -0.5 + (i + 0.5) * dy
            xs.append(x)
            ys.append(y)
            
    pts = np.array(list(zip(xs, ys)))
    pts -= pts.mean(axis=0)
    
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        extra = np.random.uniform(-0.4, 0.4, (N - len(pts), 2))
        pts = np.vstack([pts, extra])
        
    best_pts = pts.copy()
    best_score = -np.inf
    
    # Precompute pairs
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Objective function
    def objective(x, beta):
        c = x.reshape(N, 2)
        diffs = c[i_idx] - c[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        wall_dists = 0.5 - np.abs(c).max(axis=1)
        all_dists = np.concatenate([dists, wall_dists])
        
        m_d = np.min(all_dists)
        exps = np.exp(-beta * (all_dists - m_d))
        soft_min = m_d - np.log(np.sum(exps)) / beta
        return -soft_min

    def gradient(x, beta):
        c = x.reshape(N, 2)
        diffs = c[i_idx] - c[j_idx]
        dists = np.linalg.norm(diffs, axis=1)
        wall_dists = 0.5 - np.abs(c).max(axis=1)
        all_dists = np.concatenate([dists, wall_dists])
        
        m_d = np.min(all_dists)
        exps = np.exp(-beta * (all_dists - m_d))
        weights = exps / np.sum(exps)
        
        mask = dists > 1e-12
        unit_vecs = np.zeros_like(diffs)
        unit_vecs[mask] = diffs[mask] / dists[mask, None]
        
        grad_c = np.zeros_like(c)
        
        # Pairwise
        w_pairs = weights[:len(dists)]
        w_pairs = np.where(mask, w_pairs, 0)
        
        # Scatter add
        # grad_c[i] += w * u
        # grad_c[j] -= w * u
        # Using np.add.at for vectorization
        np.add.at(grad_c, i_idx, w_pairs[:, None] * unit_vecs)
        np.add.at(grad_c, j_idx, -w_pairs[:, None] * unit_vecs)
        
        # Walls
        w_walls = weights[len(dists):]
        abs_c = np.abs(c)
        max_idx = np.argmax(abs_c, axis=1)
        
        grad_wall = np.zeros_like(c)
        # x component
        grad_wall[:, 0] = np.where(max_idx == 0, -w_walls * np.sign(c[:, 0]), 0)
        # y component
        grad_wall[:, 1] = np.where(max_idx == 1, -w_walls * np.sign(c[:, 1]), 0)
        
        grad_c += grad_wall
        return grad_c.flatten()

    current_pts = best_pts
    beta = 10.0
    
    while time.time() < deadline:
        res = minimize(objective, current_pts.flatten(), args=(beta,), method='L-BFGS-B', 
                       jac=gradient, bounds=[(-0.5, 0.5)] * (2 * N), options={'maxiter': 100, 'ftol': 1e-10})
        
        new_pts = res.x.reshape(N, 2)
        dists = np.linalg.norm(new_pts[i_idx] - new_pts[j_idx], axis=1)
        wall_dists = 0.5 - np.abs(new_pts).max(axis=1)
        r = np.min(np.concatenate([dists / 2.0, wall_dists]))
        
        if r > best_score:
            best_score = r
            best_pts = new_pts
            current_pts = new_pts
            beta *= 1.5
        else:
            beta *= 0.8
            if beta < 1:
                break

    return best_pts
# EVOLVE-BLOCK-END
