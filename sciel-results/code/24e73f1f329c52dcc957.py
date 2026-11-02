import numpy as np
import time
import math

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Initialization: Hexagonal lattice
    s = math.sqrt(math.pi / (math.sqrt(3) * N))
    
    points = []
    R = 1.0
    max_r = int(R / s) + 2
    for q in range(-max_r, max_r + 1):
        for r_ in range(-max_r, max_r + 1):
            x = s * (q + r_ / 2.0)
            y = s * (math.sqrt(3) / 2.0) * r_
            if x*x + y*y <= 1.0:
                points.append([x, y])
                
    points = np.array(points)
    
    if len(points) < N:
        s *= 0.9
        points = []
        max_r = int(R / s) + 2
        for q in range(-max_r, max_r + 1):
            for r_ in range(-max_r, max_r + 1):
                x = s * (q + r_ / 2.0)
                y = s * (math.sqrt(3) / 2.0) * r_
                if x*x + y*y <= 1.0:
                    points.append([x, y])
        points = np.array(points)
        
    if len(points) > N:
        dists_center = np.linalg.norm(points, axis=1)
        sorted_indices = np.argsort(dists_center)
        points = points[sorted_indices[:N]]
        
    # Project inside
    dists = np.linalg.norm(points, axis=1)
    mask = dists > 0.99
    if np.any(mask):
        points[mask] = points[mask] * (0.99 / dists[mask])[:, np.newaxis]

    # 2. Optimization
    beta = 10.0
    lr = 0.005
    best_score = -np.inf
    best_points = points.copy()
    
    iu = np.triu_indices(N, 1)
    
    while time.time() < deadline - 5.0:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_mat = np.linalg.norm(diff, axis=2)
        norms = np.linalg.norm(points, axis=1)
        margins = 1.0 - norms
        
        eps = 1e-8
        dists_mat_safe = np.maximum(dists_mat, eps)
        margins_safe = np.maximum(margins, eps)
        
        d_pairs = dists_mat_safe[iu]
        
        exp_pairs = np.exp(-beta * d_pairs)
        exp_margins = np.exp(-beta * margins_safe)
        
        Z = np.sum(exp_pairs) + np.sum(exp_margins)
        
        # Gradient
        W = np.zeros((N, N))
        W[iu] = exp_pairs
        W = W + W.T
        np.fill_diagonal(W, 0)
        
        U = diff / dists_mat_safe[:, :, np.newaxis]
        np.fill_diagonal(U, 0)
        
        F_pairs = np.sum(W[:, :, np.newaxis] * U, axis=1)
        
        w_m = exp_margins / (beta * Z)
        norms_safe = np.maximum(norms, eps)
        unit_vecs = points / norms_safe[:, np.newaxis]
        F_boundary = -w_m[:, np.newaxis] * unit_vecs
        
        grad = F_pairs + F_boundary
        
        points = points + lr * grad
        
        # Project
        new_norms = np.linalg.norm(points, axis=1)
        outside = new_norms > 1.0
        if np.any(outside):
            points[outside] = points[outside] / new_norms[outside, np.newaxis]
            
        # Score
        d_min = np.min(d_pairs) / 2.0
        m_min = np.min(margins)
        score = min(d_min, m_min)
        
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
        beta *= 1.05
        lr *= 0.98

    points = best_points

    # 3. Polish
    for _ in range(200):
        if time.time() > deadline - 2.0:
            break
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_mat = np.linalg.norm(diff, axis=2)
        norms = np.linalg.norm(points, axis=1)
        margins = 1.0 - norms
        
        d_pairs = dists_mat[iu]
        r_pair = np.min(d_pairs) / 2.0
        r_bound = np.min(margins)
        r = min(r_pair, r_bound)
        
        if r_pair < r_bound:
            idx = np.argmin(d_pairs)
            i, j = iu[0][idx], iu[1][idx]
            vec = points[i] - points[j]
            dist = np.linalg.norm(vec)
            if dist < 1e-8: continue
            vec = vec / dist
            step = 1e-5
            points[i] += vec * step
            points[j] -= vec * step
        else:
            idx = np.argmin(margins)
            i = idx
            vec = -points[i]
            dist = np.linalg.norm(vec)
            if dist < 1e-8: continue
            vec = vec / dist
            step = 1e-5
            points[i] += vec * step
            
        new_norms = np.linalg.norm(points, axis=1)
        outside = new_norms > 1.0
        if np.any(outside):
            points[outside] = points[outside] / new_norms[outside, np.newaxis]

    return points
