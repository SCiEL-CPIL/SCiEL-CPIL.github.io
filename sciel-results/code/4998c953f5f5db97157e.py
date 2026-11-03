"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _score(centers):
    """Compute the radius score: min of half min-distance and min boundary distance."""
    if centers.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary constraint: centers must be inside [-0.5, 0.5]^2
    boundary_margin = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary_margin)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    r = min(min_dist / 2.0, min_boundary)
    return r

def _initial_hex_grid(n):
    """Generate a high-quality hexagonal grid initialization."""
    r_est = 0.038
    dx = 2 * r_est
    dy = dx * np.sqrt(3) / 2
    
    best_points = None
    best_score = 0
    
    for n_rows in range(12, 18):
        n_cols = int(np.ceil(n / n_rows))
        if n_rows * n_cols < n:
            continue
            
        points = []
        for row in range(n_rows):
            y = row * dy
            offset = 0 if row % 2 == 0 else dx / 2
            for col in range(n_cols):
                x = col * dx + offset
                points.append((x, y))
        
        points = np.array(points[:n])
        
        # Center the points
        center = points.mean(axis=0)
        points -= center
        
        # Scale to fit in [-0.5, 0.5]^2 with some margin
        max_extent = np.max(np.abs(points))
        if max_extent > 0:
            scale = 0.45 / max_extent
            points *= scale
            
        if np.all(np.abs(points) <= 0.5 - 1e-10):
            score = _score(points)
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
    if best_points is not None:
        return best_points
        
    # Fallback: Square grid
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _polish(centers, deadline, init_step=0.001, min_step=1e-8):
    """
    High-precision polish: iteratively alleviate the current bottleneck
    (closest pair or boundary) with decreasing step size.
    """
    centers = centers.copy()
    best_r = _score(centers)
    best_centers = centers.copy()
    step = init_step
    stagnant = 0
    
    while time.time() < deadline:
        if stagnant > 1000:
            step *= 0.5
            if step < min_step:
                break
            stagnant = 0
        
        # Find closest pair
        tree = cKDTree(centers)
        d, idx = tree.query(centers, k=2)
        min_d = d[:, 1].min()
        
        # Boundary
        bm = 0.5 - np.max(np.abs(centers), axis=1)
        min_bm = bm.min()
        
        r = min(min_d / 2.0, min_bm)
        if r > best_r + 1e-14:
            best_r = r
            best_centers = centers.copy()
            stagnant = 0
        else:
            stagnant += 1
        
        # Fix bottleneck
        if min_d / 2.0 < min_bm:
            i = d[:, 1].argmin()
            j = idx[i, 1]
            diff = centers[i] - centers[j]
            nrm = np.linalg.norm(diff)
            if nrm < 1e-12:
                k = np.random.normal(0, step, 2)
                centers[i] += k
                centers[j] -= k
            else:
                dv = diff / nrm
                centers[i] += dv * step
                centers[j] -= dv * step
        else:
            idx_b = bm.argmin()
            for k in range(2):
                if centers[idx_b, k] > 0:
                    centers[idx_b, k] -= step
                else:
                    centers[idx_b, k] += step
        
        centers = np.clip(centers, -0.5, 0.5)
        
        # Escape local minima
        if stagnant % 200 == 0 and step > min_step * 100:
            pi = np.random.randint(N)
            centers[pi] += np.random.normal(0, step * 3, 2)
            centers = np.clip(centers, -0.5, 0.5)
        
        step *= 0.9995
    
    return best_centers, best_r

def _anneal(centers, deadline):
    """
    Simulated annealing: random moves with Metropolis acceptance on hard-min score.
    Periodically run polish to escape local minima.
    """
    centers = centers.copy()
    cur_r = _score(centers)
    best_r = cur_r
    best_c = centers.copy()
    
    T = 5e-5
    T_min = 1e-9
    cool = 0.9998
    base_step = 0.015
    
    it = 0
    while time.time() < deadline:
        it += 1
        i = np.random.randint(N)
        old = centers[i].copy()
        s = base_step * (0.3 + 0.7 * T / 5e-5)
        centers[i] += np.random.normal(0, s, 2)
        centers[i] = np.clip(centers[i], -0.5, 0.5)
        
        new_r = _score(centers)
        dr = new_r - cur_r
        if dr >= 0 or np.random.random() < np.exp(dr / T):
            cur_r = new_r
            if cur_r > best_r:
                best_r = cur_r
                best_c = centers.copy()
        else:
            centers[i] = old
        
        T = max(T_min, T * cool)
        
        # Periodic polish to escape local minima
        if it % 300 == 0 and time.time() < deadline - 3:
            pc, pr = _polish(centers, time.time() + 1.5, init_step=0.0005)
            if pr > best_r:
                best_r = pr
                best_c = pc.copy()
                centers = pc.copy()
                cur_r = pr
    
    return best_c, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 142  # 8s safety margin
    
    # 1. Initialize
    best_centers = _initial_hex_grid(N)
    best_score = _score(best_centers)
    
    # 2. Anneal
    if time.time() < deadline - 20:
        c, s = _anneal(best_centers, deadline - 20)
        if s > best_score:
            best_score = s
            best_centers = c
    
    # 3. Final polish
    if time.time() < deadline - 3:
        c, s = _polish(best_centers, deadline - 1, init_step=0.0003, min_step=1e-9)
        if s > best_score:
            best_score = s
            best_centers = c
    
    return best_centers
# EVOLVE-BLOCK-END
