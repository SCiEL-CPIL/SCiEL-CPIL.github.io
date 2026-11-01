"""Structural Change: Simulated Annealing for Global Search + Nelder-Mead for Local Polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

# EVOLVE-BLOCK-START

def _compute_score(points):
    """Compute the minimum radius for a given set of points."""
    n = len(points)
    if n < 2:
        return max(0.0, 1.0 - np.linalg.norm(points[0])) if n == 1 else 0.0
    
    # Efficiently compute pairwise distances
    # Using broadcasting might be memory intensive for N=300 (90k pairs), but acceptable.
    # 300x300 matrix is 90,000 elements, which is small.
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    
    # Set diagonal to infinity to ignore self-distance
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists) / 2.0
    
    # Boundary distance
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    
    return min(min_pair_dist, min_boundary)

def _hex_lattice_init(n):
    """Initialize points in a hexagonal lattice clipped to the unit circle."""
    best_pts = None
    best_score = -1.0
    for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
        # Estimate spacing
        # Area per circle ~ pi * (r)^2. Total area ~ pi * 1^2.
        # r ~ 1/sqrt(n). Spacing ~ 2r.
        spacing = 2.0 / (np.sqrt(n) * scale)
        
        nx = int(2.0 / spacing) + 5
        ny = int(2.0 / (spacing * np.sqrt(3.0) / 2.0)) + 5
        
        x_coords = (np.arange(-nx // 2, nx // 2 + 1) + 0.5) * spacing
        y_coords = (np.arange(-ny // 2, ny // 2 + 1)) * spacing * np.sqrt(3.0) / 2.0
        
        pts_list = []
        for i, y in enumerate(y_coords):
            offset = 0.0 if i % 2 == 0 else spacing / 2.0
            for x in x_coords:
                px = x + offset
                py = y
                if np.hypot(px, py) <= 1.0 - spacing / 2.0:
                    pts_list.append([px, py])
        
        if len(pts_list) < n:
            continue
            
        pts = np.array(pts_list)
        # Keep the n points closest to the center
        norms = np.linalg.norm(pts, axis=1)
        idx = np.argsort(norms)
        pts = pts[idx[:n]]
        
        score = _compute_score(pts)
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
            
    if best_pts is None:
        # Fallback random
        best_pts = np.random.rand(n, 2) * 2 - 1
        norms = np.linalg.norm(best_pts, axis=1)
        best_pts = best_pts * (0.9 / norms)[:, None]
        
    return best_pts

def _ring_init(n):
    """Initialize points in concentric rings."""
    points = [[0.0, 0.0]]
    remaining = n - 1
    num_rings = int(np.sqrt(n) / 2) + 2
    
    # Distribute points in rings
    # Approximate radius of ring k
    radii = []
    for k in range(1, num_rings + 1):
        r = 0.95 * k / num_rings
        radii.append(r)
        
    # Estimate count per ring based on circumference
    circumferences = [2 * np.pi * r for r in radii]
    total_circ = sum(circumferences)
    
    counts = []
    for c in circumferences:
        count = int(remaining * c / total_circ)
        counts.append(max(1, count))
        
    # Adjust to match remaining
    diff = remaining - sum(counts)
    if diff > 0:
        for i in range(diff):
            counts[i % len(counts)] += 1
    elif diff < 0:
        for i in range(abs(diff)):
            if counts[i % len(counts)] > 1:
                counts[i % len(counts)] -= 1
                
    for k, (count, r) in enumerate(zip(counts, radii)):
        if count == 0: continue
        d_theta = 2 * np.pi / count
        # Stagger rings
        offset = np.pi / count * (k % 2)
        for j in range(count):
            theta = j * d_theta + offset
            points.append([r * np.cos(theta), r * np.sin(theta)])
            
    points = np.array(points[:n])
    return points

def _sa_optimize(points, time_limit):
    """Simulated Annealing to maximize the score."""
    current_pts = points.copy()
    current_score = _compute_score(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 1e-4 # Initial temperature (scale of score improvement)
    T_min = 1e-8
    cooling_rate = 0.9999
    
    # Precompute indices for efficiency if needed, but _compute_score is fast enough for N=300
    # 300 points -> 45k pairs. Vectorized ops are fast.
    
    while time.time() - start_time < time_limit:
        # Pick a random point to move
        i = np.random.randint(0, N)
        # Random perturbation
        # Step size proportional to current radius? Or fixed?
        # Let's use a step size that shrinks with T
        step = 1e-3 * (T / 1e-4) 
        # Actually, better to have a fixed exploration step and let T handle acceptance
        # Or adaptive step. Let's use a small random step.
        dx = np.random.normal(0, 1e-3, 2)
        
        new_pts = current_pts.copy()
        new_pts[i] += dx
        
        # Check bounds immediately to reject early
        if np.linalg.norm(new_pts[i]) > 1.0:
            continue
            
        new_score = _compute_score(new_pts)
        
        if new_score > current_score:
            current_pts = new_pts
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = new_pts.copy()
        else:
            # Metropolis acceptance
            # delta is negative
            delta = new_score - current_score
            if T > 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_pts = new_pts
                    current_score = new_score
        
        T *= cooling_rate
        
    return best_pts, best_score

def construct():
    start_time = time.time()
    deadline = 140.0 # Safety margin
    
    # Generate diverse initializations
    inits = []
    
    hex_pts = _hex_lattice_init(N)
    inits.append(hex_pts)
    
    ring_pts = _ring_init(N)
    inits.append(ring_pts)
    
    # Random init
    rand_pts = np.random.rand(N, 2) * 2 - 1
    norms = np.linalg.norm(rand_pts, axis=1)
    rand_pts = rand_pts * (0.95 / norms)[:, None]
    inits.append(rand_pts)
    
    best_overall_pts = None
    best_overall_score = -1.0
    
    # Allocate time for SA on each init
    # We have ~140s. Let's give each init 30s of SA.
    # Then use remaining time for polishing the best one.
    
    num_inits = len(inits)
    sa_time_per_init = 20.0 # 20s each, total 60s
    
    for init in inits:
        if time.time() > deadline - 30: # Leave 30s for polish
            break
            
        pts, score = _sa_optimize(init, sa_time_per_init)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_pts = pts
    
    # Polish the best result found using Nelder-Mead
    if best_overall_pts is not None and time.time() < deadline - 5:
        def neg_score(x):
            p = x.reshape(-1, 2)
            # Ensure validity
            if not np.all(np.isfinite(p)):
                return 1e6
            return -_compute_score(p)
        
        try:
            res = minimize(neg_score, best_overall_pts, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                    'adaptive': True})
            if res.fun < -best_overall_score:
                best_overall_pts = res.x.reshape(-1, 2)
                best_overall_score = -res.fun
        except Exception:
            pass
        
        # One more polish with tighter tolerances if time permits
        if time.time() < deadline - 2:
            try:
                res2 = minimize(neg_score, best_overall_pts, method='Nelder-Mead',
                                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                                         'adaptive': True})
                if res2.fun < -best_overall_score:
                    best_overall_pts = res2.x.reshape(-1, 2)
                    best_overall_score = -res2.fun
            except Exception:
                pass
    
    # Final validation and clipping
    if best_overall_pts is None:
        # Fallback
        best_overall_pts = hex_pts
        
    best_overall_pts = best_overall_pts.reshape(-1, 2)
    norms = np.linalg.norm(best_overall_pts, axis=1)
    mask = norms > 1.0
    if np.any(mask):
        best_overall_pts[mask] *= (1.0 / norms[mask] - 0.001)[:, np.newaxis]
        
    if not np.all(np.isfinite(best_overall_pts)):
        return hex_pts
        
    return best_overall_pts

# EVOLVE-BLOCK-END
