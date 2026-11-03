"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time
import itertools
import random

# Triangle vertices
V1 = np.array([0.0, 0.0])
V2 = np.array([1.0, 0.0])
V3 = np.array([0.5, np.sqrt(3)/2])

def triangle_area(p1, p2, p3):
    return 0.5 * np.abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]))

def soft_min(values, k):
    # Stable log-sum-exp approximation of min
    m = np.min(values)
    # If k is large, exp(-k*(a-m)) can underflow for large a-m, which is fine (becomes 0)
    # But if a-m is negative, it overflows. However a-m >= 0 since m is min.
    # So exp(-k * positive) is in (0, 1].
    # Sum is <= len(values).
    # log is safe.
    return m - (1.0/k) * np.log(np.sum(np.exp(-k * (values - m))))

def hard_min(values):
    return np.min(values)

def objective_flat(x, k, penalty_weight, use_soft=True):
    P = x.reshape(N, 2)
    
    # Penalty for being outside the triangle
    # 1. y >= 0
    pen1 = np.maximum(0.0, -P[:,1])
    # 2. y <= sqrt(3) * x
    pen2 = np.maximum(0.0, P[:,1] - np.sqrt(3)*P[:,0])
    # 3. sqrt(3) * x + y <= sqrt(3)
    pen3 = np.maximum(0.0, np.sqrt(3)*P[:,0] + P[:,1] - np.sqrt(3))
    
    penalty = penalty_weight * np.sum(pen1**2 + pen2**2 + pen3**2)
    
    # Calculate all triangle areas
    areas = []
    for i, j, k_idx in itertools.combinations(range(N), 3):
        a = triangle_area(P[i], P[j], P[k_idx])
        areas.append(a)
    areas = np.array(areas)
    
    if use_soft:
        val = soft_min(areas, k)
    else:
        val = hard_min(areas)
        
    return -val + penalty

def is_valid(P, tol=1e-5):
    # Check if points are inside triangle
    if np.any(P[:, 1] < -tol): return False
    if np.any(P[:, 1] > np.sqrt(3)*P[:, 0] + tol): return False
    if np.any(np.sqrt(3)*P[:, 0] + P[:, 1] > np.sqrt(3) + tol): return False
    return True

def get_score(P):
    if not is_valid(P):
        return -np.inf
    areas = [triangle_area(P[i], P[j], P[k]) for i, j, k in itertools.combinations(range(N), 3)]
    return np.min(areas)

def construct():
    start_time = time.time()
    deadline = 140.0 # 10s margin for 150s limit
    
    # 1. Generate Initial Starts
    starts = []
    rng = random.Random(7)
    
    # Start 1: Random
    pts = []
    for _ in range(N):
        u, v = rng.random(), rng.random()
        if u + v > 1:
            u, v = 1-u, 1-v
        p = u * V1 + v * V2 + (1-u-v) * V3
        pts.append(p)
    starts.append(np.array(pts).flatten())
    
    # Start 2: Another Random
    pts = []
    for _ in range(N):
        u, v = rng.random(), rng.random()
        if u + v > 1:
            u, v = 1-u, 1-v
        p = u * V1 + v * V2 + (1-u-v) * V3
        pts.append(p)
    starts.append(np.array(pts).flatten())
    
    # Start 3: Lattice-like
    # Generate a grid in barycentric coordinates
    # Side length 5 gives 15 points. We take 11.
    # We want points spread out.
    lattice_pts = []
    step = 1.0 / 5.0
    for i in range(6):
        for j in range(6 - i):
            # Barycentric (i/5, j/5, (5-i-j)/5)
            u = i * step
            v = j * step
            w = 1.0 - u - v
            if w < -1e-9: continue
            p = u * V1 + v * V2 + w * V3
            lattice_pts.append(p)
    
    # We have 15 points. Pick 11 that maximize min distance? 
    # Or just take the first 11? The first 11 are clustered near V1.
    # Better to pick evenly from the lattice.
    # Sort by distance from center?
    center = (V1 + V2 + V3) / 3.0
    lattice_pts.sort(key=lambda p: np.linalg.norm(p - center))
    # Take 11 points that are spread.
    # Simple heuristic: take every k-th point from sorted list?
    # Or just take 11 random from the lattice?
    # Let's take the 11 points that are furthest from each other? Too complex.
    # Let's just take 11 points from the lattice, ensuring some diversity.
    # The lattice points are:
    # Row 0: 1
    # Row 1: 2
    # Row 2: 3
    # Row 3: 4
    # Row 4: 5
    # Total 15.
    # If we take Row 0-3 (10 points) + 1 from Row 4, we have a good base.
    # Let's take the first 10 (rows 0-3) and one from row 4.
    # Indices in lattice_pts (generated row by row? No, nested loop i, j)
    # i=0: j=0..5 (6 pts)
    # i=1: j=0..4 (5 pts)
    # ...
    # This generation is not row-by-row in the geometric sense of "distance from V1".
    # Let's just pick 11 random points from the lattice for Start 3.
    if len(lattice_pts) >= N:
        idx = rng.sample(range(len(lattice_pts)), N)
        starts.append(np.array([lattice_pts[i] for i in idx]).flatten())
    else:
        starts.append(starts[0].copy())

    best_x = None
    best_score = -np.inf
    
    # Phase 1: Soft-Min Homotopy
    k_values = [10, 50, 200]
    
    current_starts = starts
    
    for k in k_values:
        if time.time() > deadline - 30:
            break
            
        # Optimize from each current start
        new_starts = []
        
        for x0 in current_starts:
            if time.time() > deadline - 10:
                break
                
            try:
                res = scipy.optimize.minimize(
                    objective_flat, x0, args=(k, 100.0, True), method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-5, 'fatol': 1e-5}
                )
                P = res.x.reshape(N, 2)
                if is_valid(P):
                    score = get_score(P)
                    if score > best_score:
                        best_score = score
                        best_x = res.x
                # Add to new starts for next k
                new_starts.append(res.x)
            except Exception:
                pass
        
        # Keep top 3 starts based on score for next iteration
        if new_starts:
            # Sort by score
            scored_starts = []
            for x in new_starts:
                P = x.reshape(N, 2)
                if is_valid(P):
                    s = get_score(P)
                else:
                    s = -np.inf
                scored_starts.append((s, x))
            
            scored_starts.sort(key=lambda item: item[0], reverse=True)
            current_starts = [x for s, x in scored_starts[:3]]
        else:
            current_starts = [best_x] if best_x is not None else [starts[0]]

    # Phase 2: Hard-Min Polish
    if best_x is not None:
        if time.time() < deadline - 5:
            try:
                res = scipy.optimize.minimize(
                    objective_flat, best_x, args=(10, 100.0, False), method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
                )
                P = res.x.reshape(N, 2)
                if is_valid(P):
                    score = get_score(P)
                    if score > best_score:
                        best_score = score
                        best_x = res.x
            except Exception:
                pass

    # Final Check and Return
    if best_x is None:
        # Fallback to random
        pts = []
        for _ in range(N):
            u, v = rng.random(), rng.random()
            if u + v > 1:
                u, v = 1-u, 1-v
            p = u * V1 + v * V2 + (1-u-v) * V3
            pts.append(p)
        return np.array(pts)
    
    return best_x.reshape(N, 2)

# EVOLVE-BLOCK-END
