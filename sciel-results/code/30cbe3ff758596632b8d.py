"""Improved program using greedy set construction for specific target areas."""
import numpy as np
from collections import Counter
import random
from itertools import combinations

def construct():
    """
    Find 30 distinct integer coordinate pairs in [-30,30]^2 that maximize
    the largest multiplicity of a single positive doubled area among all triples.
    
    Approach:
    1. Generate candidate target areas from structured lattice patterns.
    2. For each target area, use a greedy algorithm to select 30 points
       that maximize the number of triples with that specific area.
    3. Return the best set found.
    """
    
    # Generate the full pool of valid points
    pool = [(x, y) for x in range(-30, 31) for y in range(-30, 31)]
    pool_array = np.array(pool, dtype=np.int64)
    M = len(pool_array)
    
    # Helper to compute max multiplicity for a given set of points
    def compute_max_multiplicity(points):
        n = len(points)
        if n < 3: return 0
        pts = np.array(points, dtype=np.int64)
        
        # Vectorized area calculation for all triples
        idx_i, idx_j, idx_k = [], [], []
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    idx_i.append(i)
                    idx_j.append(j)
                    idx_k.append(k)
        
        idx_i = np.array(idx_i)
        idx_j = np.array(idx_j)
        idx_k = np.array(idx_k)
        
        xi = pts[idx_i, 0]
        yi = pts[idx_i, 1]
        xj = pts[idx_j, 0]
        yj = pts[idx_j, 1]
        xk = pts[idx_k, 0]
        yk = pts[idx_k, 1]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        positive_areas = areas[areas > 0]
        
        if len(positive_areas) == 0:
            return 0
            
        counter = Counter(positive_areas.tolist())
        return max(counter.values())

    # Generate seed configurations to identify promising target areas
    seeds = []
    
    # Grid patterns
    for w in [5, 6, 7]:
        for h in [5, 6]:
            if w*h >= 30:
                pts = [(x, y) for x in range(w) for y in range(h)]
                seeds.append(pts[:30])
                
    # Centered grids
    for size in [6, 7]:
        center = 0
        pts = []
        for x in range(-size//2, size//2 + 1):
            for y in range(-size//2, size//2 + 1):
                pts.append((center + x, center + y))
        if len(pts) >= 30:
            seeds.append(pts[:30])
            
    # Diagonal/Hex patterns
    pts_hex = []
    for row in range(6):
        for col in range(6):
            x = col * 2 + (row % 2)
            y = row * 2
            pts_hex.append((x, y))
    if len(pts_hex) >= 30:
        seeds.append(pts_hex[:30])
        
    # Identify top candidate areas
    candidate_areas = set()
    for seed in seeds:
        if len(seed) < 30: continue
        pts = np.array(seed, dtype=np.int64)
        # Compute areas for seed
        idx_i, idx_j, idx_k = [], [], []
        n = 30
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    idx_i.append(i)
                    idx_j.append(j)
                    idx_k.append(k)
        idx_i = np.array(idx_i)
        idx_j = np.array(idx_j)
        idx_k = np.array(idx_k)
        
        xi = pts[idx_i, 0]
        yi = pts[idx_i, 1]
        xj = pts[idx_j, 0]
        yj = pts[idx_j, 1]
        xk = pts[idx_k, 0]
        yk = pts[idx_k, 1]
        
        areas = np.abs(xi * (yj - yk) + xj * (yk - yi) + xk * (yi - yj))
        positive_areas = areas[areas > 0]
        counter = Counter(positive_areas.tolist())
        # Add top 5 areas to candidates
        for area, count in counter.most_common(5):
            candidate_areas.add(area)
            
    # Ensure we have some small areas if none found (fallback)
    if not candidate_areas:
        candidate_areas = {1, 2, 3, 4, 5}
        
    # Greedy Construction for each candidate area
    best_score = 0
    best_points = None
    
    for target_area in candidate_areas:
        # Greedy selection
        S = [] # List of points in the set
        S_indices = [] # Indices in pool_array
        used_mask = np.zeros(M, dtype=bool)
        
        for step in range(30):
            if step == 0:
                # First point: pick (0,0) if available
                idx_00 = np.where((pool_array[:, 0] == 0) & (pool_array[:, 1] == 0))[0]
                if len(idx_00) > 0:
                    p_idx = idx_00[0]
                else:
                    p_idx = 0
                S.append(pool[p_idx])
                S_indices.append(p_idx)
                used_mask[p_idx] = True
                continue
                
            # Calculate gain for each unused point
            gains = np.zeros(M, dtype=np.int32)
            s_arr = np.array(S, dtype=np.int64)
            k = len(S)
            
            for i in range(k):
                for j in range(i+1, k):
                    u = s_arr[i]
                    v = s_arr[j]
                    # Area(u, v, p) for all p in pool
                    a = u[1] - v[1]
                    b = v[0] - u[0]
                    c = u[0]*v[1] - v[0]*u[1]
                    
                    px = pool_array[:, 0]
                    py = pool_array[:, 1]
                    
                    vals = a * px + b * py + c
                    areas_vec = np.abs(vals)
                    
                    match_mask = (areas_vec == target_area)
                    match_mask &= ~used_mask
                    gains += match_mask.astype(np.int32)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                unused_indices = np.where(~used_mask)[0]
                if len(unused_indices) == 0:
                    break
                p_idx = unused_indices[0]
            else:
                candidates = np.where(gains == max_gain)[0]
                p_idx = candidates[0]
                
            S.append(pool[p_idx])
            S_indices.append(p_idx)
            used_mask[p_idx] = True
            
        if len(S) == 30:
            score = compute_max_multiplicity(S)
            if score > best_score:
                best_score = score
                best_points = S[:]
                
    if best_points is None or best_score < 10:
        best_points = [(x, y) for x in range(6) for y in range(5)]
        best_score = compute_max_multiplicity(best_points)
        
    return best_points
