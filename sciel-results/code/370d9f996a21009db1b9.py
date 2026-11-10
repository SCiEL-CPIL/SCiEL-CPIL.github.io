# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates.
    
    Strategy: 
    1. Start with the 2E8 ⊕ 2D3 lattice construction to get 252 points with norm^2 = 8.
    2. Attempt to extend the set by greedily adding additional integer vectors
       with norm^2 >= 8 that satisfy the distance constraint:
       min_dist^2 >= max_norm^2.
       
    The base 252 points all have norm^2 = 8.
    If we add a vector v with norm^2 = N > 8, the constraint becomes:
    For all existing u (norm^2 = 8): ||v-u||^2 >= N
    N + 8 - 2*v.u >= N  =>  8 - 2*v.u >= 0  =>  v.u <= 4.
    
    Also, for any two new vectors v, w with norm^2 = N:
    2N - 2*v.w >= N  =>  N >= 2*v.w.
    
    We will generate candidates with norm^2 = 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    d = 11
    points_list = []
    
    # --- Generate 240 Scaled E8 Vectors (8 dims) ---
    # Type 1: (±2, ±2, 0, 0, 0, 0, 0, 0)
    for i, j in combinations(range(8), 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                v = np.zeros(8, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                p = np.zeros(d, dtype=np.int64)
                p[:8] = v
                points_list.append(p)
                
    # Type 2: (±1, ±1, ±1, ±1, ±1, ±1, ±1, ±1) with even number of minus signs
    # 128 such vectors.
    for signs in product([1, -1], repeat=8):
        if signs.count(-1) % 2 == 0:
            v = np.array(signs, dtype=np.int64)
            p = np.zeros(d, dtype=np.int64)
            p[:8] = v
            points_list.append(p)
            
    # --- Generate 12 Scaled D3 Vectors (3 dims) ---
    # (±2, ±2, 0) permutations in last 3 dims (indices 8, 9, 10)
    for i, j in combinations([8, 9, 10], 2):
        for si in [1, -1]:
            for sj in [1, -1]:
                p = np.zeros(d, dtype=np.int64)
                p[i] = 2 * si
                p[j] = 2 * sj
                points_list.append(p)
    
    points = np.array(points_list, dtype=np.int64)
    base_k = len(points)
    
    # --- Greedy Extension Phase ---
    # Generate candidates with various norms
    candidates = []
    
    # We will generate candidates with norm^2 from 9 to 100
    # For each norm^2 N, we generate all integer vectors with that norm
    # But generating all vectors for high norms is expensive.
    # Instead, we will use a heuristic: 
    # 1. Start with low norms (9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20)
    # 2. For each norm, generate a limited set of candidates
    
    # Helper to generate candidates for a given norm^2
    def generate_candidates_for_norm(norm_sq, max_candidates=10000):
        cands = []
        # Try to find integer solutions to sum(x_i^2) = norm_sq
        # This is a subset sum / partition problem.
        # For simplicity, we will only generate candidates with at most 4 non-zero components.
        
        # 1 non-zero component: x^2 = norm_sq
        if int(np.sqrt(norm_sq))**2 == norm_sq:
            x = int(np.sqrt(norm_sq))
            for i in range(d):
                for s in [1, -1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = x * s
                    cands.append(p)
        
        # 2 non-zero components: x^2 + y^2 = norm_sq
        for x in range(1, int(np.sqrt(norm_sq)) + 1):
            y_sq = norm_sq - x*x
            y = int(np.sqrt(y_sq))
            if y*y == y_sq:
                for i, j in combinations(range(d), 2):
                    for si in [1, -1]:
                        for sj in [1, -1]:
                            p = np.zeros(d, dtype=np.int64)
                            p[i] = x * si
                            p[j] = y * sj
                            cands.append(p)
        
        # 3 non-zero components: x^2 + y^2 + z^2 = norm_sq
        for x in range(1, int(np.sqrt(norm_sq)) + 1):
            rem1 = norm_sq - x*x
            for y in range(1, int(np.sqrt(rem1)) + 1):
                z_sq = rem1 - y*y
                z = int(np.sqrt(z_sq))
                if z*z == z_sq:
                    for i, j, k in combinations(range(d), 3):
                        for si in [1, -1]:
                            for sj in [1, -1]:
                                for sk in [1, -1]:
                                    p = np.zeros(d, dtype=np.int64)
                                    p[i] = x * si
                                    p[j] = y * sj
                                    p[k] = z * sk
                                    cands.append(p)
                                    if len(cands) >= max_candidates:
                                        return cands
        return cands

    # Generate candidates for norms 9 to 30
    for norm_sq in range(9, 31):
        if time.time() > deadline - 10:
            break
        cands = generate_candidates_for_norm(norm_sq, max_candidates=5000)
        candidates.extend(cands)
    
    # Shuffle candidates to break symmetry
    rng = np.random.default_rng(42)
    cand_indices = np.arange(len(candidates))
    rng.shuffle(cand_indices)
    
    added_count = 0
    for idx in cand_indices:
        if time.time() > deadline - 5:
            break
            
        v = candidates[idx]
        norm_v_sq = np.dot(v, v)
        
        # Check against all existing points
        valid = True
        for u in points:
            norm_u_sq = np.dot(u, u)
            dot = np.dot(v, u)
            dist_sq = norm_v_sq + norm_u_sq - 2 * dot
            max_norm_sq = max(norm_v_sq, norm_u_sq)
            if dist_sq < max_norm_sq:
                valid = False
                break
        
        if valid:
            points = np.vstack([points, v])
            added_count += 1
            
    if time.time() > deadline:
         return np.array(points[:2], dtype=np.int64)

    return points.astype(np.int64)

# EVOLVE-BLOCK-END
