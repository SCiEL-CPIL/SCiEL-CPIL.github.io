import numpy as np
import time
import itertools
from collections import deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    D = 11

    # Generate all weight-2 vectors (minimal vectors of D_11)
    w2_vecs = []
    for i in range(D):
        for j in range(i + 1, D):
            for s_i in (-1, 1):
                for s_j in (-1, 1):
                    if time.time() > deadline - 10.0:
                        return np.array(w2_vecs, dtype=np.int8) if w2_vecs else np.zeros((0, D), dtype=np.int8)
                    v = np.zeros(D, dtype=np.int8)
                    v[i] = s_i
                    v[j] = s_j
                    w2_vecs.append(v)
    
    w2_arr = np.array(w2_vecs, dtype=np.int8)
    n_w2 = len(w2_arr)
    
    # Generate all weight-3 vectors
    w3_vecs = []
    for indices in itertools.combinations(range(D), 3):
        for signs in itertools.product([-1, 1], repeat=3):
            if time.time() > deadline - 10.0:
                break
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(indices, signs):
                v[idx] = s
            w3_vecs.append(v)
        if time.time() > deadline - 10.0:
            break
    
    if not w3_vecs:
        return w2_arr
    
    w3_arr = np.array(w3_vecs, dtype=np.int8)
    n_w3 = len(w3_arr)
    
    # Compute conflict graph between w2 and w3 vectors
    # Two vectors conflict if their distance squared is less than max_norm_sq
    # For w2-w2: norm_sq = 2, dist_sq >= 2 (min is 2 when they share one index with same sign)
    # For w3-w3: norm_sq = 3, dist_sq >= 3 (min is 3 when they share two indices)
    # For w2-w3: dist_sq = 2 + 3 - 2*dot = 5 - 2*dot
    #   Conflict if dist_sq < max_norm_sq. If max is 3, conflict if dist_sq < 3 => 5-2*dot < 3 => dot > 1
    #   But dot is integer, so dot >= 2. Max dot for w2.w3 is 2 (when w2 is subset of w3 with same signs).
    #   So conflict if dot >= 2.
    
    # Actually, let's think about this more carefully.
    # The constraint is: min pairwise distance >= max vector norm.
    # If we select a set S, let R = max_{v in S} ||v||.
    # We need min_{u,v in S, u!=v} ||u-v|| >= R.
    
    # Case 1: Only w2 vectors. R = sqrt(2). Min dist = sqrt(2). Valid. Count = 220.
    # Case 2: Only w3 vectors. R = sqrt(3). Min dist between w3 vectors:
    #   Two w3 vectors can share 0, 1, or 2 indices.
    #   If they share 2 indices: dist_sq = 3+3-2*dot. If signs match on shared, dot=2, dist_sq=2 < 3. Conflict!
    #   So we need a subset of w3 vectors where no two share 2+ indices with matching signs.
    # Case 3: Mix of w2 and w3. R = sqrt(3) (since w3 has higher norm).
    #   Need all pairwise dist >= sqrt(3).
    #   w2-w2: min dist = sqrt(2) < sqrt(3). So we can't have ANY two w2 vectors if R=sqrt(3)!
    #   Wait, that's not right. We need min dist >= R = sqrt(3).
    #   But w2-w2 min dist is sqrt(2) < sqrt(3). So if we include any w3, we can include AT MOST one w2?
    #   No, we can include w2 vectors, but they must be at least sqrt(3) apart from each other AND from w3.
    #   Two w2 vectors: dist_sq = 2+2-2*dot = 4-2*dot. Need >= 3 => dot <= 0.5 => dot <= 0.
    #   So w2 vectors must have dot <= 0. This means they can't share any index (or share with opposite signs).
    #   w2-w3: dist_sq = 2+3-2*dot = 5-2*dot. Need >= 3 => dot <= 1.
    #   w3-w3: dist_sq = 3+3-2*dot = 6-2*dot. Need >= 3 => dot <= 1.5 => dot <= 1.
    
    # So the problem is: select a subset of {w2} U {w3} such that:
    # - For any two selected w2: dot <= 0
    # - For any two selected w3: dot <= 1
    # - For any selected w2 and w3: dot <= 1
    
    # This is a complex constraint satisfaction problem. Let's try a greedy approach with local search.
    
    # First, let's see if we can do better than 220 by using w3 vectors.
    # Max independent set in the conflict graph where edges represent conflicts.
    
    # Build conflict matrix for w3-w3 conflicts (dot > 1, i.e., dot >= 2)
    # And w2-w3 conflicts (dot > 1, i.e., dot >= 2)
    # And w2-w2 conflicts (dot > 0, i.e., dot >= 1)
    
    # Let's use a simpler approach: start with all w2, try to add w3 by removing conflicting w2.
    
    # Actually, let's try a different strategy: use only w3 vectors and find a large independent set.
    # Two w3 vectors conflict if their dot product > 1 (i.e., >= 2).
    
    # Compute w3-w3 dot products
    # This is a 1320x1320 matrix, which is manageable.
    
    w3_dots = w3_arr @ w3_arr.T
    # Conflict if dot > 1 (i.e., dot >= 2)
    w3_conflicts = (w3_dots > 1)
    np.fill_diagonal(w3_conflicts, False)
    
    # Greedy independent set on w3
    selected_w3 = np.zeros(n_w3, dtype=bool)
    remaining = set(range(n_w3))
    
    while remaining:
        if time.time() > deadline - 5.0:
            break
        # Pick the vertex with minimum degree among remaining
        best_v = min(remaining, key=lambda v: sum(w3_conflicts[v, u] for u in remaining if u != v))
        selected_w3[best_v] = True
        # Remove best_v and all its neighbors from remaining
        to_remove = {best_v} | {u for u in remaining if u != best_v and w3_conflicts[best_v, u]}
        remaining -= to_remove
    
    # Now try to add w2 vectors that are compatible with selected_w3
    # A w2 vector is compatible with selected_w3 if dot <= 1 with all selected w3
    selected_w2 = np.zeros(n_w2, dtype=bool)
    
    for i in range(n_w2):
        if time.time() > deadline - 5.0:
            break
        # Check compatibility with selected w3
        if np.any(selected_w3):
            dots_with_w3 = w2_arr[i] @ w3_arr[selected_w3]
            if np.any(dots_with_w3 > 1):
                continue
        
        # Check compatibility with already selected w2
        if np.any(selected_w2):
            dots_with_w2 = w2_arr[i] @ w2_arr[selected_w2]
            if np.any(dots_with_w2 > 0):
                continue
        
        selected_w2[i] = True
    
    # Count selected
    n_selected_w2 = np.sum(selected_w2)
    n_selected_w3 = np.sum(selected_w3)
    total_selected = n_selected_w2 + n_selected_w3
    
    # Compare with seed (220 w2 vectors)
    if total_selected > 220:
        # Construct the result
        result = np.vstack([w2_arr[selected_w2], w3_arr[selected_w3]]) if n_selected_w2 > 0 and n_selected_w3 > 0 else \
                 w2_arr[selected_w2] if n_selected_w2 > 0 else \
                 w3_arr[selected_w3]
        return result
    else:
        return w2_arr
