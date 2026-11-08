"""Improved construction for kissing number in dimension 11 using greedy independent set on conflict graph."""
import time
import numpy as np
from itertools import combinations, product

D = 11

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 130  # safety margin from 150s limit

    # Generate all candidate vectors: weight 2 and weight 3 integer vectors
    # with entries in {-1, 0, 1}
    
    # Weight-2 vectors: C(11,2) * 4 = 220
    # Weight-3 vectors: C(11,3) * 8 = 165 * 8 = 1320
    # Total candidates: 1540
    
    candidates = []
    
    # Weight 2
    for i in range(D):
        for j in range(i + 1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    
    # Weight 3
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si, sj, sk in product([1, -1], repeat=3):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    candidates.append(v)
    
    n_cand = len(candidates)
    
    # For each candidate, compute its norm squared
    norms_sq = np.array([np.dot(v, v) for v in candidates], dtype=np.float64)
    
    # We need: for all pairs (i,j), ||v_i - v_j|| >= max_norm
    # Equivalent: ||v_i - v_j||^2 >= max_norm^2
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    # So we need: ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j >= max_norm^2
    
    # Strategy: greedy approach
    # 1. Start with all candidates
    # 2. The max norm is determined by the largest norm vector included
    # 3. We want to maximize count while maintaining the constraint
    
    # Key insight: if we include a vector of norm^2 = N_max, then for any other vector v with norm^2 = N_v:
    # We need N_v + N_max - 2*v_i.v_j >= N_max, i.e., N_v >= 2*v_i.v_j
    # And for two vectors both with norm^2 = N_max: N_max >= 2*v_i.v_j, i.e., v_i.v_j <= N_max/2
    
    # Since weight-2 vectors have norm^2=2 and weight-3 have norm^2=3:
    # If max norm^2 = 3: need v_i.v_j <= 1.5 for weight-3 pairs, and v_i.v_j <= (N_v)/2 for mixed
    # For weight-2 vs weight-3: need 2 + 3 - 2*v_i.v_j >= 3, i.e., v_i.v_j <= 1
    # For weight-3 vs weight-3: need 3 + 3 - 2*v_i.v_j >= 3, i.e., v_i.v_j <= 1.5, so v_i.v_j <= 1 (integer)
    # For weight-2 vs weight-2: need 2 + 2 - 2*v_i.v_j >= 3, i.e., v_i.v_j <= 0.5, so v_i.v_j <= 0
    
    # Hmm, this is more restrictive. Let me reconsider.
    
    # Actually, the constraint is: min pairwise distance >= max vector norm
    # If max norm^2 = 3 (from weight-3 vectors), then min pairwise distance must be >= sqrt(3)
    # So min pairwise distance^2 >= 3
    
    # For two weight-2 vectors: dist^2 = 2 + 2 - 2*dot = 4 - 2*dot >= 3 => dot <= 0.5 => dot <= 0
    # For weight-2 and weight-3: dist^2 = 2 + 3 - 2*dot = 5 - 2*dot >= 3 => dot <= 1
    # For two weight-3: dist^2 = 3 + 3 - 2*dot = 6 - 2*dot >= 3 => dot <= 1.5 => dot <= 1
    
    # So if we include weight-3 vectors, weight-2 vectors can only have dot product 0 with each other.
    # This is very restrictive for weight-2 vectors.
    
    # Alternative: use only weight-2 vectors (max norm^2 = 2), then need dist^2 >= 2
    # For two weight-2: 4 - 2*dot >= 2 => dot <= 1. This is what we had before (220 vectors).
    
    # Let me try a different approach: use vectors of various norms but be careful.
    
    # Actually, let's try: include all weight-2 vectors (220) plus some weight-3 vectors.
    # The max norm would be sqrt(3). For weight-2 pairs, we need dot <= 0.
    # But in our current set of 220 weight-2 vectors, many pairs have dot = 1.
    # So we can't keep all 220 weight-2 vectors if we add weight-3 vectors.
    
    # Let me try a pure weight-3 approach or a mixed approach with greedy.
    
    # Better approach: Greedy independent set on the conflict graph.
    # Two vectors conflict if their distance is less than the max norm of the set.
    # But the max norm depends on what we include...
    
    # Simpler approach: Fix the max norm. Try max_norm_sq = 2 (weight-2 only) or 3 (allow weight-3).
    
    # For max_norm_sq = 2: We can include weight-2 vectors where all pairs have dot <= 1.
    # All 220 weight-2 vectors satisfy this. Score = 220.
    
    # For max_norm_sq = 3: We can include weight-2 and weight-3 vectors.
    # Constraints:
    # - weight-2 vs weight-2: dot <= 0
    # - weight-2 vs weight-3: dot <= 1  
    # - weight-3 vs weight-3: dot <= 1
    
    # This is a maximum independent set problem on a conflict graph.
    # Let's build the conflict graph and use greedy.
    
    # Build conflict graph for max_norm_sq = 3
    # Two vectors conflict if their distance^2 < 3, i.e., dist^2 <= 2
    
    # dist^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j
    # For weight-2, weight-2: 4 - 2*dot <= 2 => dot >= 1 (conflict if dot >= 1)
    # For weight-2, weight-3: 5 - 2*dot <= 2 => dot >= 1.5 => dot >= 2 (conflict if dot >= 2)
    # For weight-3, weight-3: 6 - 2*dot <= 2 => dot >= 2 (conflict if dot >= 2)
    
    # Wait, that doesn't seem right. Let me re-derive.
    # We need dist^2 >= max_norm_sq = 3 for all pairs.
    # Conflict if dist^2 < 3, i.e., dist^2 <= 2 (since integer).
    
    # weight-2, weight-2: dist^2 = 4 - 2*dot. Conflict if 4 - 2*dot <= 2 => dot >= 1.
    # weight-2, weight-3: dist^2 = 5 - 2*dot. Conflict if 5 - 2*dot <= 2 => dot >= 1.5 => dot >= 2.
    # weight-3, weight-3: dist^2 = 6 - 2*dot. Conflict if 6 - 2*dot <= 2 => dot >= 2.
    
    # So conflicts:
    # - Two weight-2 vectors conflict if dot >= 1
    # - Weight-2 and weight-3 conflict if dot >= 2
    # - Two weight-3 vectors conflict if dot >= 2
    
    # Let's compute the dot product matrix and build the conflict graph.
    
    cand_array = np.array(candidates, dtype=np.int64)
    dot_matrix = cand_array @ cand_array.T
    
    # Build conflict matrix
    # conflict[i,j] = True if vectors i and j cannot both be in the set
    
    conflict = np.zeros((n_cand, n_cand), dtype=bool)
    
    # For efficiency, compute conflicts based on norm types
    is_w2 = norms_sq == 2
    is_w3 = norms_sq == 3
    
    # weight-2 vs weight-2: conflict if dot >= 1
    w2_indices = np.where(is_w2)[0]
    w3_indices = np.where(is_w3)[0]
    
    # w2-w2 conflicts
    if len(w2_indices) > 0:
        w2_dots = dot_matrix[np.ix_(w2_indices, w2_indices)]
        w2_conflicts = w2_dots >= 1
        np.fill_diagonal(w2_conflicts, False)  # no self-conflict
        conflict[np.ix_(w2_indices, w2_indices)] = w2_conflicts
    
    # w2-w3 conflicts
    if len(w2_indices) > 0 and len(w3_indices) > 0:
        w2w3_dots = dot_matrix[np.ix_(w2_indices, w3_indices)]
        w2w3_conflicts = w2w3_dots >= 2
        conflict[np.ix_(w2_indices, w3_indices)] = w2w3_conflicts
        conflict[np.ix_(w3_indices, w2_indices)] = w2w3_conflicts.T
    
    # w3-w3 conflicts
    if len(w3_indices) > 0:
        w3_dots = dot_matrix[np.ix_(w3_indices, w3_indices)]
        w3_conflicts = w3_dots >= 2
        np.fill_diagonal(w3_conflicts, False)
        conflict[np.ix_(w3_indices, w3_indices)] = w3_conflicts
    
    # Greedy: repeatedly pick the vertex with minimum degree, add to independent set, remove it and its neighbors
    in_set = np.zeros(n_cand, dtype=bool)
    remaining = np.ones(n_cand, dtype=bool)
    
    while np.any(remaining):
        if time.time() > deadline:
            break
        
        # Compute degrees within remaining set
        remaining_conflict = conflict[np.ix_(remaining.nonzero()[0], remaining.nonzero()[0])]
        degrees = remaining_conflict.sum(axis=1)
        
        # Pick vertex with minimum degree
        min_idx = np.argmin(degrees)
        chosen = remaining.nonzero()[0][min_idx]
        
        in_set[chosen] = True
        remaining[chosen] = False
        remaining[conflict[chosen] & remaining] = False
    
    result_indices = np.where(in_set)[0]
    result = cand_array[result_indices]
    
    # Verify the solution
    if len(result) > 1:
        result_float = result.astype(np.float64)
        norms = np.sqrt(np.sum(result_float ** 2, axis=1))
        max_norm = np.max(norms)
        
        # Compute all pairwise distances
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                dist = np.sqrt(np.sum((result_float[i] - result_float[j]) ** 2))
                if dist < max_norm - 1e-9:
                    # Invalid, fall back to weight-2 only
                    vectors = []
                    for i in range(D):
                        for j in range(i + 1, D):
                            for si in [1, -1]:
                                for sj in [1, -1]:
                                    v = [0] * D
                                    v[i] = si
                                    v[j] = sj
                                    vectors.append(v)
                    return np.array(vectors, dtype=np.int64)
    
    return result
# EVOLVE-BLOCK-END
