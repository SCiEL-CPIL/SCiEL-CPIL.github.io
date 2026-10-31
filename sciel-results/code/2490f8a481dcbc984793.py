"""Program for kissing number in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """Construct nonzero integer vectors in R^11 whose minimum pairwise distance
    is at least the maximum vector norm. Return integer array of shape (k, d)."""
    deadline = time.time() + 145.0  # 5s safety margin

    # Generate all vectors of norm sqrt(2): pairs (i,j) with signs (±1, ±1)
    # These form a kissing configuration with 220 vectors.
    # The key insight: vectors of norm sqrt(2) have pairwise distances of
    # sqrt(2), sqrt(8), or 0. Since max norm is sqrt(2), we need dist >= sqrt(2).
    # Two distinct norm-sqrt(2) vectors: if they share one coordinate, dist = sqrt(2);
    # if they share no coordinates, dist = sqrt(8) = 2*sqrt(2) >= sqrt(2). Good.
    
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)
    
    # Try to add vectors of other norms using a greedy approach with conflict checking
    # A vector v can be added if for all existing u: ||v - u|| >= ||v|| and ||u||
    # i.e., min pairwise dist >= max norm
    
    # Generate candidate vectors of small norm
    candidates = []
    # Norm 1 vectors (unit basis)
    for i in range(D):
        for s in (1, -1):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            candidates.append(v)
    # Norm sqrt(5) vectors: one coord ±1, another ±2
    for i in range(D):
        for j in range(D):
            if i == j:
                continue
            for si in (1, -1, 2, -2):
                for sj in (1, -1):
                    if i < j:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        candidates.append(v)
                    elif i > j:
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = sj
                        v[j] = si
                        candidates.append(v)
    # Norm sqrt(8) vectors: two coords ±2
    for i in range(D):
        for j in range(i + 1, D):
            for si in (2, -2):
                for sj in (2, -2):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)
    # Norm 3 vectors: three coords ±1
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        for sk in (1, -1):
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si
                            v[j] = sj
                            v[k] = sk
                            candidates.append(v)
    
    # Start with the sqrt(2) vectors
    current = np.array(vectors, dtype=np.int64)
    best = current.copy()
    best_count = len(best)
    
    # Compute norms squared
    def compute_min_dist_and_max_norm(arr):
        if len(arr) <= 1:
            return 0, np.max(arr**2, axis=1).max() if len(arr) > 0 else 0
        # Compute pairwise squared distances
        # This is expensive for large arrays, so be careful
        norms_sq = np.sum(arr**2, axis=1)
        max_norm = np.max(norms_sq)
        # Check min pairwise squared distance
        # Use chunked computation to avoid memory issues
        n = len(arr)
        min_dist_sq = np.inf
        for start in range(0, n, 50):
            end = min(start + 50, n)
            chunk = arr[start:end]
            # dist^2 = ||a||^2 + ||b||^2 - 2*a.b
            dots = chunk @ arr.T  # (chunk_size, n)
            dist_sq = norms_sq[start:end, None] + norms_sq[None, :] - 2 * dots
            # Only consider i < j
            for i in range(start, end):
                for j in range(i + 1, n):
                    d = dist_sq[i - start, j]
                    if d < min_dist_sq:
                        min_dist_sq = d
        return min_dist_sq, max_norm
    
    # Greedy: try to add each candidate if it doesn't violate the constraint
    # A candidate v can be added if:
    #   min(||v - u||^2 for all u in current) >= max(max_norm(current), ||v||^2)
    
    # Sort candidates by norm (smaller norm first, as they're "cheaper")
    cand_norms = np.sum(np.array(candidates)**2, axis=1)
    order = np.argsort(cand_norms)
    
    current_list = list(current)
    current_max_norm_sq = np.max(np.sum(current**2, axis=1))
    
    # For efficiency, precompute norms of current vectors
    current_norms_sq = np.sum(np.array(current_list)**2, axis=1)
    
    added_any = False
    for idx in order:
        if time.time() > deadline:
            break
        v = candidates[idx]
        v_norm_sq = np.sum(v**2)
        new_max_norm_sq = max(current_max_norm_sq, v_norm_sq)
        # Check: for all u in current, ||v - u||^2 >= new_max_norm_sq
        current_arr = np.array(current_list)
        dots = current_arr @ v  # (n,)
        dist_sq = current_norms_sq + v_norm_sq - 2 * dots
        if np.min(dist_sq) >= new_max_norm_sq:
            current_list.append(v)
            current_norms_sq = np.append(current_norms_sq, v_norm_sq)
            current_max_norm_sq = new_max_norm_sq
            added_any = True
    
    # Try removal-then-addition swaps (Kempe-chain style)
    # For each existing vector, try removing it and see if we can add 2+ candidates
    if time.time() < deadline - 30:
        current_arr = np.array(current_list)
        current_norms_sq = np.sum(current_arr**2, axis=1)
        current_max_norm_sq = np.max(current_norms_sq)
        n = len(current_list)
        
        # Try removing each vector and adding the best replacement
        improved = True
        while improved and time.time() < deadline - 15:
            improved = False
            for rem_idx in range(len(current_list)):
                if time.time() > deadline - 15:
                    break
                # Remove vector rem_idx
                remaining = [current_list[i] for i in range(len(current_list)) if i != rem_idx]
                rem_norm_sq = current_norms_sq[rem_idx]
                
                # Compute new max norm
                if len(remaining) == 0:
                    new_max = 0
                else:
                    rem_arr = np.array(remaining)
                    rem_norms = np.sum(rem_arr**2, axis=1)
                    new_max = np.max(rem_norms)
                
                # Try to add candidates in order of norm
                added = 0
                temp_list = remaining[:]
                temp_norms_sq = np.sum(np.array(remaining)**2, axis=1) if remaining else np.array([])
                temp_max_norm_sq = new_max
                
                for c_idx in order:
                    if time.time() > deadline - 15:
                        break
                    v = candidates[c_idx]
                    v_norm_sq = np.sum(v**2)
                    candidate_max_norm_sq = max(temp_max_norm_sq, v_norm_sq)
                    temp_arr = np.array(temp_list)
                    dots = temp_arr @ v
                    dist_sq = temp_norms_sq + v_norm_sq - 2 * dots
                    if len(dist_sq) == 0 or np.min(dist_sq) >= candidate_max_norm_sq:
                        temp_list.append(v)
                        temp_norms_sq = np.append(temp_norms_sq, v_norm_sq)
                        temp_max_norm_sq = candidate_max_norm_sq
                        added += 1
                    if added > 1:  # Net gain
                        break
                
                if added > 1:  # We removed 1 and added >1, net gain
                    current_list = temp_list
                    current_norms_sq = np.sum(np.array(current_list)**2, axis=1)
                    current_max_norm_sq = np.max(current_norms_sq)
                    improved = True
                    # Re-sort remaining candidates consideration
                    break
    
    # Final verification
    final_arr = np.array(current_list, dtype=np.int64)
    if len(final_arr) > best_count:
        best = final_arr
        best_count = len(final_arr)
    
    # Verify the result is valid
    if len(best) > 1:
        norms_sq = np.sum(best**2, axis=1)
        max_norm_sq = np.max(norms_sq)
        # Check min pairwise distance
        min_dist_sq = np.inf
        for i in range(len(best)):
            for j in range(i + 1, len(best)):
                d = np.sum((best[i] - best[j])**2)
                if d < min_dist_sq:
                    min_dist_sq = d
        if min_dist_sq < max_norm_sq:
            # Invalid, fall back to the original 220 vectors
            best = np.array(vectors, dtype=np.int64)
    
    return best
# EVOLVE-BLOCK-END
