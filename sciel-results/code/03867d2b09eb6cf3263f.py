import numpy as np
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s wall-clock limit
    
    best_vectors = None
    best_size = 0
    
    # Strategy: Use vectors with exactly k non-zero entries (all ±1).
    # Condition: min pairwise distance >= max norm.
    # For two vectors v, w with k non-zeros:
    # dist^2 = |v-w|^2 = sum (v_i - w_i)^2.
    # If supports are disjoint, dist^2 = 2k. Norm^2 = k. dist^2 >= norm^2 always holds.
    # If supports overlap, let m be number of common indices.
    # dist^2 = (k-m)*1 + (k-m)*1 + m*(v_i-w_i)^2.
    # If v_i != w_i on common index, (v_i-w_i)^2 = 4. If v_i = w_i, 0.
    # To ensure dist^2 >= k, we need careful selection.
    #
    # Simpler approach: Select vectors such that no two vectors have overlapping supports
    # in a way that reduces distance.
    # Actually, if we just take all ±1 vectors on disjoint supports, it's easy.
    # But we want max k.
    #
    # Let's use a greedy algorithm with a specific candidate set.
    # Candidate set: All vectors with exactly k non-zeros (±1).
    # Sort candidates by some heuristic or just iterate.
    # Check constraint: min dist to selected >= sqrt(k_max_norm).
    # Wait, the constraint is: min pairwise dist >= max norm.
    # If all vectors have norm sqrt(k), then max norm is sqrt(k).
    # So we need dist >= sqrt(k).
    # dist^2 >= k.
    #
    # Let's try k=5.
    # If we pick vectors with disjoint supports, dist^2 = 2k = 10 >= 5. OK.
    # But we can do better by allowing some overlap if signs differ.
    #
    # Let's use a greedy approach on a large set of candidates.
    # To be fast, we limit the candidate set or use a smart ordering.
    
    # We will try to build a set of vectors.
    # We'll use a pool of candidates.
    # To maximize k, we can use vectors with different norms?
    # No, if we mix norms, max norm is determined by the largest one.
    # So it's best to have all vectors with the same norm, or close.
    # If we use all vectors with k non-zeros, norm is sqrt(k).
    
    # Let's generate all ±1 vectors with exactly k non-zeros for k=5,6,7.
    # For k=5: C(11,5)*2^5 = 462 * 32 = 14784 candidates.
    # For k=6: C(11,6)*2^6 = 462 * 64 = 29568 candidates.
    # For k=7: C(11,7)*2^7 = 330 * 128 = 42240 candidates.
    
    # Greedy selection:
    # 1. Generate candidates.
    # 2. Shuffle.
    # 3. Pick first vector.
    # 4. For each next candidate, check if dist^2 to ALL selected vectors >= k.
    #    If yes, add it.
    #
    # Complexity: O(N * K * d). N~30k, K~1000, d=11. 30k * 1000 * 11 ~ 330M ops.
    # Might be slow in Python if not vectorized.
    # We can vectorize the distance check.
    
    for k in [5, 6, 7]:
        if time.time() > deadline:
            break
            
        # Generate candidates
        candidates = []
        # Use bitmasks for efficiency in generation?
        # itertools is fine.
        for indices in itertools.combinations(range(11), k):
            # Generate all 2^k sign combos
            # We can use product
            for signs in itertools.product([1, -1], repeat=k):
                v = np.zeros(11, dtype=np.int64)
                for idx, s in zip(indices, signs):
                    v[idx] = s
                candidates.append(v)
                
        candidates = np.array(candidates)
        N = len(candidates)
        
        # Shuffle
        np.random.seed(42)
        perm = np.random.permutation(N)
        candidates = candidates[perm]
        
        selected = []
        # Precompute norm squared = k
        threshold = k
        
        # Vectorized check:
        # Maintain selected as a list, but for checking a new vector v,
        # compute dist^2 to all selected.
        # If selected is empty, add.
        # Else, sel = np.array(selected), dists = np.sum((sel - v)**2, axis=1).
        # If np.min(dists) >= threshold, add.
        
        for i in range(N):
            if time.time() > deadline:
                break
                
            v = candidates[i]
            
            if not selected:
                selected.append(v)
            else:
                # Optimization: if selected is large, this is expensive.
                # But K is likely < 1000.
                sel_arr = np.array(selected)
                # (sel_arr - v) is (K, 11)
                # Squared dist:
                dists_sq = np.sum((sel_arr - v)**2, axis=1)
                if np.min(dists_sq) >= threshold:
                    selected.append(v)
                    
        current_size = len(selected)
        if current_size > best_size:
            best_size = current_size
            best_vectors = np.array(selected)
            
    if best_vectors is None:
        # Fallback: just return a few vectors
        v1 = np.zeros(11, dtype=np.int64)
        v1[0] = 1
        v2 = np.zeros(11, dtype=np.int64)
        v2[1] = 1
        best_vectors = np.array([v1, v2])
        
    return best_vectors
