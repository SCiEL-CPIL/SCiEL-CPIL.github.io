# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates 
    such that their maximum norm is smaller than their minimum pairwise distance.
    
    Strategy:
    1. Build set using vectors with norm squared 4 (four +/-1s).
       Condition for R=4: min_dist_sq >= 4.
       For two 4-norm vectors u, v: ||u-v||^2 = 8-2u.v >= 4 => u.v <= 2.
       
    2. Greedily add vectors from the pool of all 4-norm integer vectors.
    3. Try to add 2-norm vectors (two +/-1s) to the set.
    4. Local search: try swapping out points to fit more in.
    """
    d = 11
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    def elapsed():
        return time.time() - start_time
    
    # Generate all vectors with exactly four non-zero entries, each +1 or -1
    candidates_4 = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            for idx, val in zip(combo, signs):
                v[idx] = val
            candidates_4.append(v)
    
    # Sort candidates: prefer those with more spread (lower max overlap potential)
    # Actually, let's just use a random shuffle for diversity
    np.random.seed(42)
    idx_4 = np.random.permutation(len(candidates_4))
    candidates_4 = [candidates_4[i] for i in idx_4]
    
    # Greedy selection for 4-norm vectors
    selected = []
    selected_norms = []  # parallel list of norms
    
    for v in candidates_4:
        if elapsed() > deadline - 90:  # Reserve 90s for 2-norm + local search
            break
            
        valid = True
        if selected:
            # Check dot products against all selected
            # For 4-norm vs 4-norm: dot <= 2
            # For 4-norm vs 2-norm: dot <= 1
            sel_arr = np.array(selected, dtype=np.int64)
            dots = sel_arr @ v
            # Need to check per-norm constraints
            # Create a mask-based check
            for i, (dot, norm) in enumerate(zip(dots, selected_norms)):
                if norm == 4:
                    if dot > 2:
                        valid = False
                        break
                elif norm == 2:
                    if dot > 1:
                        valid = False
                        break
                else:
                    # Other norms - use general check
                    # ||v||^2 = 4, ||u||^2 = norm
                    # ||v-u||^2 = 4 + norm - 2*dot >= 4 => norm - 2*dot >= 0 => dot <= norm/2
                    if dot > norm // 2:
                        valid = False
                        break
        
        if valid:
            selected.append(v)
            selected_norms.append(4)
            
    # Try to add 2-norm vectors
    candidates_2 = []
    for combo in itertools.combinations(range(d), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(d, dtype=np.int64)
            for idx, val in zip(combo, signs):
                v[idx] = val
            candidates_2.append(v)
    
    np.random.seed(123)
    idx_2 = np.random.permutation(len(candidates_2))
    candidates_2 = [candidates_2[i] for i in idx_2]
    
    for v in candidates_2:
        if elapsed() > deadline - 30:
            break
            
        valid = True
        if selected:
            for u, norm in zip(selected, selected_norms):
                dot = np.dot(u, v)
                if norm == 4:
                    if dot > 1:
                        valid = False
                        break
                elif norm == 2:
                    if dot > 0:
                        valid = False
                        break
                else:
                    if dot > norm // 2:
                        valid = False
                        break
        
        if valid:
            selected.append(v)
            selected_norms.append(2)
    
    # Local search: try to improve by swapping
    # For each pair (i, j) where i is in selected and j is not, try swapping
    # This is expensive, so do a limited version
    
    if elapsed() < deadline - 20:
        # Build a set of selected vectors for quick lookup
        selected_set = set(tuple(p) for p in selected)
        all_candidates = candidates_4 + candidates_2
        
        # Try a few random swaps
        n_swaps = 0
        max_swaps = 500
        np.random.seed(777)
        
        for _ in range(max_swaps):
            if elapsed() > deadline - 15:
                break
            if n_swaps >= max_swaps:
                break
            
            # Pick a random index from selected to potentially remove
            i = np.random.randint(len(selected))
            u = selected[i]
            u_norm = selected_norms[i]
            
            # Pick a random candidate not in selected
            j = np.random.randint(len(all_candidates))
            v = all_candidates[j]
            if tuple(v) in selected_set:
                continue
            
            # Check if we can remove u and add v
            # First check v against all selected except u
            valid = True
            for idx2, (w, w_norm) in enumerate(zip(selected, selected_norms)):
                if idx2 == i:
                    continue
                dot = np.dot(w, v)
                if w_norm == 4:
                    if dot > 2:
                        valid = False
                        break
                elif w_norm == 2:
                    if dot > 1:
                        valid = False
                        break
                else:
                    if dot > w_norm // 2:
                        valid = False
                        break
            
            if not valid:
                continue
            
            # Check all pairs in selected except u still valid (they were before, so yes)
            # Just do the swap
            selected[i] = v
            selected_norms[i] = len(v[v != 0])  # compute norm
            selected_set.discard(tuple(u))
            selected_set.add(tuple(v))
            n_swaps += 1
    
    # Final validation
    if not selected:
        # Fallback
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = s1
                        p[j] = s2
                        points.append(p)
        return np.array(points, dtype=np.int64)
    
    # Validate the result
    k = len(selected)
    if k < 2:
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                p1 = np.zeros(d, dtype=np.int64)
                p1[i] = 1
                p1[j] = 1
                points.append(p1)
        return np.array(points, dtype=np.int64)
    
    arr = np.array(selected, dtype=np.int64)
    
    # Check no zero vectors
    if np.any(arr == 0).all():
        pass
    
    # Verify min dist >= max norm
    min_dist_sq = float('inf')
    max_norm_sq = 0
    for i in range(k):
        norm_sq = np.dot(arr[i], arr[i])
        max_norm_sq = max(max_norm_sq, norm_sq)
        if norm_sq == 0:
            # Invalid, remove
            pass
        for j in range(i + 1, k):
            diff = arr[i] - arr[j]
            dist_sq = np.dot(diff, diff)
            min_dist_sq = min(min_dist_sq, dist_sq)
    
    if min_dist_sq < max_norm_sq:
        # Invalid! Fall back to a known valid set
        points = []
        for i in range(d):
            for j in range(i + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        p = np.zeros(d, dtype=np.int64)
                        p[i] = s1
                        p[j] = s2
                        points.append(p)
        return np.array(points, dtype=np.int64)
    
    return arr

# EVOLVE-BLOCK-END
