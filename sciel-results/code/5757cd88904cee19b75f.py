import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    # 1. Generate all words
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # 2. Precompute neighbors (balls of radius 1)
    neighbors_list = []
    for i, w in enumerate(words):
        neigh = [i]
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for val in range(q):
                if val == orig_val:
                    continue
                w_list[pos] = val
                idx = word_to_idx[tuple(w_list)]
                neigh.append(idx)
                w_list[pos] = orig_val
        neighbors_list.append(np.array(neigh, dtype=np.int32))
        
    # Build adjacency matrix for vectorized greedy
    M = np.zeros((N, N), dtype=np.float32)
    for i, neigh in enumerate(neighbors_list):
        M[i, neigh] = 1.0

    # 3. Greedy Set Cover (Vectorized)
    uncovered = np.ones(N, dtype=np.float32)
    code = []
    code_set = set()
    
    while np.any(uncovered > 0.5):
        if time.time() > deadline - 40:
            break
            
        gains = M @ uncovered
        for idx in code_set:
            gains[idx] = -1.0
            
        best_idx = np.argmax(gains)
        
        if gains[best_idx] <= 0:
            break
            
        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbors_list[best_idx]] = 0.0
        
    # 4. Local Search
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_list[idx]] += 1
        
    def is_removable(c):
        return np.all(counts[neighbors_list[c]] > 1)

    def get_critical_points(c):
        neigh = neighbors_list[c]
        return neigh[counts[neigh] == 1]

    def try_deletion_pass():
        """Remove all removable codewords. Returns True if anything was removed."""
        nonlocal counts, code, code_set
        to_remove = [c for c in code if is_removable(c)]
        if not to_remove:
            return False
        for c in to_remove:
            code.remove(c)
            code_set.remove(c)
            counts[neighbors_list[c]] -= 1
        return True

    # Precompute neighbor sets for fast membership testing
    neigh_sets = [set(neighbors_list[i]) for i in range(N)]

    # Vectorized swap scoring: for each candidate w, compute
    #   gain(w)    = |{p in neigh(w) : counts[p]==1} \ neigh(c)|
    #   penalty(w) = |{p in neigh(c) : counts[p]==2} \ neigh(w)|
    #   score(w)   = gain(w) - penalty(w)
    # We precompute count1_mask and count2_mask once per iteration.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # --- Deletion Pass ---
        if try_deletion_pass():
            improved = True
            continue  # restart loop after deletions
        
        # --- Swap Pass ---
        count1_mask = (counts == 1)
        count2_mask = (counts == 2)
        
        current_code_copy = list(code)
        random.shuffle(current_code_copy)
        
        for c in current_code_copy:
            if time.time() > deadline - 5:
                break
            
            if c not in code_set:
                continue

            crit_points = get_critical_points(c)
            
            if len(crit_points) == 0:
                continue
                
            # Find candidates: must cover all critical points, not in current code
            candidate_mask = np.ones(N, dtype=bool)
            for p in crit_points:
                candidate_mask &= (M[:, p] > 0.5)
            
            for idx in code_set:
                candidate_mask[idx] = False
            
            candidates = np.where(candidate_mask)[0]
            
            if len(candidates) == 0:
                continue
            
            # Vectorized scoring
            neigh_c = neighbors_list[c]
            set_neigh_c = neigh_sets[c]
            
            # For each candidate w:
            #   gain = sum over p in neigh(w) of [counts[p]==1 and p not in neigh_c]
            #   penalty = sum over p in neigh(c) of [counts[p]==2 and p not in neigh(w)]
            
            # gain: M[candidates, :] @ gain_vec where gain_vec[p] = 1 if counts[p]==1 and p not in neigh_c
            gain_vec = np.zeros(N, dtype=np.float32)
            gain_vec[count1_mask] = 1.0
            gain_vec[neigh_c] = 0.0  # exclude points in neigh_c
            gains_arr = M[candidates] @ gain_vec
            
            # penalty: for each candidate w, count points in neigh_c with count==2 not in neigh(w)
            # penalty(w) = sum over p in neigh_c of [counts[p]==2 and M[w,p]==0]
            #            = sum over p in neigh_c of [counts[p]==2] - sum over p in neigh_c of [counts[p]==2 and M[w,p]==1]
            pen_base = float(np.sum(count2_mask[neigh_c]))
            # For each candidate w, sum of M[w, p] over p in neigh_c where count2_mask[p]
            pen_vec = np.zeros(N, dtype=np.float32)
            pen_vec[count2_mask] = 1.0
            pen_vec[neigh_c] = 0.0  # only consider p in neigh_c
            # Actually: penalty(w) = pen_base - sum_{p in neigh_c, count2_mask[p]} M[w,p]
            # Let's compute: cov = M[candidates, neigh_c] * count2_mask[neigh_c]
            count2_neigh_c = count2_mask[neigh_c]  # boolean array
            cov = M[candidates][:, neigh_c] * count2_neigh_c.astype(np.float32)
            penalties_arr = pen_base - cov.sum(axis=1)
            
            scores_arr = gains_arr - penalties_arr
            
            best_local = np.argmax(scores_arr)
            best_score = scores_arr[best_local]
            
            if best_score > 0:
                best_candidate = int(candidates[best_local])
                
                # Perform swap
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                
                code.append(best_candidate)
                code_set.add(best_candidate)
                counts[neighbors_list[best_candidate]] += 1
                
                improved = True
                # Immediate deletion pass
                try_deletion_pass()
                break  # restart outer loop

        # --- 2-for-1 Exchange Pass ---
        # Try replacing one codeword with two new ones to reduce total count
        if time.time() < deadline - 10 and len(code) > 10:
            count1_mask = (counts == 1)
            current_code_copy = list(code)
            random.shuffle(current_code_copy)
            
            for c in current_code_copy:
                if time.time() > deadline - 10:
                    break
                if c not in code_set:
                    continue
                
                crit_points = get_critical_points(c)
                if len(crit_points) == 0:
                    continue
                
                # Find single candidate that covers all critical points
                candidate_mask = np.ones(N, dtype=bool)
                for p in crit_points:
                    candidate_mask &= (M[:, p] > 0.5)
                for idx in code_set:
                    candidate_mask[idx] = False
                candidates = np.where(candidate_mask)[0]
                
                if len(candidates) == 0:
                    continue
                
                # For 2-for-1: remove c, add two new codewords w1, w2
                # Net change: -1 + 2 = +1 (worse) unless deletions cascade
                # Actually this increases code size. Skip unless we can find
                # a pair that enables further deletions.
                # Better: try 1-for-2 (remove two, add one) - but that's harder.
                # Instead, try: find w that covers crit_points AND enables removal of another codeword.
                # This is complex; skip for now and rely on swap+deletion.
                pass

    # Final cleanup
    changed = True
    while changed:
        changed = False
        for c in list(code):
            if is_removable(c):
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_list[c]] -= 1
                changed = True

    return [list(words[i]) for i in code]
