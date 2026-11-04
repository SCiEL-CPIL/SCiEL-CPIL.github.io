import numpy as np
import itertools
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin (evaluator kills at 150s)
    
    # For the specific problem: q=3, n=7, radius=1
    # Total words: 3^7 = 2187
    # Each codeword covers 1 + n*(q-1) = 1 + 7*2 = 15 words
    
    words = list(itertools.product(range(q), repeat=n))
    N = len(words)
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute neighbors (balls of radius 1)
    # neighbors[i] is a list of indices j such that dist(words[i], words[j]) <= radius
    neighbors = []
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
        neighbors.append(neigh)
    
    # Convert to numpy arrays for fast boolean indexing
    neighbors_arr = [np.array(neigh, dtype=np.int32) for neigh in neighbors]
    
    # Precompute for each point, which codewords cover it (reverse mapping)
    # This helps in local search
    # point_to_codewords[p] = list of codeword indices that cover point p
    point_to_codewords = [[] for _ in range(N)]
    for i, neigh in enumerate(neighbors_arr):
        for p in neigh:
            point_to_codewords[p].append(i)
    
    # Phase 1: Greedy Set Cover with numpy vectorization
    uncovered = np.ones(N, dtype=bool)
    code = []
    code_set = set()
    
    # Precompute coverage vectors for all words
    # coverage[i] is a boolean array of length N where coverage[i][p] = 1 if word i covers point p
    # This is memory intensive: 2187 * 2187 booleans ~ 4.8 MB, acceptable
    coverage = np.zeros((N, N), dtype=bool)
    for i, neigh in enumerate(neighbors_arr):
        coverage[i, neigh] = True
    
    while np.any(uncovered):
        if time.time() > deadline - 40:
            break
        
        # Vectorized gain computation
        gains = coverage[:, uncovered].sum(axis=1)
        
        # Only consider words not already in code
        for c in code_set:
            gains[c] = 0
        
        best_idx = int(np.argmax(gains))
        max_gain = gains[best_idx]
        
        if max_gain == 0:
            break
        
        code.append(best_idx)
        code_set.add(best_idx)
        uncovered[neighbors_arr[best_idx]] = False
    
    # Phase 2: Local Search - Deletion and Swap
    # counts[i] is the number of codewords in code that cover point i
    counts = np.zeros(N, dtype=np.int32)
    for idx in code:
        counts[neighbors_arr[idx]] += 1
    
    def is_removable(c):
        """Check if codeword c can be removed without uncovering any point"""
        return not np.any(counts[neighbors_arr[c]] == 1)
    
    def do_deletion_pass():
        """Remove all removable codewords. Returns True if any were removed."""
        removed = False
        for c in list(code):
            if c in code_set and is_removable(c):
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_arr[c]] -= 1
                removed = True
        return removed
    
    def do_swap_pass():
        """Try to swap codewords to enable further deletions. Returns True if any swap was made."""
        improved = False
        current_code_copy = list(code)
        random.shuffle(current_code_copy)
        
        for c in current_code_copy:
            if time.time() > deadline - 10:
                break
            if c not in code_set:
                continue
            
            # Find critical points of c (points covered only by c)
            crit_mask = counts[neighbors_arr[c]] == 1
            crit_points = neighbors_arr[c][crit_mask]
            
            if len(crit_points) == 0:
                continue  # Should have been deleted
            
            # Find candidates that cover all critical points
            # Candidate must be in intersection of neighbors of all crit_points
            if len(crit_points) == 1:
                candidates = neighbors_arr[crit_points[0]]
            else:
                # Intersect neighbor sets
                candidate_mask = np.ones(N, dtype=bool)
                for p in crit_points:
                    candidate_mask &= coverage[:, p]
                candidates = np.where(candidate_mask)[0]
            
            # Filter out existing codewords and c itself
            candidates = [x for x in candidates if x not in code_set]
            
            if not candidates:
                continue
            
            # Evaluate candidates by net reduction in critical points
            best_candidate = -1
            best_score = -1
            neigh_c = neighbors_arr[c]
            
            for w in candidates:
                neigh_w = neighbors_arr[w]
                
                # Points that become non-critical (were count 1, not covered by c, now covered by w)
                # These are in neigh_w, not in neigh_c, and have count 1
                gain_mask = (counts[neigh_w] == 1) & (~np.isin(neigh_w, neigh_c))
                gain = np.sum(gain_mask)
                
                # Points that become critical (were count 2, covered by c, not covered by w)
                # These are in neigh_c, not in neigh_w, and have count 2
                penalty_mask = (counts[neigh_c] == 2) & (~np.isin(neigh_c, neigh_w))
                penalty = np.sum(penalty_mask)
                
                score = gain - penalty
                if score > best_score:
                    best_score = score
                    best_candidate = w
            
            if best_candidate != -1 and best_score > 0:
                # Perform swap
                code.remove(c)
                code_set.remove(c)
                counts[neighbors_arr[c]] -= 1
                
                code.append(best_candidate)
                code_set.add(best_candidate)
                counts[neighbors_arr[best_candidate]] += 1
                
                improved = True
                # Immediate deletion pass after swap
                do_deletion_pass()
        
        return improved
    
    # Run local search until no improvement or time runs out
    improved = True
    while improved and time.time() < deadline - 15:
        improved = do_deletion_pass()
        if time.time() < deadline - 15:
            improved = do_swap_pass() or improved
    
    # Final cleanup
    while do_deletion_pass():
        pass
    
    return [list(words[i]) for i in code]
