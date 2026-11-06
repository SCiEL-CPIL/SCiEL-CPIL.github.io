import numpy as np
import itertools
import random
import time

def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters (not expected in this task)
        return list(itertools.product(range(q), repeat=n))
    
    N = 6561
    all_words = list(itertools.product(range(3), repeat=8))
    
    # Precompute neighbors for all words
    # neighbors[i] is a list of indices of words within distance 1 of word i
    neighbors = []
    # Map word to index
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    for i, w in enumerate(all_words):
        neigh = [i]
        for pos in range(8):
            for val in range(3):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                idx = word_to_idx[tuple(new_w)]
                neigh.append(idx)
        neighbors.append(np.array(neigh, dtype=np.int32))
    
    # Convert to a single 2D array for vectorized operations if possible?
    # Since all have same size (17), we can use a 2D array.
    neighbors_2d = np.array(neighbors, dtype=np.int32)
    
    def get_coverage_counts(code_indices):
        # code_indices is a list of int indices
        if not code_indices:
            return np.zeros(N, dtype=np.int32)
        # Flatten the neighbors of the selected codewords
        # neighbors_2d[code_indices] gives a 2D array of shape (len(code_indices), 17)
        flat_neighbors = neighbors_2d[code_indices].flatten()
        # Count occurrences
        counts = np.bincount(flat_neighbors, minlength=N)
        return counts
    
    def is_valid(counts):
        return np.all(counts >= 1)
    
    # 1. Greedy Initialization
    uncovered = np.ones(N, dtype=bool)
    C = []
    start_time = time.time()
    time_limit = 140.0 # Safety margin for 150s kill
    
    while np.any(uncovered):
        if time.time() - start_time > time_limit:
            break
        # Find word covering most uncovered
        # counts of uncovered neighbors for each candidate
        # We can speed this up by only checking a subset if C is large, 
        # but N=6561 is small enough for a full scan with optimization.
        
        # Optimization: Use the fact that we only care about uncovered points.
        # For each candidate i, count how many of its neighbors are in `uncovered`.
        # This is still a loop over N candidates.
        
        best_idx = -1
        best_count = -1
        
        # To speed up, we can sample or use a heuristic. 
        # However, greedy is the baseline. Let's try to vectorize the count.
        # We can't easily vectorize "count of neighbors in set" without a large matrix.
        # But we can update counts incrementally? No, greedy picks max.
        
        # Let's just do the loop but efficiently.
        unc = uncovered
        nbrs = neighbors_2d
        for i in range(N):
            # Count how many of nbrs[i] are True in unc
            # np.count_nonzero(unc[nbrs[i]])
            cnt = np.count_nonzero(unc[nbrs[i]])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
                if best_count == 17: # Max possible
                    break
        
        if best_count == 0:
            break
        
        C.append(best_idx)
        uncovered[nbrs[best_idx]] = False
        
    # Initial Reduction
    counts = get_coverage_counts(C)
    changed = True
    while changed:
        changed = False
        new_C = []
        for c in C:
            # If min count in neighborhood is >= 2, we can remove c
            if np.min(counts[neighbors_2d[c]]) >= 2:
                counts[neighbors_2d[c]] -= 1
                changed = True
            else:
                new_C.append(c)
        C = new_C
        
    best_C = C.copy()
    best_size = len(C)
    best_counts = get_coverage_counts(best_C)
    
    # 2. Iterated Local Search
    # We will try to swap codewords to reduce the size.
    # A swap is (remove c, add c') such that coverage remains valid and size decreases (after reduction).
    
    # Precompute "unique points" for each codeword in current code?
    # This is dynamic.
    
    while time.time() - start_time < time_limit:
        # Try to improve current C
        improved = False
        indices = list(range(len(C)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit:
                break
                
            c = C[idx]
            # Find points covered ONLY by c
            # These are points p where counts[p] == 1 and p in neighbors[c]
            # If no such points, we can remove c.
            unique_mask = (counts[neighbors_2d[c]] == 1)
            unique_points = neighbors_2d[c][unique_mask]
            
            if len(unique_points) == 0:
                # Remove c
                C.remove(c)
                counts[neighbors_2d[c]] -= 1
                improved = True
                # Reduce again
                changed = True
                while changed:
                    changed = False
                    new_C = []
                    for cc in C:
                        if np.min(counts[neighbors_2d[cc]]) >= 2:
                            counts[neighbors_2d[cc]] -= 1
                            changed = True
                        else:
                            new_C.append(cc)
                    C = new_C
                break
            
            # If unique points exist, we must find a c' that covers them.
            # If len(unique_points) > 3, it's unlikely to find a good swap quickly, skip.
            if len(unique_points) > 3:
                continue
                
            # Find candidates c' that cover all unique_points
            # c' must be in the intersection of neighbors of all unique_points
            # And c' should not be in C (or if it is, we are just moving, which doesn't reduce size unless reduction kicks in)
            
            cand_set = set(neighbors_2d[unique_points[0]].tolist())
            for p in unique_points[1:]:
                cand_set &= set(neighbors_2d[p].tolist())
                if not cand_set:
                    break
            
            if not cand_set:
                continue
            
            # Remove current C from candidates to ensure we are adding a new one
            # Actually, if c' is in C, swapping c with c' is just a move. 
            # It might allow further reduction.
            # But to reduce size, we usually want to replace c with a c' that allows removing another codeword.
            # For now, let's just try to find ANY c' that covers the unique points.
            
            # Filter out c itself
            cand_set.discard(c)
            
            if not cand_set:
                continue
            
            # Pick a random candidate
            c_prime = random.choice(list(cand_set))
            
            # Try the swap
            old_C = C.copy()
            old_counts = counts.copy()
            
            # Remove c
            C.remove(c)
            counts[neighbors_2d[c]] -= 1
            
            # Add c_prime
            # If c_prime is already in C, we are duplicating? 
            # The problem requires distinct codewords.
            # If c_prime is in C, we can't add it again.
            if c_prime in C:
                # This is a "move" not a "swap" in the sense of set membership.
                # If c_prime is in C, then c_prime is already covering some points.
                # We just removed c. If c_prime covers the unique points of c, 
                # then the coverage is still valid.
                # But we didn't add a new word, we just removed c.
                # So size decreased by 1!
                # Let's check if coverage is still valid.
                # We subtracted counts for c.
                # We did NOT add counts for c_prime (it's already there).
                # So we just removed c.
                if np.all(counts >= 1):
                    # Valid! Size reduced by 1.
                    # Reduce further
                    changed = True
                    while changed:
                        changed = False
                        new_C = []
                        for cc in C:
                            if np.min(counts[neighbors_2d[cc]]) >= 2:
                                counts[neighbors_2d[cc]] -= 1
                                changed = True
                            else:
                                new_C.append(cc)
                        C = new_C
                    improved = True
                    break
                else:
                    # Coverage broken, revert
                    C = old_C
                    counts = old_counts
                    continue
            
            # c_prime is NOT in C.
            C.append(c_prime)
            counts[neighbors_2d[c_prime]] += 1
            
            # Check validity
            if np.any(counts < 1):
                C = old_C
                counts = old_counts
                continue
            
            # Try to reduce
            new_C = []
            changed = True
            while changed:
                changed = False
                for cc in C:
                    if np.min(counts[neighbors_2d[cc]]) >= 2:
                        counts[neighbors_2d[cc]] -= 1
                        changed = True
                    else:
                        new_C.append(cc)
            C = new_C
            
            if len(C) < len(old_C):
                improved = True
                break
            else:
                # Keep the change if it's neutral? 
                # It might help future moves.
                # Let's keep it to explore.
                improved = True
                break
        
        if len(C) < best_size:
            best_size = len(C)
            best_C = C.copy()
            best_counts = get_coverage_counts(best_C)
            
        if not improved:
            # Perturbation: Randomly swap a few codewords
            if len(C) < 5:
                break
                
            for _ in range(3):
                if time.time() - start_time > time_limit:
                    break
                idx = random.randint(0, len(C)-1)
                c = C[idx]
                # Pick a random neighbor of c
                c_prime = neighbors_2d[c][random.randint(0, 16)]
                
                if c_prime in C:
                    continue
                    
                old_C = C.copy()
                old_counts = counts.copy()
                
                C.remove(c)
                C.append(c_prime)
                counts[neighbors_2d[c]] -= 1
                counts[neighbors_2d[c_prime]] += 1
                
                if np.any(counts < 1):
                    C = old_C
                    counts = old_counts
                    continue
                    
                # Reduce
                new_C = []
                changed = True
                while changed:
                    changed = False
                    for cc in C:
                        if np.min(counts[neighbors_2d[cc]]) >= 2:
                            counts[neighbors_2d[cc]] -= 1
                            changed = True
                        else:
                            new_C.append(cc)
                C = new_C
                
                # Accept if size is smaller or equal (to escape local min)
                if len(C) <= len(old_C):
                    break

    # Final Reduction
    changed = True
    while changed:
        changed = False
        new_C = []
        for c in best_C:
            if np.min(best_counts[neighbors_2d[c]]) >= 2:
                best_counts[neighbors_2d[c]] -= 1
                changed = True
            else:
                new_C.append(c)
        best_C = new_C
        
    return [list(all_words[i]) for i in best_C]
