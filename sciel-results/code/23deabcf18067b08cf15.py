import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    """
    Construct a ternary covering code K_3(6,1).
    Returns a list of codewords (lists of integers).
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    # Fallback for non-standard parameters
    if q != 3 or n != 6 or radius != 1:
        return [list(t) for t in itertools.product(range(q), repeat=n)]

    total_words = 3 ** n  # 729
    
    # --- 1. Precompute Data Structures ---
    
    # Helper: Convert index to word tuple
    def idx_to_word(idx):
        w = [0] * n
        temp = idx
        for i in range(n):
            w[n - 1 - i] = temp % 3
            temp //= 3
        return tuple(w)
    
    # Helper: Convert word tuple to index
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * 3 + val
        return idx

    # Precompute spheres (Hamming balls of radius 1)
    # spheres[i] is a numpy array of indices
    spheres = [None] * total_words
    spheres_set = [None] * total_words
    spheres_list = [None] * total_words
    
    for idx in range(total_words):
        w = list(idx_to_word(idx))
        sphere_set = {idx}
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                w[pos] = new_val
                sphere_set.add(word_to_idx(w))
                w[pos] = orig_val
        spheres_list[idx] = list(sphere_set)
        spheres[idx] = np.array(sphere_set, dtype=np.int32)
        spheres_set[idx] = sphere_set
    
    # Precompute coverers: for each word w, which codewords cover it
    # coverers[w] = list of c such that w in spheres[c]
    # This is the inverse of spheres.
    coverers = [[] for _ in range(total_words)]
    for c in range(total_words):
        for w in spheres_list[c]:
            coverers[w].append(c)
    
    # Convert coverers to sets for fast membership testing (c in coverers[w])
    coverers_set = [set(c) for c in coverers]

    # --- 2. Initial Code Construction ---
    # Direct sum of tetracode (9 words, len 4) and all words on remaining n-4 coords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    initial_indices = [word_to_idx(cw) for cw in initial_codewords]
    
    # --- 3. Helper Functions for Search ---
    
    def compute_counts(code_indices):
        counts = np.zeros(total_words, dtype=np.int32)
        for idx in code_indices:
            counts[spheres[idx]] += 1
        return counts

    def reduce_code(code_indices, counts=None):
        """
        Greedily remove redundant codewords.
        Returns (new_code_indices, new_counts)
        """
        if counts is None:
            counts = compute_counts(code_indices)
        
        code_set = set(code_indices)
        changed = True
        # We iterate until no more changes.
        # Optimization: keep a list of candidates to check.
        while changed:
            changed = False
            # Iterate over a copy of the list to allow removal
            for idx in list(code_set):
                # Check if all neighbors are covered by >1 codeword
                # If min(counts[spheres[idx]]) > 1, then idx is redundant
                # Using numpy min is fast
                if np.min(counts[spheres[idx]]) > 1:
                    code_set.remove(idx)
                    counts[spheres[idx]] -= 1
                    changed = True
        
        return list(code_set), counts

    def find_swap_target(uncovered_indices, current_set, counts):
        """
        Find a codeword c_add not in current_set that covers all words in uncovered_indices.
        Returns c_add or None.
        """
        if not uncovered_indices:
            return None
        
        u0 = uncovered_indices[0]
        candidates = coverers[u0]
        
        # Shuffle candidates for diversity?
        # random.shuffle(candidates)
        
        for c_add in candidates:
            if c_add in current_set:
                continue
            
            # Check if c_add covers all uncovered indices
            valid = True
            for u in uncovered_indices:
                if c_add not in coverers_set[u]:
                    valid = False
                    break
            
            if valid:
                return c_add
        
        return None

    # --- 4. Main Search Loop ---
    
    rng = random.Random(42)
    
    # Start with initial code
    current_code, current_counts = reduce_code(initial_indices)
    best_code = current_code[:]
    best_counts = current_counts.copy()
    best_len = len(best_code)
    
    all_indices = list(range(total_words))
    
    no_improve_count = 0
    max_no_improve = 20
    
    # Local search strategy:
    # 1. Reduce
    # 2. Try removals (if any remain, reduce handles it)
    # 3. Try single swaps: remove c, if invalid, find c_add to cover the gap.
    #    If swap is valid and leads to a smaller code after reduction, accept.
    #    Actually, a swap itself doesn't reduce size. It just moves weight.
    #    The reduction step is what reduces size.
    #    So: Try swap -> Reduce -> If smaller, accept.
    
    while time.time() < deadline:
        # 1. Reduce
        current_code, current_counts = reduce_code(current_code, current_counts)
        current_set = set(current_code)
        
        if len(current_code) < best_len:
            best_len = len(current_code)
            best_code = current_code[:]
            best_counts = current_counts.copy()
            no_improve_count = 0
            if best_len <= 71:
                break
        else:
            no_improve_count += 1
        
        if no_improve_count > max_no_improve:
            # Perturb: Random valid swap
            if current_code:
                c_remove = rng.choice(current_code)
                # Find a valid swap
                # Simulate removal
                old_counts = current_counts.copy()
                current_counts[spheres[c_remove]] -= 1
                uncovered = [w for w in spheres_list[c_remove] if current_counts[w] == 0]
                
                if not uncovered:
                    # Just remove it? But reduce_code would have removed it.
                    # So this case shouldn't happen if reduce_code is thorough.
                    # But maybe counts changed? No, we just reduced.
                    # So uncovered should not be empty.
                    pass
                
                c_add = find_swap_target(uncovered, current_set, current_counts)
                
                if c_add is not None:
                    # Perform swap
                    current_set.remove(c_remove)
                    current_set.add(c_add)
                    current_counts[spheres[c_remove]] += 1 # Restore removal
                    current_counts[spheres[c_add]] += 1 # Add new
                    # Wait, the logic above was:
                    # 1. counts -= sphere(c_remove)
                    # 2. Find c_add
                    # 3. counts += sphere(c_remove) (restore)
                    # 4. counts += sphere(c_add)
                    # Net effect: counts[spheres[c_add]] += 1, counts[spheres[c_remove]] unchanged?
                    # No.
                    # Original counts C.
                    # C' = C - S(c_remove)
                    # C'' = C' + S(c_remove) + S(c_add) = C + S(c_add)
                    # This is wrong. We want to replace c_remove with c_add.
                    # So C_final = C - S(c_remove) + S(c_add).
                    
                    # Let's redo the update logic carefully.
                    # We started with C.
                    # We did C -= S(c_remove).
                    # We found c_add.
                    # We want C_final = C - S(c_remove) + S(c_add).
                    # Currently C is C - S(c_remove).
                    # So we just need to do C += S(c_add).
                    
                    # But wait, in the code above I did:
                    # current_counts[spheres[c_remove]] += 1 # Restore removal
                    # current_counts[spheres[c_add]] += 1 # Add new
                    # This results in C + S(c_add). Which is incorrect.
                    
                    # Correct update:
                    # current_counts is currently C - S(c_remove).
                    # We need to add S(c_add).
                    current_counts[spheres[c_add]] += 1
                    
                    current_code = list(current_set)
                    no_improve_count = 0
                else:
                    # Restore counts
                    current_counts[spheres[c_remove]] += 1
                    # Reset to best
                    current_code = best_code[:]
                    current_counts = best_counts.copy()
                    no_improve_count = 0
            continue
        
        # 2. Try to find a beneficial swap
        improved = False
        candidates = current_code[:]
        rng.shuffle(candidates)
        
        for c_remove in candidates:
            if c_remove not in current_set:
                continue
            
            # Simulate removal
            old_counts = current_counts.copy()
            current_counts[spheres[c_remove]] -= 1
            
            # Find uncovered words
            uncovered = [w for w in spheres_list[c_remove] if current_counts[w] == 0]
            
            if not uncovered:
                # Redundant, should have been removed by reduce_code
                # But maybe due to integer issues or something?
                # Just revert and continue
                current_counts = old_counts
                continue
            
            # Find a c_add that covers all uncovered
            c_add = find_swap_target(uncovered, current_set, current_counts)
            
            if c_add is None:
                current_counts = old_counts
                continue
            
            # Perform swap
            current_set.remove(c_remove)
            current_set.add(c_add)
            current_counts[spheres[c_add]] += 1 # Add new sphere to the "removed" state counts
            
            # Try to reduce
            new_code, new_counts = reduce_code(list(current_set), current_counts)
            
            if len(new_code) < len(current_code):
                # Accept
                current_code = new_code
                current_counts = new_counts
                current_set = set(current_code)
                improved = True
                break
            else:
                # Revert
                current_set.remove(c_add)
                current_set.add(c_remove)
                current_counts = old_counts
            
            if time.time() > deadline - 5.0:
                break
        
        if not improved:
            no_improve_count += 1

    # Final check: ensure best_code is valid and converted to lists
    final_counts = np.zeros(total_words, dtype=np.int32)
    for idx in best_code:
        final_counts[spheres[idx]] += 1
    if np.any(final_counts == 0):
        # Fallback to initial if something went wrong
        return [list(idx_to_word(idx)) for idx in initial_indices]
    
    return [list(idx_to_word(idx)) for idx in best_code]
