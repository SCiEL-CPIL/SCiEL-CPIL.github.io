"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

def construct(q, n, radius):
    # Fallback for non-specific cases or if optimization fails
    def fallback():
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    if q != 3 or n != 7 or radius != 1:
        return fallback()
        
    total_words = q ** n
    # Precompute neighbors for all words
    # Map word to index
    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx
        
    def idx_to_word(idx):
        w = []
        for _ in range(n):
            w.append(idx % q)
            idx //= q
        return w[::-1]
        
    # Generate all words
    all_words = [idx_to_word(i) for i in range(total_words)]
    
    # Precompute neighbors (distance <= radius)
    # For radius 1, neighbors are the word itself + 1 word change in each of n positions
    neighbors = []
    for i in range(total_words):
        w = all_words[i]
        nbs = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_copy = w[:]
                w_copy[pos] = new_val
                nbs.append(word_to_idx(w_copy))
        neighbors.append(nbs)
        
    neighbors_arr = np.array(neighbors)
    neighbor_size = len(neighbors[0])
    
    def greedy_cover(randomize=False):
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        while uncovered.any():
            # Count how many uncovered words each candidate covers
            scores = uncovered[neighbors_arr].sum(axis=1)
            
            if randomize:
                max_score = scores.max()
                candidates = np.where(scores == max_score)[0]
                idx = np.random.choice(candidates)
            else:
                idx = np.argmax(scores)
                
            code.append(idx)
            uncovered[neighbors_arr[idx]] = False
            
        return code

    def reduce_code(code):
        code_set = set(code)
        cover_count = np.zeros(total_words, dtype=int)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
            
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                min_cov = cover_count[neighbors_arr[c]].min()
                if min_cov > 1:
                    cover_count[neighbors_arr[c]] -= 1
                    code_set.remove(c)
                    changed = True
        return list(code_set)
    
    def simulated_annealing(code, start_time, deadline):
        """SA local search: swap codewords to reduce size while maintaining coverage."""
        code_set = set(code)
        cover_count = np.zeros(total_words, dtype=int)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
        
        best_size = len(code_set)
        best_code = list(code_set)
        current_size = len(code_set)
        
        # Temperature schedule
        T_start = 2.0
        T_end = 0.1
        n_steps = 5000
        
        rng = random.Random(42)
        
        for step in range(n_steps):
            if time.time() > deadline:
                break
            
            # Cooling schedule
            alpha = step / n_steps
            T = T_start * (T_end / T_start) ** alpha
            
            # Try a random swap: remove one codeword, add another
            if current_size < 2:
                break
                
            # Pick a random codeword to potentially remove
            code_list = list(code_set)
            remove_idx = rng.randrange(len(code_list))
            remove_c = code_list[remove_idx]
            
            # Pick a random new codeword to add (not already in code)
            add_c = rng.randrange(total_words)
            if add_c in code_set:
                continue
            
            # Compute the effect of this swap
            # Remove remove_c: its neighbors lose one cover
            # Add add_c: its neighbors gain one cover
            
            # Check if after removal, any neighbor of remove_c would be uncovered
            # AND after addition, does it help?
            
            # Net change in coverage for each word:
            # Words in neighbors[remove_c] lose 1
            # Words in neighbors[add_c] gain 1
            
            # A swap is valid if no word ends up with cover_count 0
            # i.e., for words only in neighbors[remove_c] \ neighbors[add_c], cover_count must be >= 2
            # For words in neighbors[add_c] \ neighbors[remove_c], they gain coverage (always fine)
            # For words in intersection, net change is 0
            
            # Fast check: find words in neighbors[remove_c] that are NOT in neighbors[add_c]
            # and have cover_count == 1
            
            remove_nbs = neighbors_arr[remove_c]
            add_nbs = neighbors_arr[add_c]
            
            # Create a mask for words covered by remove_c but not add_c
            # We need to check if any of those have cover_count == 1
            
            # Efficient approach: check min cover count on remove_nbs, but only for those not in add_nbs
            # Since neighbor_size is small (1 + n*(q-1) = 1 + 7*2 = 15), we can do this directly
            
            # Build a set of add_nbs for quick lookup
            add_nbs_set = set(add_nbs.tolist())
            
            # Check words in remove_nbs that are not in add_nbs
            critical = False
            for w in remove_nbs:
                w_int = int(w)
                if w_int not in add_nbs_set:
                    if cover_count[w_int] <= 1:
                        critical = True
                        break
            
            if critical:
                # This swap would create an uncovered word - reject unless we accept it probabilistically
                # In SA, we can accept invalid states with probability exp(-delta/T)
                # delta = number of newly uncovered words
                # For simplicity, just reject invalid swaps in the main loop
                # But we can try to fix by adding extra codewords later
                continue
            
            # Perform the swap
            code_set.remove(remove_c)
            code_set.add(add_c)
            cover_count[remove_nbs] -= 1
            cover_count[add_nbs] += 1
            current_size = len(code_set)
            
            if current_size < best_size:
                best_size = current_size
                best_code = list(code_set)
            
            # Try to remove redundant codewords after each swap
            changed = True
            while changed:
                changed = False
                for c in list(code_set):
                    min_cov = cover_count[neighbors_arr[c]].min()
                    if min_cov > 1:
                        cover_count[neighbors_arr[c]] -= 1
                        code_set.remove(c)
                        changed = True
                        current_size = len(code_set)
                        if current_size < best_size:
                            best_size = current_size
                            best_code = list(code_set)
        
        return best_code
    
    best_code = None
    start_time = time.time()
    deadline = start_time + 110  # 40s margin before 150s limit
    
    # Run multiple rounds of greedy + reduction + SA
    for i in range(300):
        if time.time() > deadline:
            break
        if i % 5 == 0:
            cand = greedy_cover(randomize=False)
        else:
            cand = greedy_cover(randomize=True)
            
        cand = reduce_code(cand)
        
        # Apply simulated annealing to improve
        if time.time() < deadline - 5:  # Leave 5s margin
            cand = simulated_annealing(cand, start_time, deadline)
        
        if best_code is None or len(cand) < len(best_code):
            best_code = cand
            
    if best_code is None:
        best_code = greedy_cover(randomize=False)
        best_code = reduce_code(best_code)
        
    # Convert indices back to words
    return [all_words[i] for i in best_code]
# EVOLVE-BLOCK-END
