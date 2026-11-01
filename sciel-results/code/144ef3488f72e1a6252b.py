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
        
    neighbors_arr = np.array(neighbors, dtype=np.int32)
    neighbor_size = len(neighbors[0])  # 15 for q=3, n=7, r=1
    
    def greedy_cover(randomize=False, seed=None):
        uncovered = np.ones(total_words, dtype=bool)
        code = []
        rng = np.random.default_rng(seed) if seed is not None else np.random
        
        while uncovered.any():
            # Count how many uncovered words each candidate covers
            scores = uncovered[neighbors_arr].sum(axis=1)
            
            if randomize:
                max_score = scores.max()
                candidates = np.where(scores == max_score)[0]
                idx = rng.choice(candidates)
            else:
                idx = np.argmax(scores)
                
            code.append(int(idx))
            uncovered[neighbors_arr[idx]] = False
            
        return code

    def reduce_code(code):
        code_set = set(code)
        cover_count = np.zeros(total_words, dtype=np.int32)
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
    
    def simulated_annealing(code, deadline):
        """SA local search with swap moves and redundant removal."""
        code_set = set(code)
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
        
        best_size = len(code_set)
        best_code = list(code_set)
        current_size = len(code_set)
        
        # Temperature schedule
        T_start = 3.0
        T_end = 0.01
        n_steps = 8000
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
            
            remove_nbs = neighbors_arr[remove_c]
            add_nbs = neighbors_arr[add_c]
            
            # Fast validity check: no word in remove_nbs \ add_nbs should have cover_count == 1
            add_nbs_set = set(add_nbs.tolist())
            
            critical = False
            for w in remove_nbs:
                w_int = int(w)
                if w_int not in add_nbs_set:
                    if cover_count[w_int] <= 1:
                        critical = True
                        break
            
            if critical:
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
    
    def polish(code, deadline):
        """High-precision polish: aggressive local search with multiple restarts of greedy on uncovered."""
        code_set = set(code)
        cover_count = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cover_count[neighbors_arr[c]] += 1
        
        best_size = len(code_set)
        best_code = list(code_set)
        
        # Phase 1: Exhaustive redundant removal
        changed = True
        while changed:
            changed = False
            for c in list(code_set):
                min_cov = cover_count[neighbors_arr[c]].min()
                if min_cov > 1:
                    cover_count[neighbors_arr[c]] -= 1
                    code_set.remove(c)
                    changed = True
                    if len(code_set) < best_size:
                        best_size = len(code_set)
                        best_code = list(code_set)
        
        # Phase 2: Try all possible single swaps (limited by time)
        code_list = list(code_set)
        n = len(code_list)
        
        # Precompute which codewords are "critical" (have at least one neighbor with cover_count == 1)
        # For non-critical codewords, removal is always safe
        
        deadline_phase2 = min(deadline, time.time() + (deadline - time.time()) * 0.5)
        
        for i in range(n):
            if time.time() > deadline_phase2:
                break
            c_remove = code_list[i]
            remove_nbs = neighbors_arr[c_remove]
            
            # Check if c_remove is removable
            min_cov = cover_count[remove_nbs].min()
            if min_cov > 1:
                # Already handled in phase 1, skip
                continue
            
            # Try to find a replacement that covers the critical neighbors
            # Critical neighbors: those with cover_count == 1 in remove_nbs
            critical_mask = (cover_count[remove_nbs] == 1)
            critical_nbs = remove_nbs[critical_mask]
            
            if len(critical_nbs) == 0:
                continue
            
            # For each candidate to add, check if it covers all critical_nbs
            # and doesn't cause issues elsewhere
            # This is expensive, so limit candidates
            rng = random.Random(hash(c_remove) % 10000)
            n_attempts = 200
            for _ in range(n_attempts):
                if time.time() > deadline_phase2:
                    break
                c_add = rng.randrange(total_words)
                if c_add in code_set:
                    continue
                
                add_nbs = neighbors_arr[c_add]
                add_nbs_set = set(add_nbs.tolist())
                
                # Check all critical_nbs are covered by c_add
                all_covered = True
                for cn in critical_nbs:
                    if int(cn) not in add_nbs_set:
                        all_covered = False
                        break
                if not all_covered:
                    continue
                
                # Check no other word becomes uncovered
                # Words in remove_nbs but not in add_nbs with cover_count == 1 would be uncovered
                # But we already know critical_nbs are covered by add_c
                # Words in remove_nbs with cover_count > 1 are fine
                # So this swap is valid
                
                code_set.remove(c_remove)
                code_set.add(c_add)
                cover_count[remove_nbs] -= 1
                cover_count[add_nbs] += 1
                
                # Try to remove more redundant codewords
                changed = True
                while changed:
                    changed = False
                    for c in list(code_set):
                        min_cov2 = cover_count[neighbors_arr[c]].min()
                        if min_cov2 > 1:
                            cover_count[neighbors_arr[c]] -= 1
                            code_set.remove(c)
                            changed = True
                
                if len(code_set) < best_size:
                    best_size = len(code_set)
                    best_code = list(code_set)
                    code_list = list(code_set)
                    n = len(code_list)
                    break
                else:
                    # Revert
                    code_set.remove(c_add)
                    code_set.add(c_remove)
                    cover_count[add_nbs] -= 1
                    cover_count[remove_nbs] += 1
                    break
        
        return best_code
    
    best_code = None
    start_time = time.time()
    deadline = start_time + 115  # 35s margin before 150s limit
    
    # Run multiple rounds of greedy + reduction + SA + polish
    n_rounds = 50
    for i in range(n_rounds):
        if time.time() > deadline - 20:
            break
        
        # Vary the random seed for diversity
        seed = i * 7 + 13
        if i % 3 == 0:
            cand = greedy_cover(randomize=False)
        elif i % 3 == 1:
            cand = greedy_cover(randomize=True, seed=seed)
        else:
            cand = greedy_cover(randomize=True, seed=seed * 31)
            
        cand = reduce_code(cand)
        
        # Apply simulated annealing to improve
        remaining = deadline - time.time()
        if remaining > 10:
            sa_deadline = min(deadline - 5, time.time() + max(2, remaining * 0.4))
            cand = simulated_annealing(cand, sa_deadline)
        
        # Apply polish if time permits
        remaining = deadline - time.time()
        if remaining > 5:
            cand = polish(cand, deadline - 2)
        
        if best_code is None or len(cand) < len(best_code):
            best_code = cand
    
    if best_code is None:
        best_code = greedy_cover(randomize=False)
        best_code = reduce_code(best_code)
        
    # Final polish with whatever time is left
    remaining = deadline - time.time()
    if remaining > 2:
        best_code = polish(best_code, deadline - 1)
        
    # Convert indices back to words
    return [all_words[i] for i in best_code]
# EVOLVE-BLOCK-END
