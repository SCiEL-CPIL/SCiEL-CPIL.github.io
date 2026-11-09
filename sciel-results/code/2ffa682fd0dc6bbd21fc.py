import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # Specific parameters for this task: q=3, n=8, radius=1
    if not (q == 3 and n == 8 and radius == 1):
        # Fallback for other parameters
        # Use the seed logic (tetracode) if not the specific case, as it's guaranteed valid
        # For q=3, n=8, r=1, tetracode sum is 729.
        # For other cases, we might need a different fallback, but the task specifies this instance.
        # We'll just use a simple greedy for fallback to be safe, though it might be slow.
        total_words = q ** n
        if total_words > 10000:
             # Too big for this simple fallback, return seed-like or empty? 
             # The contract says "checked over the whole space". 
             # If we can't solve it, we should return a valid code.
             # The tetracode construction works for any n >= 4 if q=3.
             if q == 3 and n >= 4:
                tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
                return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
        # Generic small case fallback
        centers = []
        covered = set()
        all_words = set(itertools.product(range(q), repeat=n))
        candidates = list(all_words)
        random.shuffle(candidates)
        
        while covered != all_words and time.time() < deadline:
            best_center = None
            max_new = 0
            for c in candidates:
                neigh = [c]
                for i in range(n):
                    for val in range(q):
                        if val == c[i]: continue
                        new_c = list(c)
                        new_c[i] = val
                        neigh.append(tuple(new_c))
                new_covered = [w for w in neigh if w not in covered]
                if len(new_covered) > max_new:
                    max_new = len(new_covered)
                    best_center = c
                    best_neigh = neigh
            if best_center is None or max_new == 0:
                break
            centers.append(list(best_center))
            for w in best_neigh:
                covered.add(w)
            candidates.remove(best_center)
        return centers

    # --- Optimized Path for q=3, n=8, radius=1 ---
    
    total_words = 3 ** 8  # 6561
    
    # 1. Pre-compute mapping and masks
    word_to_idx = {}
    idx_to_word = [None] * total_words
    
    count = 0
    for word in itertools.product(range(3), repeat=8):
        idx = 0
        for digit in word:
            idx = idx * 3 + digit
        word_to_idx[word] = idx
        idx_to_word[idx] = word
        count += 1
        
    # masks[i] is a bitmask of all words covered by center i
    masks = [0] * total_words
    
    # Pre-compute neighbors list for fast count updates
    neighbors = [[] for _ in range(total_words)]
    
    for i in range(total_words):
        center = idx_to_word[i]
        mask_val = 1 << i
        neighbors[i].append(i)
        
        for pos in range(8):
            old_val = center[pos]
            for new_val in range(3):
                if new_val == old_val:
                    continue
                neighbor = list(center)
                neighbor[pos] = new_val
                n_idx = word_to_idx[tuple(neighbor)]
                mask_val |= (1 << n_idx)
                neighbors[i].append(n_idx)
                
        masks[i] = mask_val

    def run_greedy(uncovered_mask, seed_offset):
        """
        Runs a greedy set cover algorithm.
        Uses random tie-breaking among top candidates to diversify.
        """
        random.seed(seed_offset)
        code_indices = []
        current_uncovered = uncovered_mask
        
        while current_uncovered != 0:
            if time.time() > deadline:
                return None
            
            best_score = 0
            best_candidates = []
            
            # Find the best candidates
            # Optimization: We can skip this if we just pick the absolute best, 
            # but randomization helps escape local minima.
            for i in range(total_words):
                # If the center is already in code, its mask is fully covered? 
                # No, we don't track covered separately, just 'uncovered'.
                # A center in the code covers itself and neighbors.
                # If we pick a center already in the code, it adds 0 new coverage.
                # We should avoid picking centers already in the code if possible,
                # but the greedy naturally avoids them if they provide 0 new coverage.
                cnt = (current_uncovered & masks[i]).bit_count()
                if cnt > best_score:
                    best_score = cnt
                    best_candidates = [i]
                elif cnt == best_score and cnt > 0:
                    best_candidates.append(i)
            
            if not best_candidates:
                break # Should not happen if current_uncovered != 0
                
            # Pick randomly from top candidates
            # To ensure diversity, pick from top few if many ties?
            # Or just pick one random from the list of max-scoring ones.
            # The list might be large.
            chosen_idx = random.choice(best_candidates)
            
            code_indices.append(chosen_idx)
            current_uncovered &= ~masks[chosen_idx]
            
        return code_indices

    def minimize_code(code_indices):
        """
        Removes redundant centers from the code.
        """
        if not code_indices:
            return code_indices
            
        # counts[w] = number of centers covering w
        counts = [0] * total_words
        
        # Helper to update counts
        def add_center_counts(idx):
            for n_idx in neighbors[idx]:
                counts[n_idx] += 1
                
        def remove_center_counts(idx):
            for n_idx in neighbors[idx]:
                counts[n_idx] -= 1

        # Initialize counts
        for idx in code_indices:
            add_center_counts(idx)
            
        code_set = set(code_indices)
        removed_any = True
        while removed_any and time.time() < deadline:
            removed_any = False
            # Iterate over a copy of the list
            for idx in list(code_indices):
                # Check if idx can be removed
                # Temporarily remove
                remove_center_counts(idx)
                
                can_remove = True
                # Check if any neighbor has count 0
                for n_idx in neighbors[idx]:
                    if counts[n_idx] == 0:
                        can_remove = False
                        break
                        
                if can_remove:
                    code_indices.remove(idx)
                    code_set.remove(idx)
                    removed_any = True
                else:
                    # Restore counts
                    add_center_counts(idx)
                    
        return code_indices

    # 2. Main Search Loop
    best_code = None
    best_len = float('inf')
    
    # Initial full mask
    full_mask = (1 << total_words) - 1
    
    # Run multiple randomized greedy instances
    # Estimate time per run: ~1s
    # We have ~110s. Let's try 50 runs.
    num_restarts = 50 
    
    for i in range(num_restarts):
        if time.time() > deadline - 5: # Leave 5s for final cleanup
            break
            
        code = run_greedy(full_mask, i)
        
        if code is None:
            continue
            
        # Minimize
        code = minimize_code(code)
        
        if len(code) < best_len:
            best_len = len(code)
            best_code = code
            
        # Early exit if we hit the theoretical lower bound or known optimum
        if best_len <= 486:
            break

    # If no code was found (shouldn't happen), fallback to seed
    if best_code is None:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    # Convert indices back to list of lists
    result = [list(idx_to_word[idx]) for idx in best_code]
    
    # Final validity check (optional, but good for safety)
    # For this size, we can verify quickly
    # But we trust the bitmask logic.
    
    return result
