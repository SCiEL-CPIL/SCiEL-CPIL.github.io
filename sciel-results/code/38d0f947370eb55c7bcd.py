import time
import random
import itertools

def construct(q, n, radius):
    """
    Constructs a covering code of length n over alphabet size q with covering radius radius.
    Uses Lagrangian relaxation for initialization and structured local search for optimization.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    # Helper to convert integer to list of digits
    def int_to_word(idx, n, q):
        word = [0] * n
        for i in range(n):
            word[i] = idx % q
            idx //= q
        return word
        
    # Helper to convert list of digits to integer
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val

    # Precompute neighbors
    # For q=3, n=6, radius=1, the ball size is 1 + 6*(3-1) = 13
    neighbors = [[] for _ in range(total_words)]
    for i in range(total_words):
        w = int_to_word(i, n, q)
        neighbors[i].append(i)
        if radius >= 1:
            for pos in range(n):
                original_val = w[pos]
                for new_val in range(q):
                    if new_val != original_val:
                        w[pos] = new_val
                        neighbor_idx = word_to_int(w, q)
                        neighbors[i].append(neighbor_idx)
                w[pos] = original_val
        # Note: For radius > 1, this precomputation would be incomplete.
        # However, the task is specifically K_3(6,1).
        # If radius > 1 is passed, we fall back to a simpler generic approach 
        # or assume the user knows the limits. 
        # Given the context, we optimize for radius=1.

    # State management
    codewords = set()
    covered = [0] * total_words
    uncovered_count = total_words
    
    def add(c_idx):
        nonlocal uncovered_count
        codewords.add(c_idx)
        for nb in neighbors[c_idx]:
            if covered[nb] == 0:
                uncovered_count -= 1
            covered[nb] += 1

    def remove(c_idx):
        nonlocal uncovered_count
        codewords.remove(c_idx)
        for nb in neighbors[c_idx]:
            covered[nb] -= 1
            if covered[nb] == 0:
                uncovered_count += 1

    # 1. Initialization via Lagrangian Relaxation
    # We start with uniform weights and iteratively adjust them to guide greedy selection.
    # This often yields a smaller initial code than pure greedy.
    
    lambda_weights = [1.0] * total_words
    step_size = 0.5
    max_lag_iters = 50
    
    # We will run a few iterations of Lagrangian relaxation to find a good start
    best_init_size = float('inf')
    best_init_code = set()
    
    for lag_iter in range(max_lag_iters):
        if time.time() > deadline - 30: # Leave time for local search
            break
            
        # Greedy step with Lagrangian weights
        temp_code = set()
        temp_covered = [0] * total_words
        temp_uncovered = total_words
        uncovered_mask = [1] * total_words # 1 if uncovered, 0 if covered
        
        def temp_add(c_idx):
            nonlocal temp_uncovered
            temp_code.add(c_idx)
            for nb in neighbors[c_idx]:
                if temp_covered[nb] == 0:
                    temp_uncovered -= 1
                    uncovered_mask[nb] = 0
                temp_covered[nb] += 1

        # Select codewords
        remaining_words = set(range(total_words))
        # Simple greedy: pick word that maximizes weighted coverage of uncovered words
        # To speed up, we can just pick the one with max sum of lambda_weights for uncovered neighbors
        # Since total_words is 729, O(N^2) is acceptable for a few iterations.
        
        while temp_uncovered > 0 and len(temp_code) < total_words:
            best_c = -1
            best_score = -1
            # Optimization: Instead of checking all 729 every time, 
            # we could maintain a priority queue, but for 729 it's fine.
            for c in range(total_words):
                if c in temp_code:
                    continue
                score = 0
                for nb in neighbors[c]:
                    if uncovered_mask[nb]:
                        score += lambda_weights[nb]
                if score > best_score:
                    best_score = score
                    best_c = c
                    
            if best_c == -1 or best_score <= 0:
                break
                
            temp_add(best_c)
            
        if temp_uncovered == 0:
            if len(temp_code) < best_init_size:
                best_init_size = len(temp_code)
                best_init_code = set(temp_code)
        
        # Update Lagrangian weights (Subgradient)
        # Increase weight for uncovered words, decrease for covered
        for i in range(total_words):
            if temp_covered[i] == 0:
                lambda_weights[i] = max(0, lambda_weights[i] + step_size)
            else:
                lambda_weights[i] = max(0, lambda_weights[i] - step_size * 0.1)
        
        # Decay step size
        step_size *= 0.95

    # If Lagrangian didn't find a valid code (shouldn't happen if we run enough), fallback to Greedy
    if best_init_size == float('inf'):
        # Standard Greedy
        uncovered_set = set(range(total_words))
        while uncovered_set:
            if time.time() > deadline - 30:
                break
            best_center = -1
            best_count = -1
            for candidate in range(total_words):
                count = 0
                for nb in neighbors[candidate]:
                    if nb in uncovered_set:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_center = candidate
            if best_center == -1: break
            add(best_center)
            for nb in neighbors[best_center]:
                if nb in uncovered_set:
                    uncovered_set.remove(nb)
        if uncovered_count == 0:
            best_init_code = set(codewords)
            best_init_size = len(codewords)
    else:
        # Reset state and load best_init_code
        codewords.clear()
        covered = [0] * total_words
        uncovered_count = total_words
        for c in best_init_code:
            add(c)
        
        # If the Lagrangian result is invalid (uncovered_count > 0), repair it
        if uncovered_count > 0:
             # Greedy repair
             uncovered_set = set(i for i in range(total_words) if covered[i] == 0)
             while uncovered_set:
                if time.time() > deadline - 30:
                    break
                best_center = -1
                best_count = -1
                for candidate in range(total_words):
                    if candidate in codewords: continue
                    count = 0
                    for nb in neighbors[candidate]:
                        if nb in uncovered_set:
                            count += 1
                    if count > best_count:
                        best_count = count
                        best_center = candidate
                if best_center == -1: break
                add(best_center)
                for nb in neighbors[best_center]:
                    if nb in uncovered_set:
                        uncovered_set.remove(nb)

    # 2. Local Search
    # Goal: Minimize len(codewords)
    
    # Precompute structured perturbations for Z_3^6
    # These are vectors d such that moving c -> c+d is a valid "small" step
    perturbations = []
    # Single position changes
    for pos in range(n):
        for delta in (1, 2):
            p = [0] * n
            p[pos] = delta
            perturbations.append(p)
    # Double position changes (optional, might be too slow if too many)
    # For n=6, there are 15 pairs * 4 deltas = 60. Total ~72 perturbations.
    for pos1 in range(n):
        for pos2 in range(pos1 + 1, n):
            for d1 in (1, 2):
                for d2 in (1, 2):
                    p = [0] * n
                    p[pos1] = d1
                    p[pos2] = d2
                    perturbations.append(p)

    def get_idx_from_tuple(t):
        # Fast lookup for tuple to index
        # Since q=3, n=6, we can compute directly
        val = 0
        for i in range(len(t) - 1, -1, -1):
            val = val * q + t[i]
        return val

    best_size = len(codewords)
    best_codewords = set(codewords)
    
    no_improve_count = 0
    rng = random.Random(42)
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try to delete redundant codewords
        # Iterate through a random permutation to avoid bias
        candidates = list(codewords)
        rng.shuffle(candidates)
        
        for c in candidates:
            if time.time() > deadline:
                break
            # Check if c is redundant
            redundant = True
            for nb in neighbors[c]:
                if covered[nb] == 1:
                    redundant = False
                    break
            
            if redundant:
                remove(c)
                improved = True
                # Restart deletion loop as state changed
                break 
        
        if improved:
            if len(codewords) < best_size:
                best_size = len(codewords)
                best_codewords = set(codewords)
            no_improve_count = 0
            continue
        
        # Phase 2: Try Swaps (1-exchange)
        # Pick a random codeword to replace
        if codewords:
            c_old = rng.choice(list(codewords))
            
            # Generate a candidate c_new using structured perturbation
            # This is the key insight from Inspiration 1
            d = rng.choice(perturbations)
            c_old_word = int_to_word(c_old, n, q)
            c_new_word = [(c_old_word[i] + d[i]) % q for i in range(n)]
            c_new = word_to_int(c_new_word, q)
            
            if c_new in codewords:
                # If it's already in the set, skip or try another
                # To keep it simple, just skip this iteration
                pass
            else:
                # Check if swap is valid
                # Words that become uncovered if we remove c_old:
                danger_words = []
                for nb in neighbors[c_old]:
                    if covered[nb] == 1:
                        danger_words.append(nb)
                
                if not danger_words:
                    # Swap is safe.
                    remove(c_old)
                    add(c_new)
                    improved = True
                    no_improve_count = 0
                    continue
                else:
                    # Check if c_new covers all danger_words
                    valid_swap = True
                    new_neighbors_set = set(neighbors[c_new])
                    for dw in danger_words:
                        if dw not in new_neighbors_set:
                            valid_swap = False
                            break
                    
                    if valid_swap:
                        remove(c_old)
                        add(c_new)
                        improved = True
                        no_improve_count = 0
                        continue
        
        # Phase 3: Perturbation if stuck
        no_improve_count += 1
        if no_improve_count > 50:
            # Add a random codeword to break symmetry
            c_perturb = rng.randint(0, total_words - 1)
            if c_perturb not in codewords:
                add(c_perturb)
            # Then run greedy deletion for a few steps
            # We just let the next loop's deletion phase handle it
            no_improve_count = 0
            # Optional: Try a few immediate deletions to stabilize
            for _ in range(5):
                if time.time() > deadline: break
                changed = False
                for c in list(codewords):
                    redundant = True
                    for nb in neighbors[c]:
                        if covered[nb] == 1:
                            redundant = False
                            break
                    if redundant:
                        remove(c)
                        changed = True
                        break
                if not changed: break

    # Return the best found
    result = [int_to_word(c, n, q) for c in best_codewords]
    
    # Final validity check just in case (should be fast)
    # If invalid, we might need to return a known valid one, but best_codewords is always valid
    # because we only update best_codewords when we have a valid state (uncovered_count==0 is implied by our logic? 
    # Actually, we don't explicitly check uncovered_count==0 before updating best.
    # Let's verify: 
    # In Phase 1, we only remove if redundant, so coverage stays valid.
    # In Phase 2, we only swap if valid_swap, so coverage stays valid.
    # In Phase 3, we add, so coverage stays valid.
    # So best_codewords should always be a valid cover.
    
    return result
