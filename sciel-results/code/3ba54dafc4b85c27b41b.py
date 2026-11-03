import itertools
import time

def construct(q, n, radius):
    # For this task, q=3, n=7, radius=1.
    # The problem is small enough to use bit-parallel set cover.
    
    if q != 3 or n != 7 or radius != 1:
        # Fallback to seed for other parameters (though task is specific)
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    total_words = q ** n
    # Precompute the mask for every possible codeword
    # Index of a word w = w[0]*3^(n-1) + ... + w[n-1]
    
    def word_to_idx(w):
        idx = 0
        for i in range(n):
            idx = idx * q + w[i]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Generate masks for all codewords
    # mask[c] is an integer with bits set for all words within radius 1 of c
    masks = [0] * total_words
    
    for c_idx in range(total_words):
        c = idx_to_word(c_idx)
        mask = 0
        # The word itself
        mask |= 1 << c_idx
        # Words at distance 1
        for i in range(n):
            for delta in [-1, 1]:
                new_val = c[i] + delta
                if 0 <= new_val < q:
                    c[i] = new_val
                    n_idx = word_to_idx(c)
                    mask |= 1 << n_idx
                    c[i] = c[i] - delta # restore
        masks[c_idx] = mask

    # Initial solution: The seed's 189 codewords
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_codewords.append(h + list(tail))
    
    # Convert to indices
    current_solution = [word_to_idx(w) for w in initial_codewords]
    
    # Compute current coverage
    covered_mask = 0
    for c_idx in current_solution:
        covered_mask |= masks[c_idx]
        
    full_mask = (1 << total_words) - 1
    
    # Check if initial is valid (it should be)
    if covered_mask != full_mask:
        # Fallback if something went wrong
        return initial_codewords

    best_solution = list(current_solution)
    best_size = len(current_solution)
    
    start_time = time.time()
    deadline = start_time + 145 # Safety margin
    
    # Phase 1: Redundancy Removal
    def remove_redundant(sol, cov_mask):
        changed = True
        while changed:
            changed = False
            new_sol = []
            for c_idx in sol:
                # Check if removing c_idx leaves all words covered
                # cov_mask without masks[c_idx]
                if (cov_mask & ~masks[c_idx]) == cov_mask:
                    # c_idx is redundant
                    changed = True
                    # Do not add to new_sol
                else:
                    new_sol.append(c_idx)
                    # Note: cov_mask remains the same because we only remove redundant ones
                    # whose contribution is fully covered by others.
            sol = new_sol
            # Recalculate cov_mask just in case, though it shouldn't change if logic is right
            # Actually, if we remove c, cov_mask is still the same union of the rest.
            # So we don't need to recalculate cov_mask inside the loop if we are careful.
            # But wait, if we remove c, the "cov_mask" variable still holds the old union.
            # Since c was redundant, the union of the rest is the same as the old union.
            # So cov_mask is still correct for the new sol.
        return sol, cov_mask

    current_solution, covered_mask = remove_redundant(current_solution, covered_mask)
    
    if len(current_solution) < best_size:
        best_size = len(current_solution)
        best_solution = list(current_solution)

    # Phase 2: Local Search
    # We try to swap a codeword c with another c' such that:
    # 1. c' is not in current_solution
    # 2. c' covers all "holes" created by removing c
    # 3. After swap, we can remove more redundancies to reduce size
    
    while time.time() < deadline:
        improved = False
        # Shuffle solution to avoid bias? Or iterate in fixed order?
        # Fixed order is faster.
        for i, c_idx in enumerate(current_solution):
            if time.time() > deadline:
                break
                
            # Identify holes if we remove c_idx
            holes = covered_mask & ~masks[c_idx]
            
            if holes == 0:
                # Should have been removed in Phase 1, but just in case
                new_sol = current_solution[:i] + current_solution[i+1:]
                # Recalculate mask? No, holes==0 means covered_mask is unchanged by removal.
                current_solution = new_sol
                improved = True
                continue

            # Find a candidate c' that covers all holes
            # We need masks[c'] & holes == holes
            # To speed up, we can just iterate all possible c'
            # There are 2187 candidates.
            
            # Optimization: We can precompute for each bit position? No, simple loop is fine.
            found_swap = False
            for c_prime in range(total_words):
                if c_prime in current_solution:
                    continue
                if (masks[c_prime] & holes) == holes:
                    # Perform swap
                    # Remove c_idx, add c_prime
                    new_sol = current_solution[:i] + [c_prime] + current_solution[i+1:]
                    
                    # Calculate new covered mask
                    # It's faster to recalculate from scratch for ~180 items?
                    # Or update: new_cov = (covered_mask & ~masks[c_idx]) | masks[c_prime]
                    new_cov = (covered_mask & ~masks[c_idx]) | masks[c_prime]
                    
                    # Check validity (should be valid by construction of holes, but let's be sure)
                    if new_cov != full_mask:
                        continue
                        
                    # Try to remove redundancies from new_sol
                    new_sol_clean, new_cov_clean = remove_redundant(new_sol, new_cov)
                    
                    if len(new_sol_clean) < len(current_solution):
                        # Accept the swap
                        current_solution = new_sol_clean
                        covered_mask = new_cov_clean
                        improved = True
                        if len(current_solution) < best_size:
                            best_size = len(current_solution)
                            best_solution = list(current_solution)
                        found_swap = True
                        break
                    # If size didn't decrease, we might still accept it if it enables future decreases?
                    # For simplicity and speed, we only accept if it immediately reduces size.
                    # This is a strict hill climb.
            
            if found_swap:
                # Restart the loop over the solution as indices/contents changed
                # The for loop variable i is now stale.
                break
                
        if not improved:
            # Try a random restart or different strategy?
            # If we are stuck, we can try to randomize the order of removal/swap
            # Or break if we are confident.
            # Let's try a few random swaps even if they don't immediately reduce size?
            # That's risky for time.
            # Let's just break if no improvement in a full pass.
            break

    # Convert best_solution indices back to words
    result = [idx_to_word(idx) for idx in best_solution]
    return result
