import time
import numpy as np
import random
import itertools
import os

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallbacks for other parameters (not expected in this task, but good practice)
    if q == 3 and n >= 4:
        # Simple tetracode extension, not optimal but valid
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    
    # Precompute weights for base-3 digits
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Precompute balls (neighborhoods of radius 1)
    # balls[w] is a list of 17 indices: w itself and its 16 neighbors
    balls = np.zeros((total, 1 + 2 * n), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
                
    # Precompute ball_sets for O(1) membership checks in swap logic
    # This is a list of sets, where ball_sets[w] contains all indices in balls[w]
    ball_sets = [set(row) for row in balls]
    
    all_indices = np.arange(total, dtype=np.int32)
    
    best_size = total
    best_set = None
    
    # We will run multiple iterations of Greedy + Local Search
    iteration = 0
    while time.time() < deadline - 5.0:
        iteration += 1
        
        # 1. Randomized Greedy Construction
        # To get diverse starts, we can randomize the tie-breaking in greedy
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        
        # Greedy Loop
        while np.any(uncovered_mask):
            if time.time() > deadline - 10.0:
                break # Time out for this iteration
                
            # Count coverage for each candidate among uncovered words
            # counts[c] = number of uncovered words in balls[c]
            counts = uncovered_mask[balls].sum(axis=1)
            
            if np.max(counts) == 0:
                break # Should not happen if uncovered_mask has True
                
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            
            # Random selection among best to encourage diversity
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            
            # Update cover counts
            cover_count[balls[best_idx]] += 1
            
            # Update uncovered mask
            uncovered_mask = (cover_count == 0)
            
        if np.any(uncovered_mask):
            continue # Invalid solution due to timeout in greedy
            
        # 2. Local Search
        # Phase 1: Remove redundant codewords
        # We iterate until no more can be removed
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                # Check if c is removable: all words in balls[c] are covered by >= 2 codewords
                # i.e., min(cover_count[balls[c]]) > 1
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # Phase 2: Swap / Replace
        # Try to replace a codeword c with c' if it leads to a better state
        # Valid swap condition:
        # Let S_c = {x in balls[c] | cover_count[x] == 1} (critical words for c)
        # If we remove c, cover_count[x] becomes 0 for x in S_c.
        # We need c' to cover all x in S_c.
        # i.e., S_c subset of balls[c']
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            new_selected = []
            for c in selected:
                # First, try to remove (might have become removable due to previous swaps in this pass)
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue
                
                # Try to swap c with some c'
                swapped = False
                # Heuristic: Look for c' that covers the critical set of c
                # Critical set: words covered ONLY by c
                critical_mask = (cover_count[balls[c]] == 1)
                critical_words = balls[c][critical_mask]
                
                if len(critical_words) == 0:
                    # Should have been removed, but just in case
                    cover_count[balls[c]] -= 1
                    improved = True
                    continue

                # Try a few random candidates for c'
                # Use precomputed ball_sets for fast membership checks
                num_trials = 50
                for _ in range(num_trials):
                    if time.time() > deadline - 5.0:
                        break
                    c_prime = all_indices[np.random.randint(total)]
                    if c_prime == c:
                        continue
                    
                    # Check if balls[c_prime] covers all critical_words using set membership
                    valid = True
                    for cw in critical_words:
                        if cw not in ball_sets[c_prime]:
                            valid = False
                            break
                            
                    if valid:
                        # Swap is valid
                        cover_count[balls[c]] -= 1
                        cover_count[balls[c_prime]] += 1
                        new_selected.append(c_prime)
                        swapped = True
                        improved = True
                        break
                        # Note: We must NOT revert cover_count if we accept the swap.
                        # But we haven't updated new_selected yet.
                        # We appended c_prime. We skipped c.
                        # The cover_count update is correct.
                    
                if not swapped:
                    new_selected.append(c)
            
            selected = new_selected
            
        # Phase 3: 2-Exchange (Remove 2, Add 1)
        # If len(selected) is still > 486, try this to escape local minima
        if len(selected) > 486 and time.time() < deadline - 5.0:
            # Random sampling based 2-exchange
            # Try to find c1, c2 in selected and c3 not in selected
            # such that removing c1, c2 and adding c3 is valid
            # Validity condition:
            # Let S1 = critical words of c1 (covered only by c1)
            # Let S2 = critical words of c2 (covered only by c2)
            # Let S12 = (S1 ∪ S2) \ (S1 ∩ S2) ... wait, we need to be careful.
            # If we remove c1 and c2, the words that become uncovered are those with cover_count == 1 
            # and covered by c1 or c2, AND not covered by any other codeword.
            # Actually, simpler:
            # Let U1 = {x in balls[c1] | cover_count[x] == 1}
            # Let U2 = {x in balls[c2] | cover_count[x] == 1}
            # If we remove c1, U1 becomes uncovered.
            # If we remove c2, U2 becomes uncovered.
            # But if x is in both U1 and U2, it's covered by both c1 and c2? 
            # No, cover_count[x] == 1 means it's covered by exactly one codeword.
            # So U1 and U2 are disjoint.
            # The set of words that become uncovered is U1 ∪ U2.
            # We need c3 to cover all words in U1 ∪ U2.
            
            # Also, we must ensure that c3 does not "break" other words?
            # No, adding c3 only increases cover counts, so it can't make a covered word uncovered.
            # The only risk is if c3 is already in selected? No, we pick c3 not in selected.
            
            # Let's try a few random pairs.
            n_selected = len(selected)
            if n_selected >= 2:
                for _ in range(100): # Limit trials
                    if time.time() > deadline - 5.0:
                        break
                    
                    # Pick two random distinct indices from selected
                    i1 = np.random.randint(n_selected)
                    i2 = np.random.randint(n_selected)
                    if i1 == i2:
                        continue
                    c1 = selected[i1]
                    c2 = selected[i2]
                    
                    # Calculate U1 and U2
                    U1 = balls[c1][(cover_count[balls[c1]] == 1)]
                    U2 = balls[c2][(cover_count[balls[c2]] == 1)]
                    
                    # Combine
                    U_combined = np.concatenate([U1, U2])
                    
                    if len(U_combined) == 0:
                        # Both removable? Should have been caught by Phase 1/2.
                        # But maybe they became removable due to each other? No, removal is independent in this check.
                        # If U1 and U2 are empty, we can just remove them.
                        # This is handled by Phase 1/2, but if we get here, let's just remove.
                        cover_count[balls[c1]] -= 1
                        cover_count[balls[c2]] -= 1
                        # Remove from selected
                        # This is messy to do in place. Let's just skip this case and let Phase 1/2 handle it.
                        # Actually, if U1 and U2 are empty, min(cover_count[balls[c]]) > 1 for both.
                        # So Phase 1 should have removed them.
                        # If we are here, it means something else. Let's just break.
                        break
                    
                    # Pick c3 randomly
                    c3 = all_indices[np.random.randint(total)]
                    if c3 in selected: # This check is O(N), might be slow. 
                        # Use a set for selected? 
                        # For now, let's assume random c3 is likely not in selected (6561 size, ~500 selected)
                        # But we need to be sure.
                        # Let's use a set for selected for this check.
                        pass
                    
                    # Check if c3 covers all U_combined
                    valid = True
                    for u in U_combined:
                        if u not in ball_sets[c3]:
                            valid = False
                            break
                    
                    if valid:
                        # Check if c3 is already in selected
                        if c3 in selected:
                            continue
                            
                        # Perform swap
                        cover_count[balls[c1]] -= 1
                        cover_count[balls[c2]] -= 1
                        cover_count[balls[c3]] += 1
                        
                        # Update selected list
                        # Remove c1 and c2, add c3
                        # This is O(N). 
                        # Since we are in a loop, we need to be careful with indices.
                        # Let's create a new list.
                        new_selected = []
                        for c in selected:
                            if c != c1 and c != c2:
                                new_selected.append(c)
                        new_selected.append(c3)
                        selected = new_selected
                        improved = True
                        break # Break out of 2-exchange trials, restart Phase 2? 
                        # Or just continue Phase 2? 
                        # Let's set improved=True so Phase 2 loops again.
                        # But we broke out of the for loop. 
                        # We need to signal that we made a change.
                        # The `improved` flag is used in the while loop condition.
                        # So setting it to True will cause the while loop to run again.
                        # This is good because the removal of c1, c2 might make others removable.

        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 403:
                break # Hit lower bound (unlikely)
            if best_size <= 486:
                # We hit the known best. We can try to improve further or break.
                # Let's keep searching for a bit to see if we can go lower.
                # But 486 is the proven best for a while.
                # Actually, 486 is the BEST KNOWN. If we find 486, we are at the state-of-the-art.
                # We should break to save time for other restarts or just return.
                # But maybe we can find 485? Unlikely.
                # Let's break if we find <= 486 to ensure we return a valid good answer quickly.
                break

    # Convert best_set to output format
    if best_set is None:
        # Fallback: Return a valid (though large) code
        # Simple greedy without local search
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered_mask = np.ones(total, dtype=bool)
        while np.any(uncovered_mask):
            counts = uncovered_mask[balls].sum(axis=1)
            max_count = np.max(counts)
            candidates = np.where(counts == max_count)[0]
            best_idx = candidates[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            uncovered_mask = (cover_count == 0)
        best_set = selected

    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
