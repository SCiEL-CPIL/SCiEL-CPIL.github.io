import time
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s margin
    
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Fallback
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _solve_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n
    ball_size = 1 + 2 * n
    
    # Precompute balls as indices
    # weights[i] is the multiplier for position i
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    balls = np.zeros((total, ball_size), dtype=np.int32)
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
                
    # 1. Greedy Initialization
    cov = np.zeros(total, dtype=np.int32)
    selected = []
    uncovered = np.ones(total, dtype=bool)
    
    # Precompute candidate scores for greedy? 
    # Just simple greedy: pick word with max uncovered neighbors
    while np.any(uncovered) and time.time() < deadline - 20.0:
        # Count uncovered neighbors for each word
        # uncovered[balls] gives a (total, ball_size) boolean array
        # Sum along axis 1 gives count of uncovered neighbors
        counts = np.sum(uncovered[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        # Pick smallest index with max count for determinism
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        
        selected.append(best_idx)
        cov[balls[best_idx]] += 1
        uncovered = (cov == 0)
        
    # If greedy timed out, we have a partial cover. We must complete it.
    # But greedy is fast (usually < 1s for 6561 words).
    
    # 2. Local Search: Remove and Swap
    # We maintain 'selected' as a list of indices.
    # We maintain 'cov' as the coverage counts.
    # We maintain 'in_code' boolean array for fast lookup.
    
    in_code = np.zeros(total, dtype=bool)
    for c in selected:
        in_code[c] = True
        
    # Helper to update cov and in_code
    def add_code(c):
        cov[balls[c]] += 1
        in_code[c] = True
        selected.append(c)
        
    def remove_code(c):
        cov[balls[c]] -= 1
        in_code[c] = False
        # Remove from selected list
        # To keep it fast, we can mark as removed and clean up later,
        # or use a set for membership and list for iteration.
        # Given small size, list removal is O(N), which is fine for 500 items.
        selected.remove(c)
        
    # Optimization: Use a set for 'in_code' checks, but numpy bool array is faster for vector ops.
    # We will keep 'selected' as a list, but use 'in_code' for O(1) checks.
    # When removing, we just set in_code[c]=False and decrement cov.
    # We don't actually need to remove from 'selected' immediately if we filter at the end,
    # but for the swap logic, we need to know which are active.
    
    # Let's use a different structure:
    # active_codes: list of indices
    # We will iterate over a copy of active_codes.
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        
        # --- Phase 1: Remove redundant codewords ---
        # A codeword c is removable if min(cov[balls[c]]) > 1
        # We can check this for all active codes at once?
        # active_codes is a list.
        # Let's check one by one, but in a randomized order to break symmetry?
        # Deterministic order is fine.
        
        # To speed up, we can check if ANY code is removable.
        # We can vectorize the check for all active codes if we store them in an array.
        # But the set of active codes changes.
        
        # Simple loop:
        for c in list(selected):
            if not in_code[c]:
                continue
            if np.min(cov[balls[c]]) > 1:
                remove_code(c)
                improved = True
                # After removal, cov changes, so other codes might become removable.
                # We continue the loop.
                
        # --- Phase 2: Swap ---
        # For each active code c, try to find a better replacement w.
        # Condition for w to be a valid swap for c:
        # 1. w is not in code.
        # 2. All words uniquely covered by c must be covered by w.
        #    Unique words for c: U_c = {v in ball[c] | cov[v] == 1}
        #    So U_c must be subset of ball[w].
        
        # We can precompute for each c, the set of words that MUST be covered by the replacement.
        # This is U_c.
        
        # To find valid w:
        # w must satisfy: ball[w] contains U_c.
        
        # We can iterate through all w not in code?
        # That's 6561 * 500 = 3M checks.
        # Each check: `np.all(np.isin(U_c, balls[w]))`
        # This is expensive in Python.
        
        # Optimization:
        # Instead of iterating all w, we can precompute for each word v, the list of words w that cover v.
        # Let `coverers[v]` be the list of words w such that v in ball[w].
        # Note: coverers[v] is just ball[v]! Because distance is symmetric.
        # So w is a valid swap for c iff:
        # For all v in U_c, w in ball[v].
        # i.e., w is in the intersection of ball[v] for all v in U_c.
        
        # So, valid_w_candidates = intersection_{v in U_c} ball[v]
        # And w must not be in code.
        
        # If U_c is empty, c is removable (handled in Phase 1).
        # So U_c is not empty.
        
        # We can compute the intersection of balls for v in U_c.
        # Since |U_c| is small (usually 1-3), this is fast.
        
        for c in list(selected):
            if not in_code[c]:
                continue
                
            # Find unique words covered by c
            # cov[balls[c]] gives coverage of neighbors.
            # We want neighbors with cov == 1.
            # Note: cov[c] is 1 (covered by itself).
            # If cov[c] == 1, c is uniquely covered by itself?
            # Yes, if no other code covers c.
            
            unique_neighbors = balls[c][cov[balls[c]] == 1]
            
            if len(unique_neighbors) == 0:
                # Should have been removed in Phase 1
                continue
                
            # Find intersection of balls for these unique neighbors
            # Start with first neighbor's ball
            valid_w_set = set(balls[unique_neighbors[0]].tolist())
            for v in unique_neighbors[1:]:
                valid_w_set.intersection_update(balls[v].tolist())
                
            # Filter out words already in code
            # valid_w_set = {w for w in valid_w_set if not in_code[w]}
            
            # Find the best w in valid_w_set
            # Heuristic: Choose w that maximizes redundancy.
            # Score(w) = sum(1 for v in ball[w] if cov[v] > 1)
            # Or simply: Pick the first one? Or random?
            # Let's pick the one that covers the most "highly covered" words.
            
            best_w = -1
            best_score = -1
            
            # Convert to list for iteration
            w_list = [w for w in valid_w_set if not in_code[w]]
            
            if not w_list:
                continue
                
            # To avoid O(N) Python loop, we can vectorize score calculation if we have many candidates.
            # But w_list is usually small (intersection of 2-3 balls of size 17).
            # |ball| = 17. Intersection of 2 is ~ small.
            # So w_list is very small.
            
            for w in w_list:
                # Calculate score
                # Score = number of neighbors of w that are covered by > 1 code
                score = np.sum(cov[balls[w]] > 1)
                if score > best_score:
                    best_score = score
                    best_w = w
                    
            if best_w != -1:
                # Perform swap
                remove_code(c)
                add_code(best_w)
                improved = True
                # Break and restart? Or continue?
                # Continuing might be fine, but cov changed.
                # Let's break to re-evaluate all codes with new cov.
                break
                
    # 3. Final Verification and Cleanup
    # Remove any remaining redundant codes
    for c in list(selected):
        if np.min(cov[balls[c]]) > 1:
            remove_code(c)
            
    # Verify coverage
    final_cov = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cov[balls[c]] += 1
        
    if np.any(final_cov == 0):
        # Add missing words
        missing = np.where(final_cov == 0)[0]
        for m in missing:
            add_code(m)
            final_cov[balls[m]] += 1
            
    # Convert to lists of digits
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
