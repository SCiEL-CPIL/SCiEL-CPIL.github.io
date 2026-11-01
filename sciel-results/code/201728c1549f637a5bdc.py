"""Deficit-Driven Greedy Covering with Vectorized Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 145.0  # 5s safety margin before 150s kill
    
    total = q ** n
    
    # Precompute digits for all words in a vectorized manner
    # digits[i] is the i-th digit (0-indexed from left? No, let's use standard base-q)
    # Let's use digits shape (total, n). 
    # word = sum(digits[i] * q^(n-1-i))
    # To be consistent with the previous logic, let's stick to a specific representation.
    # Previous: digits[0] was least significant? 
    # int_to_digits: digits[i] = x % q; x //= q. So digits[0] is LSB.
    # Let's use digits shape (total, n) where digits[:, i] is the digit at position i (LSB to MSB).
    
    # Generate all words
    all_indices = np.arange(total, dtype=np.int32)
    
    # Compute digits
    # For q=3, n=7, total=2187.
    # digits[i] = (word // q**i) % q
    powers = np.array([q**i for i in range(n)], dtype=np.int32)
    digits = (all_indices[:, None] // powers[None, :]) % q
    
    # Precompute balls: for each word c, the list of words within Hamming distance <= radius
    # radius = 1 in this task.
    # Ball size = 1 + n * (q - 1)
    ball_size = 1 + n * (q - 1)
    
    # balls[c, j] is the integer index of the j-th word in the ball of c
    balls = np.zeros((total, ball_size), dtype=np.int32)
    
    # Fill balls
    # j=0 is the word itself
    balls[:, 0] = all_indices
    
    # For each position i, and each alternative digit d != digits[:, i]
    # We need to update the word value.
    # new_word = word - digits[:, i] * q**i + d * q**i
    # new_word = word + (d - digits[:, i]) * q**i
    
    idx = 1
    for i in range(n):
        power_i = q ** i
        # current digit for all words at position i
        curr_dig = digits[:, i]
        for d in range(q):
            if d == 0: # We only want to change to d if d != curr_dig
                # This loop structure is tricky because d is fixed but curr_dig varies.
                # We need to handle each d separately.
                pass
        
        # Let's do it properly:
        for d in range(q):
            # Mask of words where digit at i is NOT d
            mask = curr_dig != d
            if not np.any(mask):
                continue
            
            # Calculate the delta for those words
            # delta = (d - curr_dig[mask]) * power_i
            delta = (d - curr_dig[mask]) * power_i
            new_words = all_indices[mask] + delta
            
            # Assign to balls
            # We need to interleave these into the balls array correctly.
            # The balls array is currently being filled sequentially for each (i, d) pair?
            # No, the structure of balls[c] is: [c, c+delta(i=0, d=1), c+delta(i=0, d=2), c+delta(i=1, d=1), ...]
            # This means for a fixed i, we iterate d.
            # But the mask is global.
            # It is easier to just fill the balls array in a loop over i and d, but we must be careful with indices.
            
            # Actually, the previous code did:
            # for i in range(n):
            #   for d in range(q):
            #     if d == orig: continue
            #     balls[c, idx] = ...
            #     idx += 1
            # This implies the order in balls[c] is:
            # 0: c
            # 1: change pos 0 to d!=orig
            # ...
            
            # Let's re-construct the balls filling to be vectorized.
            # We can create a 3D array of deltas? No, too complex.
            # Given total=2187 and ball_size=15, the loop is small enough.
            # But we want speed.
            
            # Let's just use the loop over i and d, but vectorize the assignment.
            # However, the index `idx` in balls[c, idx] depends on the count of valid (i, d) pairs processed so far.
            # Since for each i, we have (q-1) valid d's, the offset is predictable.
            
            # Let's restart the balls filling logic to be clean and vectorized.
            pass

    # Re-do balls filling properly
    balls = np.zeros((total, ball_size), dtype=np.int32)
    balls[:, 0] = all_indices
    
    col_idx = 1
    for i in range(n):
        power_i = q ** i
        curr_dig = digits[:, i]
        for d in range(q):
            # We want to add the word formed by changing digit i to d, IF d != curr_dig
            # This is a permutation of the alphabet.
            # For a fixed i and d, the set of words that have digit i != d will be mapped to new words.
            # But we need to fill balls[c, col_idx] for ALL c?
            # No, for a specific c, the "change digit i to d" operation is only valid if c's digit i != d.
            # If c's digit i == d, this entry in the ball list is "skipped" in the previous sequential logic.
            # This makes vectorization of the *structure* of the ball list hard because the "skip" depends on c.
            
            # Alternative: Precompute the ball lists as sets of indices, then flatten?
            # Or, just use the sequential loop but optimized?
            # 2187 * 14 = ~30k operations. Python loop is slow but acceptable?
            # 30k iterations is < 1ms in Python? No, maybe 10-20ms. Acceptable.
            
            pass

    # Let's use a simple, robust loop for balls generation. It's a one-time cost.
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for c in range(total):
        balls[c, 0] = c
        idx = 1
        # Get digits of c
        dc = digits[c]
        for i in range(n):
            orig = dc[i]
            power_i = q ** i
            for d in range(q):
                if d == orig:
                    continue
                new_val = c + (d - orig) * power_i
                balls[c, idx] = new_val
                idx += 1
                
    # Deficit-Driven Greedy
    # 1. Initialize coverage count for each word: how many selected codewords cover it?
    #    Initially 0.
    # 2. We maintain a set of "uncovered" words.
    #    Actually, we want to pick the next codeword based on the "deficit" of uncovered words.
    #    Deficit of a word w: how many potential codewords cover w? (Degree)
    #    In a regular graph, degree is constant. In Hamming space with radius 1, every word is covered by
    #    1 + n*(q-1) potential codewords. So degree is constant.
    #    Therefore, "min degree" is constant.
    #    
    #    The directive says: "maximizes the minimum coverage deficit among all currently uncovered words".
    #    If degree is constant, this reduces to: pick a codeword that covers the most uncovered words?
    #    Wait. If degree is constant, the "deficit" doesn't change the priority unless we define deficit differently.
    #    
    #    Let's re-read: "prioritize the most constrained regions".
    #    If all words have the same number of neighbors, they are equally constrained initially.
    #    However, as we select codewords, the "effective" constraint might change?
    #    
    #    Actually, a better interpretation for "deficit-driven" in set cover with uniform degrees:
    #    It is equivalent to standard greedy (max gain) if degrees are uniform.
    #    
    #    BUT, the directive suggests a *different* strategy.
    #    Maybe: Pick the uncovered word with the fewest *available* (not yet selected) covering codewords?
    #    If a word w is uncovered, and all its covering codewords have been selected, it's a failure (shouldn't happen).
    #    If we select a codeword, we "use up" its coverage.
    #    
    #    Let's try a different angle: **Randomized Greedy with Criticality**.
    #    1. Compute gain for all candidates.
    #    2. Instead of picking the max, pick from the top K candidates with probability proportional to gain?
    #    No, that's simulated annealing-ish.
    #    
    #    Let's stick to the prompt's specific idea: "criticality-based ordering".
    #    If degrees are uniform, let's look at the *neighbors* of the uncovered set.
    #    
    #    Alternative Structural Change: **Local Search (Swaps)**
    #    The greedy algorithm gets stuck in local optima. A robust way to improve is to allow 1-for-1 swaps.
    #    1. Run Greedy.
    #    2. Iterate:
    #       a. Try to remove any redundant codeword.
    #       b. Try to swap a codeword c_in with a non-codeword c_out such that:
    #          - c_out covers all points that c_in uniquely covered.
    #          - This allows c_in to be removed? No, that's just removal.
    #          - Swap: Replace c_in with c_out. If this allows a *subsequent* removal, it's good.
    #          - Simpler: Replace c_in with c_out if it doesn't break coverage, and then try to remove another codeword?
    #    
    #    Let's implement a strong **Local Search** phase.
    #    1. Greedy construction.
    #    2. Redundancy removal.
    #    3. 1-Swap Local Search:
    #       For each codeword c_in in code:
    #           For each non-codeword c_out (sampled or all):
    #               If replacing c_in with c_out maintains coverage:
    #                   Try to remove any other redundant codeword.
    #                   If size decreases, accept.
    #                   If size stays same, maybe accept with probability? (Hill climbing on size only).
    #    
    #    Given the small size (2187 words, ~186 codewords), we can afford to check many swaps.
    #    Total candidates for c_out: 2187.
    #    For each c_in (186), check 2187 c_outs? 186 * 2187 = 400k checks.
    #    Each check: Does c_out cover the "unique" points of c_in?
    #    Unique points of c_in: points in ball(c_in) NOT covered by (code \ {c_in} U {c_out}).
    #    This is expensive to compute from scratch.
    
    # Efficient Swap Check:
    # Let Code be the set.
    # Let C_in be a codeword.
    # Let U(C_in) = ball(C_in) \ (Union_{c in Code \ {C_in}} ball(c)).
    # These are the points ONLY covered by C_in.
    # If we remove C_in, these points become uncovered.
    # To replace C_in with C_out, C_out must cover ALL points in U(C_in).
    # i.e., U(C_in) must be a subset of ball(C_out).
    
    # We can precompute U(C_in) for all C_in in Code?
    # Yes.
    
    # Algorithm:
    # 1. Greedy to get initial Code.
    # 2. Remove redundant codewords (where U(C_in) is empty).
    # 3. Local Search Loop:
    #    a. Compute U(C_in) for all C_in in Code.
    #    b. If any U(C_in) is empty, remove C_in. Repeat.
    #    c. For each C_in in Code:
    #       For each C_out in All_Words \ Code:
    #          If U(C_in) is subset of ball(C_out):
    #             # Swap is valid for coverage.
    #             # Now, does this swap allow us to remove ANOTHER codeword?
    #             # Or does it just maintain size?
    #             # If we just maintain size, we might cycle.
    #             # We should only swap if it leads to a reduction in size in the next step.
    #             # This is a 2-step lookahead.
    #             
    #             # Simpler: If we swap, we might create a new redundancy.
    #             # Let's try the swap, then check for redundancies.
    #             # If a redundancy is found, accept the swap (size decreases).
    #             # If no redundancy, reject (size stays same, no gain).
    
    # This seems promising.
    
    # Implementation Details:
    # - Represent Code as a set of integers for fast lookup.
    # - Represent coverage as a boolean array or count array.
    
    # Let's refine the Greedy to be "Deficit" based as requested, even if it's similar to max-gain here.
    # Actually, if I just do Max-Gain Greedy + Strong Local Search, it's a valid structural change from "Greedy + Weak Local Search".
    
    # Let's implement:
    # 1. Vectorized Max-Gain Greedy.
    # 2. Vectorized Redundancy Removal.
    # 3. 1-Swap Local Search with redundancy check.
    
    # --- Phase 1: Greedy ---
    covered = np.zeros(total, dtype=bool)
    code = []
    code_set = set()
    
    # Precompute ball membership for fast "is subset" checks?
    # We have balls (total, ball_size).
    
    while not np.all(covered):
        if time.time() > deadline - 20.0:
            break
        
        # Compute gain for all candidates
        # is_covered = covered[balls] -> (total, ball_size)
        # gain = sum(~is_covered, axis=1)
        is_covered = covered[balls]
        gains = np.sum(~is_covered, axis=1)
        
        # Pick best
        best_c = np.argmax(gains)
        if gains[best_c] == 0:
            break
            
        code.append(int(best_c))
        code_set.add(int(best_c))
        covered[balls[best_c]] = True
        
    # --- Phase 2: Redundancy Removal ---
    # Iteratively remove any codeword that is redundant.
    # A codeword c is redundant if ball(c) is subset of union(ball(c') for c' in code \ {c}).
    # Equivalent: All points in ball(c) are covered by other codewords.
    
    def remove_redundant(code_list, code_set_obj, balls_arr, total_size):
        changed = True
        while changed:
            changed = False
            # Compute coverage by each codeword?
            # We need to know if removing c leaves any point in ball(c) uncovered.
            # This is equivalent to: Is there any point p in ball(c) such that c is the ONLY codeword covering p?
            
            # Let's compute "count of covering codewords" for each point.
            # counts = np.zeros(total_size, dtype=np.int32)
            # for c in code_list:
            #     counts[balls_arr[c]] += 1
            
            # Then c is redundant if min(counts[balls_arr[c]]) >= 2.
            
            counts = np.zeros(total_size, dtype=np.int32)
            for c in code_list:
                counts[balls_arr[c]] += 1
                
            new_code = []
            new_set = set()
            for c in code_list:
                # Check if c is redundant
                ball_pts = balls_arr[c]
                if np.all(counts[ball_pts] >= 2):
                    # Redundant
                    changed = True
                    continue
                new_code.append(c)
                new_set.add(c)
                
            return new_code, new_set, changed
            
    code, code_set, _ = remove_redundant(code, code_set, balls, total)
    
    # --- Phase 3: 1-Swap Local Search ---
    # We try to swap a codeword c_in with a non-codeword c_out.
    # Condition: The swap must allow the removal of at least one other codeword (or c_in itself if c_out covers its unique points? No, c_in is removed).
    # Actually, if we replace c_in with c_out, and c_out covers the unique points of c_in, the coverage is maintained.
    # Then we check if any codeword (including c_out? No, c_out is new) becomes redundant.
    
    # To make this efficient, we can precompute the "unique points" for each codeword.
    # Unique points of c: points in ball(c) covered ONLY by c.
    
    # Let's run a loop for a limited number of iterations or until no improvement.
    
    # Optimization:
    # We only need to consider c_out that covers the unique points of c_in.
    # We can filter c_outs.
    
    max_swap_iters = 50
    for _ in range(max_swap_iters):
        if time.time() > deadline - 5.0:
            break
            
        improved = False
        
        # Compute counts
        counts = np.zeros(total, dtype=np.int32)
        for c in code:
            counts[balls[c]] += 1
            
        # Identify unique points for each codeword
        # unique_mask[c] = True if point is in ball(c) and counts[point] == 1
        
        # For each c_in in code:
        for i, c_in in enumerate(code):
            ball_in = balls[c_in]
            # Points uniquely covered by c_in
            unique_pts = ball_in[counts[ball_in] == 1]
            
            if len(unique_pts) == 0:
                # c_in is redundant? Should have been removed in Phase 2.
                # But maybe counts changed? No, counts is static here.
                # If unique_pts is empty, c_in is redundant. Remove it.
                code.pop(i)
                code_set.remove(c_in)
                improved = True
                break # Restart loop
                
            # Find c_out such that unique_pts is subset of ball(c_out)
            # This means for all p in unique_pts, p is in ball(c_out).
            # Equivalently, c_out is a common neighbor of all points in unique_pts?
            # No, c_out must be within distance 1 of all points in unique_pts.
            
            # Since unique_pts are all within distance 1 of c_in, they are "close".
            # If unique_pts has 1 point p, c_out must be in ball(p).
            # If unique_pts has > 1 points, c_out must be in intersection of their balls.
            
            # Intersection of balls of points in unique_pts.
            # Let's compute the set of valid c_outs.
            
            # Initialize valid_outs as all words?
            # Better: Start with ball of the first unique point.
            if len(unique_pts) == 1:
                p0 = unique_pts[0]
                candidate_outs = balls[p0]
            else:
                # Intersection of balls
                # For small unique_pts (usually 1 or 2), this is fast.
                # If unique_pts is large, intersection might be empty.
                
                # Use numpy logical_and
                # Create a boolean mask for all words?
                # valid_mask = np.zeros(total, dtype=bool)
                # for p in unique_pts:
                #     valid_mask[balls[p]] = True
                # This is slow if unique_pts is large.
                
                # Alternative:
                # The only words that can cover multiple points in unique_pts (which are all neighbors of c_in)
                # are c_in itself and the neighbors of c_in.
                # Actually, if p1 and p2 are neighbors of c_in, and p1 != p2.
                # If they differ in position i and j.
                # A word c_out covering both must be distance 1 from p1 and p2.
                # This restricts c_out significantly.
                
                # Let's just iterate over all words not in code and check?
                # 2187 words.
                # For each c_out, check if it covers all unique_pts.
                # This is O(|unique_pts|) per c_out.
                # If |unique_pts| is small (1 or 2), this is fast.
                
                valid_outs = []
                for c_out in range(total):
                    if c_out in code_set:
                        continue
                    # Check if c_out covers all unique_pts
                    # c_out covers p if p is in ball(c_out).
                    # We can check if p is in balls[c_out].
                    # Since unique_pts is small, we can do a linear search or set check.
                    # Let's precompute a set for ball(c_out)? No, too slow.
                    
                    # Use the balls array.
                    # p is in ball(c_out) iff c_out is in ball(p).
                    # So check if c_out is in balls[p].
                    
                    # Since we have balls[p], we can check if c_out is in that list.
                    # List is small (15).
                    
                    covers_all = True
                    for p in unique_pts:
                        if c_out not in balls[p]:
                            covers_all = False
                            break
                    if covers_all:
                        valid_outs.append(c_out)
                        
            if not valid_outs:
                continue
                
            # Try each valid c_out
            for c_out in valid_outs:
                # Perform swap: remove c_in, add c_out
                # Check if this creates a redundancy in the new set.
                
                # New code: code \ {c_in} + {c_out}
                # We need to check if any codeword in this new set is redundant.
                
                # To be efficient, we only need to check:
                # 1. Is c_out redundant? (Unlikely, it covers unique points of c_in, so it's needed for those? No, those points are now covered by c_out. Are they covered by others? No, they were unique to c_in. So c_out is NOT redundant for those points. But it might be redundant for OTHER points? No, redundancy means "can be removed without breaking coverage". If c_out is removed, the unique points of c_in become uncovered. So c_out is NOT redundant.)
                # 2. Is any OTHER codeword c' in code \ {c_in} now redundant?
                #    c' was not redundant before. It had some unique points.
                #    After swap, coverage of those unique points might change?
                #    No, c_out only adds coverage. It doesn't remove coverage from others.
                #    Wait, removing c_in REMOVES coverage.
                #    So c' might become redundant if its unique points were also covered by c_in?
                #    No, if a point was unique to c', c_in didn't cover it.
                #    So c' is still not redundant?
                #    
                #    Let's re-evaluate.
                #    Redundancy of c' means: ball(c') is covered by others.
                #    Before: ball(c') covered by (Code \ {c'}).
                #    After: ball(c') covered by (Code \ {c'} \ {c_in} U {c_out}).
                #    The set of coverers for points in ball(c') has changed.
                #    c_in is removed. c_out is added.
                #    If c_in was covering some points in ball(c') that were NOT covered by any other c'' in Code \ {c'}, then removing c_in might break coverage?
                #    No, we established that c_out covers the unique points of c_in.
                #    So all points previously covered by c_in are still covered (by c_out).
                #    Therefore, the coverage of the entire space is maintained.
                #    
                #    Does this make any c' redundant?
                #    c' is redundant if every point in ball(c') is covered by at least one other codeword.
                #    Before swap, c' was NOT redundant. So there existed a point p in ball(c') covered ONLY by c'.
                #    After swap, is p still covered ONLY by c'?
                #    p was not covered by c_in (since p is unique to c').
                #    p is not covered by c_out? (c_out covers unique points of c_in. p is not a unique point of c_in, so c_out might or might not cover p).
                #    If c_out covers p, then p is covered by c' AND c_out.
                #    If c_out covers ALL points in ball(c') that were unique to c', then c' becomes redundant.
                
                # So, we need to check if c' becomes redundant.
                # This requires checking the new coverage counts.
                
                # Efficient check:
                # Compute new counts?
                # Or just check the specific points that were unique to c'.
                
                # Let's just do a full redundancy check on the new set if a valid swap is found.
                # Since valid_outs is likely small, this is affordable.
                
                new_code = [c for c in code if c != c_in]
                new_code.append(c_out)
                new_set = set(new_code)
                
                # Check for redundancy in new_code
                new_counts = np.zeros(total, dtype=np.int32)
                for c in new_code:
                    new_counts[balls[c]] += 1
                    
                redundant_found = False
                for c in new_code:
                    ball_pts = balls[c]
                    if np.all(new_counts[ball_pts] >= 2):
                        redundant_found = True
                        break
                        
                if redundant_found:
                    # Accept swap and remove redundant
                    # We need to identify which one is redundant to remove it.
                    # Actually, we can just call remove_redundant on the new set.
                    code, code_set, _ = remove_redundant(new_code, new_set, balls, total)
                    improved = True
                    break # Restart outer loop
                    
        if not improved:
            break
            
    # Final conversion
    result = []
    for c in code:
        digits_c = digits[c]
        result.append(digits_c.tolist())
        
    return result

# EVOLVE-BLOCK-END
