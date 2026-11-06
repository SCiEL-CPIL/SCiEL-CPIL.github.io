import random
import numpy as np

def construct():
    N_range = 129 # Sums from 128 to 256 inclusive
    S_min = 128
    S_max = 256
    U = 128 # Universe [0, 128]
    
    def get_multiplicities(A_set):
        """
        Compute multiplicities of sums in [S_min, S_max] for unordered pairs from A_set.
        Returns an array of length 129 where index i corresponds to sum S_min + i.
        """
        if not A_set:
            return np.zeros(N_range, dtype=np.int32)
        A_arr = np.array(list(A_set), dtype=np.int32)
        # Outer sum
        sums = A_arr[:, None] + A_arr[None, :]
        # We only care about sums in [S_min, S_max]
        # Filter sums
        mask = (sums >= S_min) & (sums <= S_max)
        valid_sums = sums[mask]
        # Bincount the valid sums
        # np.bincount returns an array of length max+1
        # We want to map sum S to index S - S_min
        if len(valid_sums) == 0:
            return np.zeros(N_range, dtype=np.int32)
        
        counts_full = np.bincount(valid_sums)
        # Extract the relevant range
        # counts_full[S_min + i] gives the count for sum S_min + i
        # Note: np.bincount handles the offset naturally if we just slice
        # However, bincount index is the value itself.
        # So counts_full[S_min] is count for S_min.
        # We need indices S_min to S_max.
        result = np.zeros(N_range, dtype=np.int32)
        if S_max < len(counts_full):
            result[:] = counts_full[S_min:S_max+1]
        else:
            # If S_max is larger than max sum possible (which is 256), it's fine
            # But if S_max >= len(counts_full), we just take what's there
            max_idx = min(S_max + 1, len(counts_full))
            result[:max_idx - S_min] = counts_full[S_min:max_idx]
            
        return result

    def is_covered(counts):
        return np.all(counts > 0)

    def objective(counts):
        """
        Objective: Minimize peak multiplicity.
        We use a combination of max and sum of squares to guide the search.
        """
        if not is_covered(counts):
            return float('inf')
        return np.max(counts) + 0.01 * np.sum(counts**2)

    def local_search_refine(initial_A, steps=10000):
        """
        Fast local search using swaps to refine a covered set.
        """
        A = set(initial_A)
        counts = get_multiplicities(A)
        if not is_covered(counts):
            return None, float('inf')
        
        current_obj = objective(counts)
        best_A = set(A)
        best_obj = current_obj
        best_max = np.max(counts)
        
        A_list = list(A)
        
        for step in range(steps):
            if not A_list:
                break
            a_idx = random.randint(0, len(A_list) - 1)
            a = A_list[a_idx]
            
            # Choose a random replacement
            b = random.randint(0, U)
            if b in A:
                continue
                
            # Calculate delta
            # Remove a, add b
            # New A is A \ {a} U {b}
            
            # Efficient delta calculation:
            # Old sums involving a: 2a, a+x for x in A\{a}
            # New sums involving b: 2b, b+x for x in A\{a}
            
            # We update counts in place temporarily? 
            # Or just compute new counts for affected sums.
            
            affected_sums = set()
            s_aa = 2 * a
            if S_min <= s_aa <= S_max: affected_sums.add(s_aa)
            s_bb = 2 * b
            if S_min <= s_bb <= S_max: affected_sums.add(s_bb)
            
            for x in A:
                if x == a: continue
                s_ax = a + x
                if S_min <= s_ax <= S_max: affected_sums.add(s_ax)
                s_bx = b + x
                if S_min <= s_bx <= S_max: affected_sums.add(s_bx)
                
            if not affected_sums:
                continue
                
            valid = True
            new_max = best_max # Assume max doesn't drop below best_max initially? No, it can drop.
            # Let's just compute new counts for affected sums and check coverage/max.
            
            # Store old values
            old_vals = {}
            new_vals = {}
            
            for s in affected_sums:
                idx = s - S_min
                old_c = counts[idx]
                old_vals[idx] = old_c
                
                change = 0
                if s == 2 * a: change -= 1
                # x in A, x != a
                for x in A:
                    if x == a: continue
                    if s == a + x: change -= 1
                    if s == b + x: change += 1
                if s == 2 * b: change += 1
                
                new_c = old_c + change
                if new_c < 0:
                    valid = False
                    break
                if new_c == 0:
                    # Coverage lost?
                    # If old_c was > 0 and new_c is 0, we lose coverage for this sum.
                    # But wait, if old_c was 1, and we remove the only representation, it's 0.
                    # This is invalid.
                    valid = False
                    break
                new_vals[idx] = new_c
                
            if not valid:
                continue
                
            # Check if new state is better
            # We need to know the new max.
            # The new max is max( max(new_vals.values()), max(counts for unaffected sums) )
            # We don't have max of unaffected sums readily.
            # However, if max(new_vals) < best_max, it's a good candidate.
            # If max(new_vals) == best_max, we check sum of squares.
            
            max_affected_new = max(new_vals.values()) if new_vals else 0
            
            # Heuristic: Accept if max_affected_new < current_max (which is best_max if we track it)
            # Or if sum of squares improves.
            
            # Let's compute sum of squares delta
            delta_sq = 0
            for idx, new_c in new_vals.items():
                old_c = old_vals[idx]
                delta_sq += new_c*new_c - old_c*old_c
            
            # We need current_max to compare.
            # Let's maintain current_max.
            
            # If max_affected_new < current_max, we might have reduced the global max.
            # If max_affected_new == current_max, we need to check if the *only* sums with current_max were affected and reduced?
            # This is complex. Let's just accept if delta_sq < 0 or max_affected_new < current_max.
            
            # To be safe, let's just compute the full objective if it looks promising.
            # But full objective is O(K^2).
            
            # Simple acceptance:
            if max_affected_new < current_max:
                accept = True
            elif max_affected_new == current_max and delta_sq < 0:
                accept = True
            else:
                accept = False
                
            if accept:
                # Apply changes
                for idx, new_c in new_vals.items():
                    counts[idx] = new_c
                A.remove(a)
                A.add(b)
                A_list[a_idx] = b # Update list, order doesn't matter for set logic
                
                # Update current_max
                # If the previous max was in affected_sums and reduced, current_max might drop.
                # If max_affected_new < current_max, current_max = max(max_affected_new, max_unaffected).
                # We don't know max_unaffected.
                # Let's just recompute max if max_affected_new < current_max.
                if max_affected_new < current_max:
                    current_max = np.max(counts)
                    
                current_obj = objective(counts)
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_max = current_max
                    best_A = set(A)
                    
        return best_A, best_max

    # ADMM-inspired Relaxation
    # We maintain a probability vector p of length 129.
    # p[i] is the probability of including i in A.
    # We update p based on the gradient of the objective.
    
    # Initialize p
    # Start with a dense set to ensure coverage
    # A good initial set is all even numbers or a random dense set.
    # Let's start with a random set of size 80 to ensure coverage.
    
    best_global_max = float('inf')
    best_global_A = None
    
    for trial in range(5):
        # Random initialization
        K_init = random.randint(70, 90)
        A_init = set(random.sample(range(U + 1), K_init))
        
        # Ensure coverage
        counts = get_multiplicities(A_init)
        if not is_covered(counts):
            # Add elements greedily to cover missing sums
            missing = [s for s in range(S_min, S_max+1) if counts[s-S_min] == 0]
            while missing:
                best_elem = -1
                best_cover = 0
                for e in range(U+1):
                    if e in A_init: continue
                    cover = 0
                    for ms in missing:
                        if (ms - e) in A_init or ms == 2*e:
                            cover += 1
                    if cover > best_cover:
                        best_cover = cover
                        best_elem = e
                if best_elem == -1 or best_cover == 0:
                    break
                A_init.add(best_elem)
                counts = get_multiplicities(A_init)
                missing = [s for s in range(S_min, S_max+1) if counts[s-S_min] == 0]
                
        if not is_covered(get_multiplicities(A_init)):
            continue
            
        # ADMM Iteration
        p = np.zeros(U + 1)
        for e in A_init:
            p[e] = 1.0
        for e in range(U+1):
            if e not in A_init:
                p[e] = 0.1 # Small probability for exploration
                
        current_A = set([i for i in range(U+1) if p[i] > 0.5])
        # Ensure coverage in current_A
        counts = get_multiplicities(current_A)
        if not is_covered(counts):
             # Fallback to A_init
             current_A = set(A_init)
             counts = get_multiplicities(current_A)

        for admm_step in range(200):
            counts = get_multiplicities(current_A)
            if not is_covered(counts):
                # Repair: add element that covers most missing sums
                missing = [s for s in range(S_min, S_max+1) if counts[s-S_min] == 0]
                best_elem = -1
                best_cover = 0
                for e in range(U+1):
                    if e in current_A: continue
                    cover = 0
                    for ms in missing:
                        if (ms - e) in current_A or ms == 2*e:
                            cover += 1
                    if cover > best_cover:
                        best_cover = cover
                        best_elem = e
                if best_elem != -1:
                    current_A.add(best_elem)
                    p[best_elem] = 1.0
                continue

            # Compute gradient for each element e
            # Contribution of e to sum s:
            # If e is in A, it contributes to 2e and e+x for x in A.
            # Removing e decreases count of these sums by 1.
            # Adding e (if not in A) increases count of these sums by 1.
            
            # We want to minimize peak multiplicity.
            # Elements that are part of "high count" sums should be removed.
            # Elements that would add to "low count" sums should be added.
            
            # For each e, compute a "score"
            scores = np.zeros(U + 1)
            
            for e in range(U + 1):
                # Sums involving e
                # 2e
                # e + x for x in current_A
                
                # If e is in current_A:
                # Removing it reduces counts.
                # Benefit: reduces high counts.
                # Cost: might lose coverage.
                
                # If e is not in current_A:
                # Adding it increases counts.
                # Benefit: covers missing sums (if any, but we ensured coverage).
                # Cost: increases counts, potentially creating new peaks.
                
                # Let's define a penalty for e based on the sums it participates in.
                # Penalty = sum of (count[s] - target) for s involving e.
                # We want to minimize the max. A proxy is sum of squares.
                # d/dp_e (sum s^2) = 2 * sum_{s involving e} (2*count[s] - 1) ... roughly.
                
                # Simpler:
                # If e in A:
                #   score = - sum_{s in S_e} count[s]  (we want to remove e if it's in high count sums)
                # If e not in A:
                #   score = sum_{s in S_e} (1 - count[s]) ... no.
                
                # Let's use the change in sum of squares if we toggle e.
                # Delta SQ for removing e:
                #   For each s in S_e (sums involving e), count[s] -> count[s] - 1
                #   Delta = sum ( (c-1)^2 - c^2 ) = sum ( -2c + 1 ) = -2*sum(c) + |S_e|
                # Delta SQ for adding e:
                #   For each s in S_e, count[s] -> count[s] + 1
                #   Delta = sum ( (c+1)^2 - c^2 ) = sum ( 2c + 1 ) = 2*sum(c) + |S_e|
                
                # We want to minimize SQ.
                # If e in A: if Delta_remove < 0 (i.e., SQ decreases), we want to keep p[e] high? 
                # Wait, if removing e decreases SQ, we should REMOVE e. So p[e] should go down.
                # If e not in A: if Delta_add < 0 (impossible, 2c+1 > 0), we should ADD e.
                # Actually, adding e always increases SQ. So we only add e if it's necessary for coverage or if it balances the distribution better?
                # No, adding e increases the number of representations, which generally increases SQ.
                # So the ADMM step should mostly REMOVE elements that are causing high multiplicity, 
                # and only ADD elements if coverage is at risk.
                
                # Strategy:
                # 1. Identify "hot" sums (count > avg).
                # 2. For each e in A, if it contributes to hot sums, decrease p[e].
                # 3. For each e not in A, if it contributes to "cold" sums (count == 1) and removing any current element from those cold sums would break coverage, maybe increase p[e]?
                
                # Let's just compute the "cost" of e.
                # cost[e] = sum_{s in S_e} count[s]
                # If e in A, high cost is bad.
                # If e not in A, high cost is also bad (adding it would make it worse).
                
                # So we want to minimize cost[e] for all e?
                # No, we want the SET A to have low total cost.
                
                # Update rule:
                # p[e] = sigmoid( -lambda * cost[e] + bias )
                
                # Let's compute cost for all e.
                # cost[e] = count[2e] (if in range) + sum_{x in A, x!=e} count[e+x] (if in range)
                # Note: count[e+x] is the count BEFORE adding/removing e.
                
                # This is O(N * K). N=129, K~80. 10k ops. Fast.
                
                # Optimization: Precompute? 
                # For each e, S_e = {2e} U {e+x | x in A}
                
                for x in current_A:
                    # e + x
                    s = e + x
                    if S_min <= s <= S_max:
                        scores[e] += counts[s - S_min]
                        # Also x + e is same sum, but we are iterating e.
                        # We need to be careful not to double count in the set definition?
                        # No, cost[e] is just the sum of counts of sums involving e.
                        # Each sum s involving e is counted once for e.
                        # Wait, if s = e + x, it involves e and x.
                        # When we compute cost[e], we add count[s].
                        # When we compute cost[x], we add count[s].
                        # This is correct. The "pressure" on e is proportional to the load of the sums it touches.
                
                s_2e = 2 * e
                if S_min <= s_2e <= S_max:
                    scores[e] += counts[s_2e - S_min]
                    
            # Normalize scores
            # We want to remove high score elements from A.
            # And add low score elements to A?
            
            # Update p
            # p[e] = 1.0 if e in A and scores[e] < threshold
            # p[e] = 0.0 if e not in A and scores[e] > threshold
            # This is a hard step.
            
            # Let's do a soft update:
            # p[e] = p[e] * exp(-eta * scores[e]) for e in A
            # p[e] = p[e] * exp(-eta * scores[e]) for e not in A
            # Then renormalize or threshold.
            
            # Simpler:
            # Sort elements by score.
            # Keep the K elements with lowest scores?
            # But we must maintain coverage.
            
            # Let's try:
            # 1. Compute scores.
            # 2. Candidate A_new = A \ {high score elements} U {low score elements not in A}
            # 3. Check coverage.
            
            # Let's just do a local search step on the current A, guided by scores.
            # Pick an element a in A with high score.
            # Pick an element b not in A with low score.
            # Try swap a -> b.
            
            # This is effectively local search.
            # To make it ADMM-like, we can use the scores to guide the choice.
            
            # For now, let's just run local_search_refine on the current A.
            # The ADMM part is the initialization and the global view of scores.
            
            A_res, max_res = local_search_refine(current_A, steps=500)
            if A_res is not None and max_res < best_global_max:
                best_global_max = max_res
                best_global_A = A_res
                current_A = A_res
                
            if best_global_max <= 12:
                break

    if best_global_A is None:
        # Fallback
        return list(range(64, 129))
        
    return list(best_global_A)
