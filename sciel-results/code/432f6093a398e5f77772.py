import random
import itertools
import time

def construct():
    start_time = time.time()
    timeout = 115.0  # Safety margin for 120s limit
    
    n = 16
    points = list(range(n))
    
    # Precompute all possible triples and their bitmasks
    all_triples = []
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                mask = (1 << a) | (1 << b) | (1 << c)
                p_ab = (a, b)
                p_ac = (a, c)
                p_bc = (b, c)
                all_triples.append((mask, p_ab, p_ac, p_bc, (a,b,c)))
                
    # Index triples by their pairs for quick lookup? 
    # Or just maintain used_pairs set in the packing.
    
    best_alpha = n + 1
    best_triples = []
    
    # Exact calculation of alpha
    def calc_alpha_exact(triple_masks):
        max_ind = 0
        # Iterate all subsets
        # Optimization: If we find an independent set of size k, 
        # we know max_ind >= k.
        # We want the max.
        # We can iterate sizes from n down to 1 and return the first one found?
        # No, we need the max size that IS independent.
        # So we check size n, n-1, ... until we find one that is NOT independent?
        # No, that's for finding the max independent set in a graph.
        # Here, "independent" means "contains no triple".
        # If S is independent, all subsets of S are independent.
        # If S is NOT independent, some superset might be independent? No.
        # If S contains a triple, S is NOT independent.
        # If S does NOT contain a triple, S IS independent.
        # So the set of independent sets is a "down-set" (hereditary).
        # Therefore, if a set of size k is independent, all smaller sets are independent.
        # If a set of size k is NOT independent, it might have independent subsets.
        # To find the MAX size independent set:
        # We can check sizes from n down to 1.
        # The first size k for which we find AT LEAST ONE independent set is the answer?
        # No. If there is an independent set of size 10, there are many.
        # If there is NO independent set of size 11, then max is <= 10.
        # So:
        # Check if there exists an independent set of size k.
        # Start k = n.
        # If yes, alpha >= k. Try k+1? No, we start high.
        # Start k = n. Check if ALL sets of size k are independent? No.
        # We want to know if THERE EXISTS an independent set of size k.
        # If yes, alpha >= k.
        # We want the MAX k such that there exists an independent set.
        # So we can binary search? No, the property "exists independent set of size k" is monotonic decreasing with k?
        # If there is an independent set of size k, there is one of size k-1.
        # So if P(k) is true, P(k-1) is true.
        # So P(k) is True for k <= alpha, False for k > alpha.
        # So we can binary search for the largest k where P(k) is True.
        
        def exists_independent_set_of_size(k):
            if k == 0: return True
            if k > n: return False
            # Iterate combinations of size k
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for p in comb:
                    mask |= (1 << p)
                # Check if independent
                is_ind = True
                for t in triple_masks:
                    if (t & mask) == t:
                        is_ind = False
                        break
                if is_ind:
                    return True
            return False

        # Binary search
        lo, hi = 0, n
        ans = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if exists_independent_set_of_size(mid):
                ans = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return ans

    # Fast check: Does an independent set of size k exist?
    # Used in local search to quickly reject moves that don't improve alpha.
    def has_independent_set_of_size(triple_masks, k):
        if k == 0: return True
        if k > n: return False
        # Optimization: Pre-compute list of combinations for common k?
        # Or just generate on the fly.
        for comb in itertools.combinations(range(n), k):
            mask = 0
            for p in comb:
                mask |= (1 << p)
            is_ind = True
            for t in triple_masks:
                if (t & mask) == t:
                    is_ind = False
                    break
            if is_ind:
                return True
        return False

    def generate_packing():
        random.shuffle(all_triples)
        used_pairs = set()
        triples = []
        masks = []
        
        for mask, p1, p2, p3, pts in all_triples:
            if p1 in used_pairs or p2 in used_pairs or p3 in used_pairs:
                continue
            triples.append(pts)
            masks.append(mask)
            used_pairs.add(p1)
            used_pairs.add(p2)
            used_pairs.add(p3)
            
        return triples, masks, used_pairs

    def local_search(triples, masks, used_pairs, current_alpha):
        # current_alpha is the known alpha of this packing.
        # We try to reduce it.
        # Move: Swap a triple.
        # Evaluation: Check if new packing has independent set of size current_alpha.
        # If NO, then new_alpha < current_alpha.
        
        # To speed up, we pre-compute the set of "bad" pairs? 
        # When we remove a triple, we free up its 3 pairs.
        # We can add any triple that uses only free pairs (plus the freed ones).
        
        # We will try a few random swaps.
        # Time budget for this local search call: ~0.1s?
        ls_start = time.time()
        ls_timeout = 0.5 # 0.5 seconds per local search attempt
        
        # Store current state
        cur_triples = triples[:]
        cur_masks = masks[:]
        cur_used_pairs = set(used_pairs)
        
        # Precompute all triples not in current packing?
        # Or just iterate all_triples and check if valid.
        
        # To make it faster, we can just try to swap one triple at a time.
        # Pick a random index i.
        # Remove triple i.
        # Try to find a replacement triple j (j != i) that is valid.
        # Prefer triples that intersect the "critical" independent set?
        # We don't know the critical independent set easily without finding it.
        # But we know that IF an independent set of size current_alpha exists, we fail.
        # So we just check the condition.
        
        for _ in range(500): # Max 500 swaps
            if time.time() - ls_start > ls_timeout:
                break
            
            if not cur_triples:
                break
                
            # Pick a random triple to remove
            i = random.randint(0, len(cur_triples) - 1)
            t_old_mask = cur_masks[i]
            t_old_pts = cur_triples[i]
            # Extract pairs of t_old
            a, b, c = t_old_pts
            pairs_to_free = [(a,b), (a,c), (b,c)]
            
            # Remove from state
            cur_triples.pop(i)
            cur_masks.pop(i)
            for p in pairs_to_free:
                cur_used_pairs.discard(p)
            
            # Try to add a new triple
            # We want a triple that is valid (no pair in cur_used_pairs)
            # And ideally, it "breaks" the independent sets of size current_alpha.
            # Since we don't know which ones exist, we just pick a random valid one.
            
            # To be faster, we can pre-filter all_triples?
            # No, just shuffle a subset or iterate.
            # Let's pick a random candidate from all_triples.
            # If it's invalid, try another.
            
            found = False
            # Try up to 20 random candidates
            for _ in range(20):
                idx = random.randint(0, len(all_triples) - 1)
                cand_mask, p1, p2, p3, cand_pts = all_triples[idx]
                
                # Check validity
                if p1 in cur_used_pairs or p2 in cur_used_pairs or p3 in cur_used_pairs:
                    continue
                
                # It's valid. Let's see if it improves alpha.
                # Add it to a temp state
                temp_masks = cur_masks + [cand_mask]
                
                # Check if independent set of size current_alpha exists
                # If it DOES NOT, then new_alpha < current_alpha.
                if not has_independent_set_of_size(temp_masks, current_alpha):
                    # Accept move
                    cur_triples.append(cand_pts)
                    cur_masks.append(cand_mask)
                    cur_used_pairs.add(p1)
                    cur_used_pairs.add(p2)
                    cur_used_pairs.add(p3)
                    
                    # Calculate new exact alpha (it is < current_alpha)
                    new_alpha = calc_alpha_exact(cur_masks)
                    return cur_triples, cur_masks, cur_used_pairs, new_alpha
                    
                # If it DOES, reject this candidate (and the removal for now? 
                # Or keep removal and try another candidate?)
                # If we keep the removal, we have a packing with one fewer triple.
                # This might have a larger alpha (worse).
                # So we should probably revert the removal if we don't find a good replacement.
                # But finding a good replacement is the goal.
                # Let's just revert the removal if we fail to find a candidate in this step.
                # Actually, let's just break and revert.
                pass
            
            # Revert removal
            cur_triples.insert(i, t_old_pts)
            cur_masks.insert(i, t_old_mask)
            cur_used_pairs.add(pairs_to_free[0])
            cur_used_pairs.add(pairs_to_free[1])
            cur_used_pairs.add(pairs_to_free[2])
            
        return cur_triples, cur_masks, cur_used_pairs, current_alpha

    # Main loop
    attempts = 0
    while time.time() - start_time < timeout:
        attempts += 1
        triples, masks, used_pairs = generate_packing()
        
        # If packing is too small, skip
        if len(triples) < 35:
            continue
            
        alpha = calc_alpha_exact(masks)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
            
            # If we get alpha <= 5, it's likely optimal or very good.
            if best_alpha <= 5:
                break
            
            # Run local search on this good candidate
            if time.time() - start_time < timeout - 10: # Leave time for final checks
                triples, masks, used_pairs, alpha = local_search(triples, masks, used_pairs, alpha)
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_triples = triples[:]
                    if best_alpha <= 5:
                        break

    if not best_triples:
        return [[3 * i, 3 * i + 1, 3 * i + 2] for i in range(5)]
        
    return best_triples
