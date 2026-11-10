import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 12
    m = 40
    total_masks = 1 << n  # 4096
    all_masks = list(range(total_masks))
    
    # Helper to convert mask to list
    def mask_to_list(mask):
        return [i for i in range(n) if (mask >> i) & 1]
        
    def is_comparable(a, b):
        if a == b:
            return False
        return (a & b) == a or (a & b) == b

    def count_comparable(family):
        count = 0
        for i in range(len(family)):
            mi = family[i]
            for j in range(i + 1, len(family)):
                mj = family[j]
                if is_comparable(mi, mj):
                    count += 1
        return count

    def score_candidate(cand, family):
        """Returns the number of sets in family comparable to cand."""
        s = 0
        for fs in family:
            if fs != cand and is_comparable(cand, fs):
                s += 1
        return s

    # --- Phase 1: Initialization ---
    # Start with a chain to ensure high initial connectivity
    # Chain: 0, 1, 3, 7, ..., 4095
    chain = []
    mask = 0
    for i in range(n + 1):
        chain.append(mask)
        if i < n:
            mask |= (1 << i)
            
    current_family = chain[:]
    used = set(chain)
    
    # Greedily add remaining sets
    # We want to add sets that maximize the number of comparable pairs with existing sets
    # To speed up, we can pre-sort or just iterate
    remaining = [m for m in all_masks if m not in used]
    
    # Heuristic: Add sets that are "close" to the chain in terms of comparability
    # A set with size k is comparable with sets of size < k (if subset) or > k (if superset)
    # Greedy: pick the candidate with highest score_candidate
    # This is O(remaining * current_size) per addition. 
    # 4096 * 40 * 27 additions is ~4.4M ops. Fast enough.
    
    for _ in range(m - len(current_family)):
        if time.time() > deadline - 5:
            break
        best_cand = -1
        best_score = -1
        # Sample a subset of remaining to speed up if needed, but full scan is likely fast enough
        # Let's just do full scan
        for cand in remaining:
            s = score_candidate(cand, current_family)
            if s > best_score:
                best_score = s
                best_cand = cand
        
        if best_cand != -1:
            current_family.append(best_cand)
            used.add(best_cand)
            remaining.remove(best_cand)
        else:
            # Should not happen unless remaining is empty
            break

    if len(current_family) < m:
        # Fallback: add random
        for cand in all_masks:
            if cand not in used:
                current_family.append(cand)
                used.add(cand)
            if len(current_family) == m:
                break

    current_score = count_comparable(current_family)
    
    # --- Phase 2: Simulated Annealing with 1-swap and 2-swap ---
    
    # Precompute a set of unused masks for quick lookup? 
    # We will maintain a list of unused masks.
    unused_list = [m for m in all_masks if m not in used]
    
    def try_1_swap(family, unused_list, current_score, temperature, time_limit):
        """
        Performs 1-swap moves.
        Returns updated family, unused_list, score, and whether an improvement was found.
        """
        improved = False
        end_time = time.time() + time_limit
        
        # Number of attempts per temperature step
        # We will run until time_limit is reached or no improvements for a while
        # But SA accepts worse moves, so we just run for time.
        
        while time.time() < end_time:
            # Pick a random index to remove
            i = random.randint(0, len(family) - 1)
            old_mask = family[i]
            
            # Pick a random unused mask to add
            if not unused_list:
                break
            j = random.randint(0, len(unused_list) - 1)
            new_mask = unused_list[j]
            
            # Calculate delta score
            # Old contribution: old_mask comparable with all other masks in family
            # New contribution: new_mask comparable with all other masks in family (excluding old_mask, but old_mask is removed)
            
            # Let's calculate the score of the family with old_mask removed and new_mask added
            # Delta = (comparables(new_mask, family \ {old_mask}) + (0 if new==old else 0)) - (comparables(old_mask, family \ {old_mask}))
            
            # Comparables of old_mask with others:
            old_comp = 0
            for k, mk in enumerate(family):
                if k == i:
                    continue
                if is_comparable(old_mask, mk):
                    old_comp += 1
                    
            # Comparables of new_mask with others (in family \ {old_mask}):
            new_comp = 0
            for k, mk in enumerate(family):
                if k == i:
                    continue
                if is_comparable(new_mask, mk):
                    new_comp += 1
            
            delta = new_comp - old_comp
            
            # Acceptance
            if delta > 0 or random.random() < math.exp(delta / temperature):
                # Perform swap
                family[i] = new_mask
                # Update used/unused
                unused_list[j] = old_mask
                # Note: We don't update the set 'used' here, we rely on the list
                current_score += delta
                improved = True
                
        return family, unused_list, current_score, improved

    def try_2_swap(family, unused_list, current_score, temperature, time_limit):
        """
        Performs 2-swap moves.
        """
        improved = False
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Pick two distinct indices to remove
            i1 = random.randint(0, len(family) - 1)
            i2 = random.randint(0, len(family) - 1)
            if i1 == i2:
                continue
            
            old_mask1 = family[i1]
            old_mask2 = family[i2]
            
            # Pick two distinct unused masks to add
            if len(unused_list) < 2:
                break
            j1 = random.randint(0, len(unused_list) - 1)
            j2 = random.randint(0, len(unused_list) - 1)
            if j1 == j2:
                continue
            
            new_mask1 = unused_list[j1]
            new_mask2 = unused_list[j2]
            
            # Ensure new masks are distinct from each other and old masks
            # They are from unused_list, so distinct from family.
            # new_mask1 != new_mask2 guaranteed by j1 != j2
            
            # Calculate delta score
            # Remove old1, old2. Add new1, new2.
            # Old score contribution involving old1/old2:
            # 1. old1 comp old2
            # 2. old1 comp others
            # 3. old2 comp others
            
            old_comp_12 = 1 if is_comparable(old_mask1, old_mask2) else 0
            
            # Comparables of old1 with family \ {old1, old2}
            old_comp_1_others = 0
            for k, mk in enumerate(family):
                if k == i1 or k == i2:
                    continue
                if is_comparable(old_mask1, mk):
                    old_comp_1_others += 1
                    
            # Comparables of old2 with family \ {old1, old2}
            old_comp_2_others = 0
            for k, mk in enumerate(family):
                if k == i1 or k == i2:
                    continue
                if is_comparable(old_mask2, mk):
                    old_comp_2_others += 1
            
            total_old = old_comp_12 + old_comp_1_others + old_comp_2_others
            
            # New score contribution involving new1/new2:
            # 1. new1 comp new2
            # 2. new1 comp others
            # 3. new2 comp others
            
            new_comp_12 = 1 if is_comparable(new_mask1, new_mask2) else 0
            
            # Comparables of new1 with family \ {old1, old2}
            new_comp_1_others = 0
            for k, mk in enumerate(family):
                if k == i1 or k == i2:
                    continue
                if is_comparable(new_mask1, mk):
                    new_comp_1_others += 1
                    
            # Comparables of new2 with family \ {old1, old2}
            new_comp_2_others = 0
            for k, mk in enumerate(family):
                if k == i1 or k == i2:
                    continue
                if is_comparable(new_mask2, mk):
                    new_comp_2_others += 1
            
            total_new = new_comp_12 + new_comp_1_others + new_comp_2_others
            
            delta = total_new - total_old
            
            # Acceptance
            if delta > 0 or random.random() < math.exp(delta / temperature):
                # Perform swap
                family[i1] = new_mask1
                family[i2] = new_mask2
                unused_list[j1] = old_mask1
                unused_list[j2] = old_mask2
                current_score += delta
                improved = True
                
        return family, unused_list, current_score, improved

    # SA Parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.95
    
    # We will alternate between 1-swap and 2-swap phases
    # Or just mix them.
    
    best_family = current_family[:]
    best_score = current_score
    
    while time.time() < deadline - 2:
        # Perform a batch of moves
        # 100 1-swaps
        if time.time() < deadline - 2:
            current_family, unused_list, current_score, _ = try_1_swap(
                current_family, unused_list, current_score, T, 0.05
            )
        
        # 10 2-swaps (more expensive)
        if time.time() < deadline - 2:
            current_family, unused_list, current_score, _ = try_2_swap(
                current_family, unused_list, current_score, T, 0.02
            )
            
        if current_score > best_score:
            best_score = current_score
            best_family = current_family[:]
            
        T *= cooling_rate
        if T < T_min:
            T = T_min

    # Final Polish: Deterministic Hill Climbing with 1-swap and 2-swap
    # Try to find any improving move
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        
        # 1-swap polish
        for i in range(len(best_family)):
            if time.time() > deadline - 1:
                break
            old_mask = best_family[i]
            old_comp = 0
            for k, mk in enumerate(best_family):
                if k == i:
                    continue
                if is_comparable(old_mask, mk):
                    old_comp += 1
            
            best_gain = 0
            best_new = None
            for j, new_mask in enumerate(unused_list):
                # We need to check if new_mask is comparable with others
                # But we need to be careful: if we swap, the set of "others" changes slightly?
                # No, "others" is best_family \ {best_family[i]}
                # So we just need comparables(new_mask, best_family \ {best_family[i]})
                
                new_comp = 0
                for k, mk in enumerate(best_family):
                    if k == i:
                        continue
                    if is_comparable(new_mask, mk):
                        new_comp += 1
                
                gain = new_comp - old_comp
                if gain > best_gain:
                    best_gain = gain
                    best_new = j
                    
            if best_new is not None and best_gain > 0:
                best_family[i] = unused_list[best_new]
                unused_list[best_new] = old_mask
                best_score += best_gain
                improved = True
                
        # 2-swap polish (sampled to save time, or exhaustive if fast enough)
        # Exhaustive 2-swap is O(m^2 * unused^2 * m) which is too slow.
        # We'll do a few random 2-swaps to see if we can improve
        for _ in range(100):
            if time.time() > deadline - 1:
                break
            i1 = random.randint(0, len(best_family) - 1)
            i2 = random.randint(0, len(best_family) - 1)
            if i1 == i2:
                continue
            old1 = best_family[i1]
            old2 = best_family[i2]
            
            j1 = random.randint(0, len(unused_list) - 1)
            j2 = random.randint(0, len(unused_list) - 1)
            if j1 == j2:
                continue
            new1 = unused_list[j1]
            new2 = unused_list[j2]
            
            # Calculate delta as in try_2_swap
            old_comp_12 = 1 if is_comparable(old1, old2) else 0
            old_comp_1_others = sum(1 for k, mk in enumerate(best_family) if k != i1 and k != i2 and is_comparable(old1, mk))
            old_comp_2_others = sum(1 for k, mk in enumerate(best_family) if k != i1 and k != i2 and is_comparable(old2, mk))
            total_old = old_comp_12 + old_comp_1_others + old_comp_2_others
            
            new_comp_12 = 1 if is_comparable(new1, new2) else 0
            new_comp_1_others = sum(1 for k, mk in enumerate(best_family) if k != i1 and k != i2 and is_comparable(new1, mk))
            new_comp_2_others = sum(1 for k, mk in enumerate(best_family) if k != i1 and k != i2 and is_comparable(new2, mk))
            total_new = new_comp_12 + new_comp_1_others + new_comp_2_others
            
            delta = total_new - total_old
            if delta > 0:
                best_family[i1] = new1
                best_family[i2] = new2
                unused_list[j1] = old1
                unused_list[j2] = old2
                best_score += delta
                improved = True

    # Convert to list of lists
    result = [mask_to_list(mask) for mask in best_family]
    
    # Ensure types are correct
    for i in range(len(result)):
        for j in range(len(result[i])):
            result[i][j] = int(result[i][j])
            
    # Validation
    assert len(result) == m
    assert len(set(tuple(s) for s in result)) == m
    for s in result:
        assert isinstance(s, list)
        for x in s:
            assert isinstance(x, int)
            assert not isinstance(x, bool)
            assert 0 <= x <= 11
        assert len(s) == len(set(s))
        
    return result

