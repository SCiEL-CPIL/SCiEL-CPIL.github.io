"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _check_can_add(x, selected):
    """
    Check if x can be added to selected without violating the condition.
    Condition: No element divides the sum of any 3 distinct OTHER selected elements.
    
    When adding x to selected (S), we need to check:
    1. x does not divide sum of any 3 elements in S.
    2. For each a in S, a does not divide sum of x and any 2 other elements in S \ {a}.
    """
    n = len(selected)
    if n < 2:
        return True
    
    # 1. Check if x divides sum of any 3 elements in S
    # If n < 3, no such triple exists in S, so this check passes trivially for x being the divisor.
    if n >= 3:
        for i in range(n - 2):
            si = selected[i]
            for j in range(i + 1, n - 1):
                sij = si + selected[j]
                for k in range(j + 1, n):
                    if (sij + selected[k]) % x == 0:
                        return False

    # 2. Check if any a in S divides sum of x and 2 other elements in S
    # For a fixed a at index i, we need to pick 2 other indices j, k from S \ {i}.
    if n >= 2:
        for i, a in enumerate(selected):
            # We need to pick 2 elements from S excluding index i.
            # There are n-1 elements available. We need to check all pairs (j, k) from these n-1.
            # To do this efficiently, we can iterate j, k over all indices, skipping i.
            # But we must ensure j != k.
            # Let's optimize: 
            # Sum = x + selected[j] + selected[k].
            # We can precompute nothing, just loop.
            # Number of pairs is (n-1)*(n-2)/2.
            # Total complexity O(n^3) worst case per check.
            
            # Optimization: If a > x + 2*192, it can't divide the sum (max sum 3*192=576).
            # Actually max sum is x + 192 + 191.
            # If a > x + 383, skip? 
            # Let's just do the check.
            
            for j in range(n):
                if j == i:
                    continue
                sj = selected[j]
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    s = x + sj + selected[k]
                    if s % a == 0:
                        return False

    return True

def _check_validity(selected):
    """
    Full check for validity. Used for final verification or initialization.
    """
    n = len(selected)
    if n < 4:
        return True
    for i, a in enumerate(selected):
        # Check if a divides sum of any 3 others
        # Others are indices j, k, l != i
        found = False
        for j in range(n):
            if j == i: continue
            sj = selected[j]
            for k in range(j + 1, n):
                if k == i: continue
                sk = selected[k]
                for l in range(k + 1, n):
                    if l == i: continue
                        # sum = sj + sk + selected[l]
                    if (sj + sk + selected[l]) % a == 0:
                        return False
    return True

def _greedy_construct(order, deadline):
    """Greedy construction: add elements in given order if valid."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _check_can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, deadline):
    """
    Improve set by trying:
    1. Add 1
    2. Swap 1 for 1
    3. Swap 1 for 2 (increase size by 1)
    4. Swap 2 for 2 (maintain size, escape local optimum)
    """
    best = list(selected)
    best_set = set(best)
    improved = True
    no_improve_count = 0
    
    # We use a "temperature" or just random selection for swaps
    # To keep it simple and fast, we'll do passes.
    
    while improved and time.time() < deadline - 1.0:
        improved = False
        no_improve_count = 0
        
        # 1. Try adding any element
        # Sort candidates by value descending? Large numbers are safer.
        candidates = [x for x in range(192, 0, -1) if x not in best_set]
        random.shuffle(candidates) # Randomize to avoid bias
        
        for x in candidates:
            if time.time() > deadline - 1.0:
                break
            if _check_can_add(x, best):
                best.append(x)
                best_set.add(x)
                improved = True
                break # Restart loop with new state
        
        if improved:
            continue
            
        # If no single addition worked, try swaps
        
        # 2. Try Swap 1-for-2 (Remove one, add two)
        # This increases size by 1.
        if len(best) > 0:
            # Pick a random element to remove
            # We prioritize removing small elements as they are more likely to be "bad" divisors
            # But let's just randomize for now, or sort by value.
            # Small elements are more likely to divide sums, so removing them is good.
            # Let's try removing the smallest element first? Or random?
            # Let's try a few random removals.
            for _ in range(5):
                if time.time() > deadline - 1.0:
                    break
                idx = random.randint(0, len(best) - 1)
                removed = best[idx]
                temp_best = [e for i, e in enumerate(best) if i != idx]
                temp_set = set(temp_best)
                
                # Try to add 2 elements
                # We need to find x, y not in temp_set such that temp_best + [x, y] is valid.
                # This is harder. We can greedily add one, then try to add another.
                # But adding one might block the other.
                # Let's just try to add elements one by one to temp_best, but allow size to grow.
                # Actually, if we remove 1, size is N-1. We want to add 2 to get N+1.
                # So we just run a greedy add loop on temp_best, but we must ensure we add exactly 2?
                # No, we just want to maximize size. If we can add 2, great. If we can add 1, that's a swap 1-for-1 (size same).
                # If we can add 0, size decreases.
                
                # Let's try to add elements to temp_best greedily.
                # We prefer large elements.
                add_candidates = [x for x in range(192, 0, -1) if x not in temp_set]
                
                added = 0
                new_best = list(temp_best)
                new_set = set(temp_best)
                
                for x in add_candidates:
                    if added >= 2: # We only want to try to add 2 to see if we can grow
                        break
                    if _check_can_add(x, new_best):
                        new_best.append(x)
                        new_set.add(x)
                        added += 1
                
                if len(new_best) > len(best):
                    best = new_best
                    best_set = new_set
                    improved = True
                    break
                elif len(new_best) == len(best):
                    # It's a swap 1-for-1. This might help escape a local optimum for future additions.
                    # We accept it with some probability or if it looks "better" (e.g., larger elements).
                    # For now, let's accept it if it allows further improvement? 
                    # To keep it simple, let's only accept if size increases.
                    # However, 1-for-1 swaps are crucial for 2-for-2.
                    pass

        # 3. Try Swap 2-for-2 (Remove two, add two)
        # This maintains size but changes composition.
        if len(best) >= 2:
            # Try a few random pairs to remove
            for _ in range(5):
                if time.time() > deadline - 1.0:
                    break
                # Pick 2 distinct indices
                idx1 = random.randint(0, len(best) - 1)
                idx2 = random.randint(0, len(best) - 1)
                if idx1 == idx2:
                    continue
                
                # Create temp_best
                indices_to_remove = {idx1, idx2}
                temp_best = [e for i, e in enumerate(best) if i not in indices_to_remove]
                temp_set = set(temp_best)
                
                # Try to add 2 elements
                add_candidates = [x for x in range(192, 0, -1) if x not in temp_set]
                
                new_best = list(temp_best)
                new_set = set(temp_best)
                added = 0
                
                for x in add_candidates:
                    if added >= 2:
                        break
                    if _check_can_add(x, new_best):
                        new_best.append(x)
                        new_set.add(x)
                        added += 1
                
                if len(new_best) == len(best):
                    # Size is same. Do we accept?
                    # We should accept if it leads to a state where we can add more?
                    # Hard to know. 
                    # Let's accept it with a small probability to explore the space.
                    # Or, compare "quality". E.g., sum of elements? Larger sum is better?
                    # Or just random restarts handle this.
                    # Let's not accept 2-for-2 swaps unless they enable an immediate 3rd addition.
                    # Try to add a 3rd element to new_best
                    if len(new_best) + 1 > len(best):
                         # Check if we can add one more
                         for y in add_candidates:
                             if y in new_set:
                                 continue
                             if _check_can_add(y, new_best):
                                 new_best.append(y)
                                 new_set.add(y)
                                 break
                         if len(new_best) > len(best):
                             best = new_best
                             best_set = new_set
                             improved = True
                             break
                    # If no immediate improvement, skip this swap to save time.

    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Initial greedy from high to low
    # Large numbers are generally safer as divisors.
    order = list(range(192, 0, -1))
    best = _greedy_construct(order, deadline)
    
    # Randomized restarts
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline - 15.0:
            break
        order = list(range(192, 0, -1))
        # Bias towards large numbers? 
        # Just shuffle for now.
        random.shuffle(order)
        candidate = _greedy_construct(order, deadline)
        if len(candidate) > len(best):
            best = candidate
    
    # Local search on the best found
    if time.time() < deadline - 1.0:
        improved = _local_search(best, deadline)
        if len(improved) > len(best):
            best = improved
            
    # Final check
    if not _check_validity(best):
        # Fallback to a known valid small set if something went wrong
        # e.g. {192, 191, 190, 189}
        # Let's just return the best valid prefix we found?
        # The greedy ensures validity at each step, so best should be valid.
        pass

    if len(best) == 0:
        best = [192]
        
    return best
# EVOLVE-BLOCK-END
