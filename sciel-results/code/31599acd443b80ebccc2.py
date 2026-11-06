import time
import math
import random

def _is_dissociated(S):
    """Check if set S has distinct reciprocal subset sums."""
    if not S:
        return True
    # Compute LCM
    L = 1
    for x in S:
        L = (L * x) // math.gcd(L, x)
    
    # Compute integer weights
    weights = [L // x for x in S]
    
    # Efficient early-exit check
    seen = {0}
    for w in weights:
        new_sums = []
        for s in seen:
            ns = s + w
            if ns in seen:
                return False
            new_sums.append(ns)
        seen.update(new_sums)
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 55 # Safety margin
    
    # Initial good set (size 9)
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    current_set = best_set[:]
    
    all_elements = list(range(1, 23))
    random.seed(42)
    
    no_improve_count = 0
    max_no_improve = 500
    
    def _count_addable(S):
        """Count how many elements from [1,22] not in S can be added to S while maintaining dissociation."""
        in_set = set(S)
        count = 0
        for x in all_elements:
            if x not in in_set:
                if _is_dissociated(S + [x]):
                    count += 1
        return count
    
    while time.time() < deadline:
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            no_improve_count = 0
        
        # 1. Try to add an element - prefer elements that maximize future addability
        added = False
        remaining = [x for x in all_elements if x not in set(current_set)]
        if remaining:
            # Score each candidate by how many further elements can be added
            scored_cands = []
            for cand in remaining:
                test_set = current_set + [cand]
                if _is_dissociated(test_set):
                    # Count how many more elements can be added to this new set
                    addable = _count_addable(test_set)
                    scored_cands.append((addable, cand))
            
            if scored_cands:
                # Pick the one that maximizes future addability, with tie-breaking by random
                scored_cands.sort(key=lambda x: (-x[0], random.random()))
                best_cand = scored_cands[0][1]
                current_set = current_set + [best_cand]
                added = True
                no_improve_count = 0
        
        if not added:
            # 2. Try swaps - bias toward swaps that increase addability
            swapped = False
            in_set = set(current_set)
            remaining = [x for x in all_elements if x not in in_set]
            
            if remaining and current_set:
                # Try all possible swaps, score by resulting addability
                best_swap_score = -1
                best_swap = None
                
                for out_cand in remaining:
                    for in_cand in current_set:
                        new_set = [x for x in current_set if x != in_cand] + [out_cand]
                        if _is_dissociated(new_set):
                            addable = _count_addable(new_set)
                            if addable > best_swap_score:
                                best_swap_score = addable
                                best_swap = (in_cand, out_cand)
                
                if best_swap:
                    in_cand, out_cand = best_swap
                    current_set = [x for x in current_set if x != in_cand] + [out_cand]
                    swapped = True
            
            if not swapped:
                # 3. Try removing one element to free up space for additions
                removed = False
                for in_cand in current_set[:]:  # Copy to allow modification
                    test_set = [x for x in current_set if x != in_cand]
                    addable = _count_addable(test_set)
                    if addable > 0:
                        current_set = test_set
                        removed = True
                        no_improve_count = 0
                        break
                
                if not removed:
                    no_improve_count += 1
                    # If stuck for a while, try random restart
                    if no_improve_count > 100:
                        # Reset to best to try different additions
                        current_set = best_set[:]
                        no_improve_count = 0
                        # Or try a random restart of size len(best_set)
                        if random.random() < 0.3:
                            # Random restart
                            k = len(best_set)
                            for _ in range(10):
                                current_set = random.sample(all_elements, k)
                                if _is_dissociated(current_set):
                                    break
                            else:
                                current_set = best_set[:] # Fallback

    # Final polish: systematically try all single additions to best_set
    final_set = best_set[:]
    in_set = set(final_set)
    for x in all_elements:
        if x not in in_set:
            if _is_dissociated(final_set + [x]):
                final_set = final_set + [x]
                in_set.add(x)
    
    return sorted(final_set)
