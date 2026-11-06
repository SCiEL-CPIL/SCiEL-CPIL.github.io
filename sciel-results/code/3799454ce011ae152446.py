import time
import math
import random

def _is_dissociated(S):
    """Check if set S has distinct reciprocal subset sums using integer arithmetic."""
    if not S:
        return True
    # Compute LCM
    L = 1
    for x in S:
        L = (L * x) // math.gcd(L, x)
    
    # Compute integer weights
    weights = [L // x for x in S]
    
    # Generate all subset sums
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
    
    all_elements = list(range(1, 23))
    random.seed(42)
    
    # Start with a known good set
    # Greedy initialization: add largest first (safest)
    current_set = []
    for x in sorted(all_elements, reverse=True):
        if _is_dissociated(current_set + [x]):
            current_set.append(x)
            
    best_set = current_set[:]
    best_size = len(best_set)
    
    no_improve = 0
    
    while time.time() < deadline:
        # 1. Try to add any element
        added = False
        # Try elements in random order to diversify
        candidates = [x for x in all_elements if x not in current_set]
        random.shuffle(candidates)
        for x in candidates:
            if _is_dissociated(current_set + [x]):
                current_set.append(x)
                added = True
                no_improve = 0
                break
        
        if not added:
            # 2. Try to swap: remove one, add one (net 0 size, but different composition)
            # Or remove one, add two (net +1 size)
            
            # Try single swap to open up space
            swapped = False
            if current_set:
                # Pick a random element to remove
                idx = random.randint(0, len(current_set) - 1)
                out_elem = current_set[idx]
                current_set.pop(idx)
                
                # Try to add back or add new
                # First try to add the removed one back (undo) if no better
                # Then try to add any other
                # We want to see if removing 'out_elem' allows adding 'in_elem'
                # which then allows adding 'in_elem2'?
                
                # Simple swap: try to find an element 'in_elem' not in set
                # such that set - {out_elem} + {in_elem} is dissociated
                # AND (set - {out_elem} + {in_elem}) + {new_elem} is dissociated?
                # That's complex.
                
                # Let's just try to add any element to the reduced set
                reduced_set = current_set[:]
                # Try to add 2 elements
                added_count = 0
                candidates2 = [x for x in all_elements if x not in reduced_set]
                random.shuffle(candidates2)
                for x in candidates2:
                    if added_count >= 2:
                        break
                    if _is_dissociated(reduced_set + [x]):
                        reduced_set.append(x)
                        added_count += 1
                
                if len(reduced_set) > len(current_set):
                    current_set = reduced_set
                    no_improve = 0
                    swapped = True
                else:
                    # Revert
                    current_set.append(out_elem)
            
            if not swapped:
                no_improve += 1
                # Random restart if stuck
                if no_improve > 100:
                    # Restart with different strategy
                    if random.random() < 0.5:
                        # Greedy smallest first
                        current_set = []
                        for x in sorted(all_elements):
                            if _is_dissociated(current_set + [x]):
                                current_set.append(x)
                    else:
                        # Random start
                        current_set = random.sample(all_elements, 10)
                        # Ensure it's dissociated? If not, prune
                        # Prune: remove elements that cause collision
                        # Simple prune: keep only if dissociated
                        # This might result in small set.
                        # Better: start with empty and add random
                        current_set = []
                        rand_elems = all_elements[:]
                        random.shuffle(rand_elems)
                        for x in rand_elems:
                            if _is_dissociated(current_set + [x]):
                                current_set.append(x)
                    
                    no_improve = 0
        
        if len(current_set) > best_size:
            best_set = current_set[:]
            best_size = len(best_set)
            
        if best_size >= 18:
            break

    return sorted(best_set)
