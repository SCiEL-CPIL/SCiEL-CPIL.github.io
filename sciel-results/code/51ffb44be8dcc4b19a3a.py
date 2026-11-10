import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s
    
    def is_dissociated(elements):
        """Check if {1/e for e in elements} is dissociated (all subset sums distinct)."""
        n = len(elements)
        if n == 0:
            return True
        reciprocals = [Fraction(1, e) for e in elements]
        seen = {Fraction(0, 1)}
        sums = [Fraction(0, 1)]
        for r in reciprocals:
            new_sums = []
            for s in sums:
                ns = s + r
                if ns in seen:
                    return False
                new_sums.append(ns)
                seen.add(ns)
            sums.extend(new_sums)
        return True

    all_elements = list(range(1, 23))
    
    # 1. Try to find a large set using greedy with multiple random orders
    best_set = []
    
    # Deterministic orders
    orders = [
        all_elements,
        all_elements[::-1],
        sorted(all_elements, key=lambda x: -x),
    ]
    
    # Add some random orders
    for _ in range(20):
        random_order = all_elements[:]
        random.shuffle(random_order)
        orders.append(random_order)
        
    for order in orders:
        if time.time() > deadline:
            break
        current = []
        for e in order:
            if time.time() > deadline:
                break
            candidate = current + [e]
            if is_dissociated(candidate):
                current = candidate
        if len(current) > len(best_set):
            best_set = current[:]
            
    # 2. Local Search: Try to improve best_set
    # Strategy: If we can't add any element, try removing one element that allows adding more.
    improved = True
    while improved and time.time() < deadline:
        improved = False
        remaining = [e for e in all_elements if e not in best_set]
        
        # Try to add any remaining element
        added = False
        for e in remaining[:]:
            if time.time() > deadline:
                break
            candidate = best_set + [e]
            if is_dissociated(candidate):
                best_set = candidate
                remaining.remove(e)
                improved = True
                added = True
                break # Restart the loop to check for more additions
        
        if added:
            continue
            
        # If we couldn't add any, try to swap
        # For each element in best_set, remove it and see if we can add more
        best_swap = None
        best_swap_gain = 0
        
        for e_to_remove in best_set:
            if time.time() > deadline:
                break
            temp_set = [x for x in best_set if x != e_to_remove]
            
            # Try to add elements from remaining
            temp_remaining = remaining[:]
            new_set = temp_set
            count_added = 0
            
            # Greedily add from temp_remaining
            for e_add in temp_remaining[:]:
                if time.time() > deadline:
                    break
                candidate = new_set + [e_add]
                if is_dissociated(candidate):
                    new_set = candidate
                    temp_remaining.remove(e_add)
                    count_added += 1
                    
            # Gain is count_added (since we removed 1)
            # We want to maximize the final size, which is len(best_set) - 1 + count_added
            # So gain is count_added - 1
            if count_added > 1: # Net gain > 0
                gain = count_added - 1
                if gain > best_swap_gain:
                    best_swap_gain = gain
                    best_swap = (e_to_remove, new_set)
                    
        if best_swap:
            e_to_remove, new_set = best_swap
            best_set = new_set
            improved = True
            
    # Fallback
    if not best_set or len(best_set) <= 9:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    if not is_dissociated(best_set):
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
