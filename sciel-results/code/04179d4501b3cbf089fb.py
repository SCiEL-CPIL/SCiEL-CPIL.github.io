import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 2s safety margin
    
    # Precompute reciprocals
    reciprocals = {i: Fraction(1, i) for i in range(1, 23)}
    all_elements = list(range(1, 23))
    
    best_set = []
    best_size = 0
    
    # Helper to check if an element can be added
    def check_addable(current_sums, current_seen, element):
        r = reciprocals[element]
        new_sums = []
        # We need to check if s + r is in current_seen for all s in current_sums
        # If any is, return None
        # Otherwise, return the list of new sums
        for s in current_sums:
            ns = s + r
            if ns in current_seen:
                return None
            new_sums.append(ns)
        return new_sums

    # Helper to build sums for a set from scratch (for verification or initialization)
    def build_sums(elements):
        sums = [Fraction(0, 1)]
        seen = {Fraction(0, 1)}
        for e in elements:
            r = reciprocals[e]
            new_sums = []
            for s in sums:
                ns = s + r
                if ns in seen:
                    return None, None # Conflict
                new_sums.append(ns)
            sums.extend(new_sums)
            seen.update(new_sums)
        return sums, seen

    # Main loop: Randomized Greedy + Local Search
    while time.time() < deadline:
        # 1. Randomized Greedy Construction
        random.shuffle(all_elements)
        current_set = []
        current_sums = [Fraction(0, 1)]
        current_seen = {Fraction(0, 1)}
        
        for e in all_elements:
            new_sums = check_addable(current_sums, current_seen, e)
            if new_sums is not None:
                current_set.append(e)
                current_sums.extend(new_sums)
                current_seen.update(new_sums)
        
        # 2. Local Search: 1-for-many swap
        # Try removing each element and seeing if we can add more
        improved = True
        while improved and time.time() < deadline:
            improved = False
            remaining = [e for e in all_elements if e not in current_set]
            
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                
                # Remove element i
                removed_elem = current_set[i]
                
                # Rebuild sums without removed_elem
                # This is expensive, so we do it carefully.
                # Instead of rebuilding from scratch, we can't easily "subtract" from a set of sums.
                # So we rebuild.
                test_set = current_set[:i] + current_set[i+1:]
                test_sums, test_seen = build_sums(test_set)
                
                if test_sums is None:
                    continue # Should not happen if current_set was valid
                
                # Try to greedily add from remaining
                # We want to see if we can add MORE than 1 element (i.e., at least 2)
                # Or if we can add 1 element that allows a larger total?
                # Simple heuristic: Greedily add as many as possible.
                # If the final size > len(current_set), accept.
                
                test_sums_copy = test_sums[:]
                test_seen_copy = set(test_seen)
                
                # Greedy fill
                for e in remaining:
                    new_sums = check_addable(test_sums_copy, test_seen_copy, e)
                    if new_sums is not None:
                        test_sums_copy.extend(new_sums)
                        test_seen_copy.update(new_sums)
                        test_set.append(e)
                
                if len(test_set) > len(current_set):
                    current_set = test_set
                    current_sums = test_sums_copy
                    current_seen = test_seen_copy
                    improved = True
                    remaining = [e for e in all_elements if e not in current_set]
                    break # Restart the inner loop with the new set
        
        # Update global best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we found a very good set, we might want to try harder to improve it
            # But the loop above already does local search.
            
            # Optional: Try to verify and maybe do a more aggressive local search on the best set
            # The local search above is already quite strong.
            
    # Final Verification
    if not best_set:
        return [1]
        
    # Verify the best set is valid
    sums, seen = build_sums(best_set)
    if sums is None:
        # Fallback to a known valid set
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
