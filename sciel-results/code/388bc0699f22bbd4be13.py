# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 seconds safety margin (limit 60s)
    
    def get_subset_sums(elements):
        """Returns a set of all non-empty subset sums."""
        sums = {0}
        for x in elements:
            new_sums = set()
            for s in sums:
                new_sums.add(s + x)
            sums.update(new_sums)
        sums.discard(0)
        return sums

    def is_valid_fast(sums_set):
        """Check if no two sums divide each other using Numpy."""
        if not sums_set:
            return True
        
        max_sum = max(sums_set)
        # Max sum for 30 elements of max 256 is 7680. 
        # We allocate a slightly larger buffer to be safe or dynamic.
        # Dynamic allocation is fine.
        size = max_sum + 1
        arr = np.zeros(size, dtype=bool)
        for s in sums_set:
            arr[s] = True
        
        # Iterate through sums. 
        # Optimization: Only check 'a' if it is not 1 (1 divides everything).
        # Actually, if 1 is in sums, it's invalid immediately unless set size is 1? 
        # No, if 1 is in sums, and any other sum s exists, 1|s. 
        # So if len(sums_set) > 1 and 1 in sums_set: return False.
        if 1 in sums_set and len(sums_set) > 1:
            return False
            
        for a in sums_set:
            if a == 1:
                continue
            # Check multiples 2a, 3a, ...
            # arr[2*a::a] gives the slice from 2a to end with step a
            # If any is True, then a multiple exists in the set.
            if np.any(arr[2*a::a]):
                return False
        return True

    def try_add(x, current_sums, current_elements):
        """
        Attempts to add x to the set.
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        if x in current_elements:
            return False, None
            
        # Check for collisions (duplicates)
        # 1. x must not be in S.
        if x in current_sums:
            return False, None
        
        # 2. x + s must not be in S for any s in S.
        for s in current_sums:
            if (s + x) in current_sums:
                return False, None
        
        # Construct new_sums
        new_sums = set(current_sums)
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)
        
        # Check validity of new_sums
        if not is_valid_fast(new_sums):
            return False, None
            
        return True, new_sums

    def greedy_construct(ordering):
        """
        Greedy construction using a specific ordering of candidates.
        """
        elements = []
        sums = set()
        
        for x in ordering:
            if x in elements:
                continue
            ok, new_sums = try_add(x, sums, elements)
            if ok:
                elements.append(x)
                sums = new_sums
                
        return elements, sums

    def local_search(initial_elements, initial_sums, deadline):
        """
        Local search to improve the set size.
        """
        current_elements = list(initial_elements)
        current_sums = set(initial_sums)
        best_elements = list(current_elements)
        best_sums = set(current_sums)
        
        all_candidates = list(range(1, 257))
        
        # Loop until deadline
        while time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add a random element
            random.shuffle(all_candidates)
            for x in all_candidates:
                if x in current_elements:
                    continue
                ok, new_sums = try_add(x, current_sums, current_elements)
                if ok:
                    current_elements.append(x)
                    current_sums = new_sums
                    if len(current_elements) > len(best_elements):
                        best_elements = list(current_elements)
                        best_sums = set(current_sums)
                    improved = True
                    break
            
            if improved:
                continue
                
            # 2. If no add possible, try a swap
            if not current_elements:
                break
                
            # Remove a random element
            idx = random.randint(0, len(current_elements) - 1)
            x_remove = current_elements[idx]
            
            # Recalculate sums for the reduced set
            new_elements = [e for e in current_elements if e != x_remove]
            new_sums = get_subset_sums(new_elements)
            
            # Try to add candidates to the reduced set
            # We want to add at least 1 element to restore size, ideally 2 to increase size.
            
            # Try a few random swaps
            for _ in range(20):
                if time.time() > deadline - 2:
                    break
                    
                available = [x for x in all_candidates if x not in new_elements]
                if not available:
                    break
                x_add = random.choice(available)
                
                ok, new_sums_after_add = try_add(x_add, new_sums, new_elements)
                if ok:
                    final_elements = new_elements + [x_add]
                    final_sums = new_sums_after_add
                    
                    # Try to extend this new set greedily
                    random.shuffle(all_candidates)
                    for x_next in all_candidates:
                        if x_next in final_elements:
                            continue
                        ok2, new_sums2 = try_add(x_next, final_sums, final_elements)
                        if ok2:
                            final_elements.append(x_next)
                            final_sums = new_sums2
                            if len(final_elements) > len(best_elements):
                                best_elements = list(final_elements)
                                best_sums = set(final_sums)
                                improved = True
                                current_elements = final_elements
                                current_sums = final_sums
                                break
                    if improved:
                        break
                    else:
                        # Adopt the same-size set to escape local minima
                        current_elements = final_elements
                        current_sums = final_sums
                        break
                        
        return best_elements, best_sums

    # Main logic
    best_elements = []
    best_sums = set()
    
    # Strategies for ordering
    # 1. Descending (Large numbers are safer)
    # 2. Random
    # 3. Ascending
    # 4. "Binary-like": Prioritize numbers with high 2-adic valuations? 
    #    Or just random with a bias towards large numbers.
    
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        
        # Choose a strategy
        if restart_count % 3 == 0:
            order = list(range(256, 0, -1))
        elif restart_count % 3 == 1:
            order = list(range(1, 257))
            random.shuffle(order)
        else:
            # Heuristic: Sort by value descending, but with random jitter
            order = list(range(1, 257))
            random.shuffle(order)
            order.sort(key=lambda x: -x) # Back to descending? 
            # Let's try a "Large First" random mix
            large = list(range(128, 257))
            small = list(range(1, 128))
            random.shuffle(large)
            random.shuffle(small)
            order = large + small
            
        elements, sums = greedy_construct(order)
        
        if len(elements) > len(best_elements):
            best_elements = elements
            best_sums = sums
            
        # Local search
        new_best_elements, new_best_sums = local_search(best_elements, best_sums, deadline)
        if len(new_best_elements) > len(best_elements):
            best_elements = new_best_elements
            best_sums = new_best_sums
            
    # Final verification and cleanup
    result = [int(x) for x in best_elements]
    
    final_sums = get_subset_sums(result)
    if not is_valid_fast(final_sums):
        # Fallback: simple greedy descending
        fallback_elements = []
        fallback_sums = set()
        for x in range(256, 0, -1):
            ok, new_sums = try_add(x, fallback_sums, fallback_elements)
            if ok:
                fallback_elements.append(x)
                fallback_sums = new_sums
        result = [int(x) for x in fallback_elements]
        
    return result

# EVOLVE-BLOCK-END
