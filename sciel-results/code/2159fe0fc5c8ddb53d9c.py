# EVOLVE-BLOCK-START
import time
import random

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

    def is_valid(sums_set):
        """Check if no two sums divide each other."""
        if not sums_set:
            return True
        
        max_sum = max(sums_set)
        for a in sums_set:
            for k in range(2, max_sum // a + 1):
                if (a * k) in sums_set:
                    return False
        return True

    def try_add(x, current_sums, current_elements):
        """
        Attempts to add x to the set.
        Returns (True, new_sums) if valid, (False, None) otherwise.
        current_sums is the set of subset sums of current_elements.
        """
        # 1. Check for collisions (duplicates)
        if x in current_sums:
            return False, None
        
        new_sums_from_x = {s + x for s in current_sums}
        
        # Check collision with old sums
        if new_sums_from_x & current_sums:
            return False, None
            
        New = {x} | new_sums_from_x
        
        # 1. New divides Old
        max_S = max(current_sums) if current_sums else 0
        for a in New:
            for k in range(2, max_S // a + 1):
                if (a * k) in current_sums:
                    return False, None

        # 2. Old divides New
        for b in New:
            # Check divisors of b
            # If b % s == 0 for any s in current_sums, invalid.
            # Iterate divisors of b
            for i in range(1, int(b**0.5) + 1):
                if b % i == 0:
                    if i in current_sums:
                        return False, None
                    j = b // i
                    if j != i and j in current_sums:
                        return False, None

        # 3. New divides New
        max_New = max(New)
        for a in New:
            for k in range(2, max_New // a + 1):
                if (a * k) in New:
                    return False, None

        final_new_sums = current_sums | New
        return True, final_new_sums

    def greedy_construct(ordering):
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

    def local_search(initial_elements, initial_sums):
        current_elements = list(initial_elements)
        current_sums = set(initial_sums)
        best_elements = list(current_elements)
        best_sums = set(current_sums)
        
        all_candidates = list(range(1, 257))
        
        def can_add(x):
            if x in current_elements:
                return False
            return try_add(x, current_sums, current_elements)[0]
            
        def remove(x):
            new_elements = [e for e in current_elements if e != x]
            new_sums = get_subset_sums(new_elements)
            return new_elements, new_sums

        while time.time() < deadline - 2:
            improved = False
            
            # 1. Try to add a random element
            random.shuffle(all_candidates)
            added = False
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
                
            idx = random.randint(0, len(current_elements) - 1)
            x_remove = current_elements[idx]
            
            new_elements, new_sums = remove(x_remove)
            
            # Try to add candidates to the reduced set
            # We want to add at least 1 element to restore size, ideally 2 to increase size.
            
            # Let's try a few random swaps
            for _ in range(10):
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
                    
                    # Try to extend this new set
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
                        current_elements = final_elements
                        current_sums = final_sums
                        break
            
    best_elements = []
    best_sums = set()
    
    # Multiple random restarts
    for i in range(100): # Increase restarts due to speedup
        if time.time() > deadline - 5:
            break
            
        if i % 2 == 0:
            order = list(range(256, 0, -1))
        else:
            order = list(range(1, 257))
            random.shuffle(order)
            
        elements, sums = greedy_construct(order)
        
        if len(elements) > len(best_elements):
            best_elements = elements
            best_sums = sums
            
        local_search(best_elements, best_sums)
        
    result = [int(x) for x in best_elements]
    
    final_sums = get_subset_sums(result)
    if not is_valid(final_sums):
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
