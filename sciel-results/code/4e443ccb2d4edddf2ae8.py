import time
import random

def construct():
    N = 1000
    max_sum = 2 * N
    best_set = []
    best_len = 0
    start_time = time.time()
    timeout = 115  # Safety margin

    def can_add(chosen_set, counts, x):
        """Check if x can be added without violating B2[2] constraint."""
        if counts[2 * x] + 1 > 2:
            return False
        for y in chosen_set:
            if y != x:
                if counts[x + y] + 1 > 2:
                    return False
        return True

    def add_element(chosen_list, chosen_set, counts, x):
        for y in chosen_list:
            if y != x:
                counts[x + y] += 1
        counts[2 * x] += 1
        chosen_list.append(x)
        chosen_set.add(x)

    def remove_element(chosen_list, chosen_set, counts, x):
        for y in chosen_list:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        chosen_list.remove(x)
        chosen_set.discard(x)

    def greedy_insert(chosen_list, chosen_set, counts, candidates=None):
        if candidates is None:
            candidates = list(range(1, N + 1))
            random.shuffle(candidates)
        added_any = False
        for x in candidates:
            if x not in chosen_set:
                if can_add(chosen_set, counts, x):
                    add_element(chosen_list, chosen_set, counts, x)
                    added_any = True
        return added_any

    def local_search(chosen_list, chosen_set, counts):
        # Phase 1: Pure insertion
        while greedy_insert(chosen_list, chosen_set, counts):
            pass
        
        # Phase 2: Swap search
        improved = True
        while improved:
            improved = False
            to_try = chosen_list[:]
            random.shuffle(to_try)
            
            for x in to_try:
                if time.time() - start_time > timeout:
                    return chosen_list, chosen_set, counts
                
                # Remove x
                remove_element(chosen_list, chosen_set, counts)
                
                # Try to find a y that allows an insertion
                # Use a fixed order or random? Random helps avoid local minima.
                candidates = list(range(1, N + 1))
                random.shuffle(candidates)
                
                success = False
                for y in candidates:
                    if y in chosen_set:
                        continue
                    if can_add(chosen_set, counts, y):
                        # Add y
                        add_element(chosen_list, chosen_set, counts, y)
                        
                        # Check if we can add any z
                        # We pass a new random candidate list to greedy_insert
                        if greedy_insert(chosen_list, chosen_set, counts):
                            # Success: size increased
                            success = True
                            improved = True
                            break
                        else:
                            # Revert y
                            remove_element(chosen_list, chosen_set, counts)
                
                if success:
                    break
                else:
                    # Revert x
                    add_element(chosen_list, chosen_set, counts, x)

        return chosen_list, chosen_set, counts

    # Main loop: Multi-start
    while time.time() - start_time < timeout:
        chosen_list = []
        chosen_set = set()
        counts = [0] * (max_sum + 1)
        
        # Run local search
        chosen_list, chosen_set, counts = local_search(chosen_list, chosen_set, counts)
        
        if len(chosen_list) > best_len:
            best_len = len(chosen_list)
            best_set = chosen_list[:]
            
        # Early stop if we find a very good solution
        # Theoretical max is ~89.
        if best_len >= 88:
            break

    return best_set
