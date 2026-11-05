# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 4096
    
    def is_valid_addition(x, selected, sums):
        """
        Check if x can be added to selected with existing sums.
        Returns True if valid, and the new sums if valid.
        """
        new_sums = set()
        
        # (x, x, x)
        s = 3*x
        if s in sums:
            return False, None
        new_sums.add(s)
        
        # (x, x, y) and (x, y, y)
        for y in selected:
            s = 2*x + y
            if s in sums or s in new_sums:
                return False, None
            new_sums.add(s)
            
            s = x + 2*y
            if s in sums or s in new_sums:
                return False, None
            new_sums.add(s)
            
        # (x, y, z)
        for i in range(len(selected)):
            y = selected[i]
            for j in range(i+1, len(selected)):
                z = selected[j]
                s = x + y + z
                if s in sums or s in new_sums:
                    return False, None
                new_sums.add(s)
                
        return True, new_sums

    def remove_element(x, selected, sums):
        """Remove x from selected and update sums. Returns (new_selected, new_sums)."""
        # Find index of x
        idx = selected.index(x)
        new_selected = selected[:idx] + selected[idx+1:]
        
        # Rebuild sums from scratch (fast for K <= 40)
        new_sums = set()
        k = len(new_selected)
        for i in range(k):
            a = new_selected[i]
            for j in range(i, k):
                b = new_selected[j]
                for m in range(j, k):
                    c = new_selected[m]
                    new_sums.add(a + b + c)
        
        return new_selected, new_sums

    def greedy_build(order):
        selected = []
        sums = set()
        for x in order:
            valid, new_sums = is_valid_addition(x, selected, sums)
            if valid:
                selected.append(x)
                sums.update(new_sums)
        return selected, sums

    best_set = []
    best_sums = set()
    best_size = 0
    
    # 1. Initial Greedy: Standard
    standard_order = list(range(1, n+1))
    s, sums = greedy_build(standard_order)
    best_set = s
    best_sums = sums
    best_size = len(s)
    
    # 2. Randomized Greedy Restarts - increased count
    np.random.seed(42)
    max_restarts = 500
    for i in range(max_restarts):
        if time.time() > deadline - 30:  # Leave time for local search
            break
        order = np.random.permutation(n) + 1
        s, sums = greedy_build(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            best_sums = sums
    
    # 3. Local Search: try to improve by removing 1 and adding 2
    current_set = list(best_set)
    current_sums = set(best_sums)
    current_size = best_size
    
    improvement_found = True
    while improvement_found and current_size < 80:
        improvement_found = False
        
        # Try multiple rounds of random removals
        for _ in range(2000):
            if time.time() > deadline - 3:
                break
            if current_size >= 80:
                break
            
            # Pick a random element to remove
            idx = np.random.randint(len(current_set))
            y_to_remove = current_set[idx]
            
            # Remove it
            reduced_set, reduced_sums = remove_element(y_to_remove, current_set, current_sums)
            
            # Try to add 2 elements
            # First, find one valid addition
            candidates = np.random.choice(n, 50, replace=False) + 1
            found_first = False
            for x1 in candidates:
                if x1 in current_set:
                    continue
                valid1, sums1 = is_valid_addition(x1, reduced_set, reduced_sums)
                if not valid1:
                    continue
                
                # Found x1, now try to find x2
                temp_set = reduced_set + [x1]
                temp_sums = reduced_sums.union(sums1)
                
                candidates2 = np.random.choice(n, 50, replace=False) + 1
                for x2 in candidates2:
                    if x2 in current_set or x2 == x1:
                        continue
                    valid2, sums2 = is_valid_addition(x2, temp_set, temp_sums)
                    if valid2:
                        # Success!
                        current_set = temp_set + [x2]
                        current_sums = temp_sums.union(sums2)
                        current_size += 1
                        improvement_found = True
                        found_first = True
                        break
                if found_first:
                    break
            
            if found_first:
                break
        
        if current_size > best_size:
            best_size = current_size
            best_set = current_set
            best_sums = current_sums

    return best_set

# EVOLVE-BLOCK-END
