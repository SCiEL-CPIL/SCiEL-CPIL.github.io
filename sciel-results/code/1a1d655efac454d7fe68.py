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
    
    # 2. Randomized Greedy Restarts
    np.random.seed(42)
    max_restarts = 200
    for i in range(max_restarts):
        if time.time() > deadline - 20: # Leave time for local search
            break
        order = np.random.permutation(n) + 1
        s, sums = greedy_build(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            best_sums = sums
    
    # 3. Local Search to try to reach 40
    # If best_size < 40, try to improve.
    if best_size < 40:
        current_set = list(best_set)
        current_sums = set(best_sums)
        current_size = best_size
        
        # We will try "remove 1, add 2" moves.
        # This allows increasing the size by 1.
        # We also try "remove 1, add 1" to shuffle.
        
        improvement_found = True
        while improvement_found and current_size < 40:
            improvement_found = False
            # Try many random removals
            # We want to find a removal that allows adding 2 new elements.
            # This is O(N^2) per removal candidate if we scan all x.
            # But we can randomize.
            
            for _ in range(500): # Try 500 random removals
                if time.time() > deadline - 5:
                    break
                if current_size >= 40:
                    break
                
                # Pick a random element to remove
                idx = np.random.randint(len(current_set))
                y_to_remove = current_set[idx]
                
                # Remove y_to_remove from current_set and current_sums
                # We need to rebuild sums or update them.
                # Rebuilding sums for K=40 is fast.
                # Sums involving y_to_remove:
                # (y,y,y), (y,y,z), (y,z,z), (y,z,w)
                # It's easier to just rebuild the sums set from scratch for the reduced set.
                
                reduced_set = [x for x in current_set if x != y_to_remove]
                reduced_sums = set()
                # Rebuild reduced_sums
                for i in range(len(reduced_set)):
                    a = reduced_set[i]
                    for j in range(i, len(reduced_set)):
                        b = reduced_set[j]
                        for k in range(j, len(reduced_set)):
                            c = reduced_set[k]
                            reduced_sums.add(a+b+c)
                
                # Now try to add 2 elements
                # We can try to add one element x1, then another x2.
                # To be efficient, we can try adding a random x1, then scan for x2.
                
                # Try 20 random candidates for x1
                candidates = np.random.choice(n, 20, replace=False) + 1
                for x1 in candidates:
                    if x1 in current_set:
                        continue
                    valid1, sums1 = is_valid_addition(x1, reduced_set, reduced_sums)
                    if not valid1:
                        continue
                    
                    # Try to add x2 to (reduced_set + [x1])
                    temp_set = reduced_set + [x1]
                    temp_sums = reduced_sums.union(sums1)
                    
                    # Try 20 random candidates for x2
                    candidates2 = np.random.choice(n, 20, replace=False) + 1
                    for x2 in candidates2:
                        if x2 in current_set or x2 == x1:
                            continue
                        valid2, sums2 = is_valid_addition(x2, temp_set, temp_sums)
                        if valid2:
                            # Success! We found a set of size current_size + 1
                            new_set = temp_set + [x2]
                            new_sums = temp_sums.union(sums2)
                            
                            current_set = new_set
                            current_sums = new_sums
                            current_size += 1
                            improvement_found = True
                            break
                    if improvement_found:
                        break
                if improvement_found:
                    break
            
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
                best_sums = current_sums

    return best_set

# EVOLVE-BLOCK-END
