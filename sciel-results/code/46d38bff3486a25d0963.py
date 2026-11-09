import time
import random

N = 256

def construct():
    # Deadline guard
    deadline = time.time() + 25.0
    
    # Initialize with a greedy ascending result to have a valid fallback
    # Though random search will likely find something better quickly
    def greedy_ascending():
        selected = []
        sums = set()
        for x in range(1, N + 1):
            new_sums = {2 * x} | {x + a for a in selected}
            if not new_sums.intersection(sums):
                selected.append(x)
                sums.update(new_sums)
        return selected

    best_set = greedy_ascending()
    best_size = len(best_set)
    
    # Randomized search
    while time.time() < deadline:
        # Random permutation
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        
        current_set = []
        sums = set()
        
        for x in perm:
            # Check if x can be added
            # Sums to check: 2x and x+a for all a in current_set
            # If any of these are in sums, x cannot be added
            # Also, we don't need to check distinctness of new sums among themselves 
            # because a_i != a_j implies x+a_i != x+a_j, and 2x != x+a implies x!=a (true).
            # So we just check intersection with existing sums.
            
            # Optimization: check 2x first
            if 2 * x in sums:
                continue
            
            # Check x+a
            can_add = True
            for a in current_set:
                if (x + a) in sums:
                    can_add = False
                    break
            
            if can_add:
                current_set.append(x)
                sums.add(2 * x)
                for a in current_set[:-1]: # Exclude x itself? No, we added x to current_set, so a is previous elements
                    sums.add(x + a)
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set
            # If we find a very small set, we might want to continue to see if we can go smaller
            # But 6 or 7 is likely the minimum. 
            # We can break if best_size <= 5, but let's run to the end to be sure.
            
    return best_set
