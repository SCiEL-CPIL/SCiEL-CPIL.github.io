import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    N = 128
    
    def is_valid(S):
        """
        S is a set of ints.
        Returns True if S is valid.
        """
        if len(S) < 3:
            return True
            
        # Use a list for iteration, set for lookup
        S_set = S
        S_list = list(S)
        n = len(S_list)
        
        # Sort to ensure a < b < c
        S_list.sort()
        
        for i in range(n):
            a = S_list[i]
            for j in range(i + 1, n):
                b = S_list[j]
                s_ab = a + b
                if s_ab > N:
                    break # Since c > b, a+c > a+b, so all further sums > N?
                    # No, a+c increases with c. But we are iterating j (b).
                    # If a+b > N, then for any c > b, a+c > a+b > N.
                    # So we can break the inner loop?
                    # Wait, the condition is s_ab in S.
                    # If s_ab > N, it's not in S. So no violation with this a,b.
                    # Can we break?
                    # If a+b > N, then a+c > N for all c > b.
                    # So no violation for this a with any b' >= b.
                    # So we can break the j loop.
                    break
                
                if s_ab not in S_set:
                    continue
                
                # Check for c
                # c must be > b
                for k in range(j + 1, n):
                    c = S_list[k]
                    s_ac = a + c
                    if s_ac > N:
                        break # Further c will also be > N
                    s_bc = b + c
                    if s_bc > N:
                        break # Further c will also be > N
                    
                    if s_ac in S_set and s_bc in S_set:
                        return False
        return True

    # Initial Set: All odds
    best_set = set(range(1, N + 1, 2))
    best_size = len(best_set)
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    current_set = set(best_set)
    
    while time.time() < deadline and T > min_T:
        # Generate neighbor
        if random.random() < 0.5:
            # Add random element
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            
            new_set = set(current_set)
            new_set.add(x)
            
            if is_valid(new_set):
                delta = len(new_set) - len(current_set)
                if delta > 0 or random.random() < pow(2.718, delta / T):
                    current_set = new_set
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
        else:
            # Swap
            if not current_set:
                continue
            y = random.choice(list(current_set))
            candidates = [x for x in range(1, N + 1) if x not in current_set]
            if not candidates:
                continue
            x = random.choice(candidates)
            
            new_set = set(current_set)
            new_set.remove(y)
            new_set.add(x)
            
            if is_valid(new_set):
                # Accept swap (delta = 0)
                # To avoid cycling, we might want to accept with some prob?
                # In SA, neutral moves are accepted with prob 1.
                current_set = new_set
                # Note: best_set is not updated here as size is same.
                # But this new structure might be better for future adds.
        
        T *= cooling
    
    # Final greedy fill on best_set
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if x in best_set:
                continue
            new_set = set(best_set)
            new_set.add(x)
            if is_valid(new_set):
                best_set = new_set
                changed = True
                if len(best_set) > best_size:
                    best_size = len(best_set)
    
    return [int(x) for x in best_set]
