import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    required = [1, 2, 5, 11]
    N = 284
    
    # Helper to calculate cost (number of duplicate sums)
    def get_cost(s):
        sums = {}
        for i in range(len(s)):
            for j in range(i, len(s)):
                val = s[i] + s[j]
                if val in sums:
                    sums[val] += 1
                else:
                    sums[val] = 1
        cost = 0
        for count in sums.values():
            if count > 1:
                cost += count - 1
        return cost

    # Greedy initialization
    def greedy_construct():
        s = list(required)
        used_sums = set()
        for i, x in enumerate(s):
            for y in s[i:]:
                used_sums.add(x + y)
        
        for x in range(12, N + 1):
            new_sums = set()
            valid = True
            for y in s:
                ns = x + y
                if ns in used_sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if valid:
                # Check 2*x
                if 2*x in used_sums or 2*x in new_sums:
                    valid = False
                else:
                    new_sums.add(2*x)
            
            if valid:
                s.append(x)
                used_sums |= new_sums
        return s

    best_set = greedy_construct()
    best_size = len(best_set)
    
    # Try to find larger sets using SA
    # We fix the required elements.
    # We search for sets of size K.
    
    def sa_search(K, time_limit):
        if K < best_size:
            return None
        
        # Initialize state
        # Fixed: required
        # Free: K - 4 elements
        free_count = K - 4
        if free_count < 0:
            return None
            
        # Initial random free elements
        # Ensure distinct and not in required
        available = [x for x in range(1, N + 1) if x not in required]
        
        s = list(required)
        # Randomly pick free elements
        # To avoid initial conflicts, maybe pick somewhat spread out?
        # Just random for now.
        free_elements = random.sample(available, free_count)
        s.extend(free_elements)
        s.sort()
        
        current_cost = get_cost(s)
        if current_cost == 0:
            return s
            
        T = 10.0
        cooling_rate = 0.999
        no_improve_count = 0
        max_no_improve = 10000
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if no_improve_count > max_no_improve:
                # Restart with new random state
                free_elements = random.sample(available, free_count)
                s = list(required) + free_elements
                s.sort()
                current_cost = get_cost(s)
                if current_cost == 0:
                    return s
                no_improve_count = 0
                T = 10.0
                
            # Pick a random index to change
            # We can change any element, but required ones are fixed?
            # The problem says "containing 1, 2, 5 and 11".
            # It doesn't say ONLY those are fixed, but we must include them.
            # So we can move them?
            # "distinct integers in 1..284 containing 1, 2, 5 and 11"
            # This implies the set MUST have these elements.
            # So we cannot remove them. We can only add/replace the others.
            # So indices 0..3 (after sort? no, values are fixed)
            # Let's identify indices of required elements.
            
            # It is safer to treat the set as a bag of values.
            # We maintain a list `s`.
            # We must ensure 1, 2, 5, 11 are in `s`.
            # If we pick an index to change, and that value is in required, we cannot change it to something else.
            # So we only pick indices where s[i] is NOT in required.
            
            movable_indices = [i for i, val in enumerate(s) if val not in required]
            
            if not movable_indices:
                # All are required? Then K=4.
                return s if current_cost == 0 else None
                
            idx = random.choice(movable_indices)
            old_val = s[idx]
            
            # Pick new value
            # Exclude current values in s
            current_vals = set(s)
            candidates = [x for x in range(1, N + 1) if x not in current_vals]
            if not candidates:
                break
            new_val = random.choice(candidates)
            
            s[idx] = new_val
            s.sort() # Keep sorted for consistency? Not strictly needed for cost, but good.
            
            new_cost = get_cost(s)
            
            if new_cost < current_cost:
                current_cost = new_cost
                no_improve_count = 0
                if current_cost == 0:
                    return s
            else:
                # Accept with probability
                delta = new_cost - current_cost
                if delta == 0:
                    # Keep? Maybe
                    if random.random() < 0.1:
                        current_cost = new_cost
                        no_improve_count += 1
                    else:
                        s[idx] = old_val
                        s.sort()
                        no_improve_count += 1
                else:
                    prob = pow(2.718281828, -delta / T)
                    if random.random() < prob:
                        current_cost = new_cost
                        no_improve_count += 1
                    else:
                        s[idx] = old_val
                        s.sort()
                        no_improve_count += 1
            
            T *= cooling_rate
            if T < 0.01:
                T = 0.01
        
        return None

    # Try sizes from 20 down to best_size + 1
    for K in range(20, best_size, -1):
        if time.time() > deadline:
            break
        
        # Allocate time based on K
        # Give more time to larger K
        time_alloc = 10.0 if K >= 20 else 5.0
        if time.time() + time_alloc > deadline:
            time_alloc = deadline - time.time()
            if time_alloc < 1.0:
                break
                
        result = sa_search(K, time_alloc)
        if result:
            best_set = result
            best_size = K
            # If we found 20, we can stop?
            # The max is 20.
            if K == 20:
                break

    return best_set

if __name__ == "__main__":
    print(construct())
