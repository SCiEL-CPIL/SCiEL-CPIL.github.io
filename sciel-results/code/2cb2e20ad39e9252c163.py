import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    fixed_elements = [1, 2, 5, 11]
    fixed_set = set(fixed_elements)
    N = 284
    max_possible_size = 20
    
    # All available values that are not in the fixed set
    all_values = list(range(1, N + 1))
    candidates = [x for x in all_values if x not in fixed_set]
    
    # Helper to check if a set is a valid Sidon set (cost 0)
    # Returns the cost (number of "excess" collisions)
    def calc_cost(s):
        # s is a list of distinct integers
        sums = {}
        n = len(s)
        for i in range(n):
            for j in range(i, n):
                val = s[i] + s[j]
                sums[val] = sums.get(val, 0) + 1
        
        cost = 0
        for c in sums.values():
            if c > 1:
                cost += c - 1
        return cost

    # Greedy construction to get a baseline
    def greedy_construct():
        s = list(fixed_elements)
        sums = set()
        for i, x in enumerate(s):
            for y in s[i:]:
                sums.add(x + y)
        
        # Try to add elements in increasing order? Or random?
        # Increasing order is deterministic and might fail early if 3,4 etc are blocked.
        # Randomized greedy is better for finding large sets.
        # Let's try a simple greedy first.
        for x in sorted(candidates):
            # Check if x can be added
            # New sums: x + y for all y in s, and 2*x
            new_sums = {x + y for y in s} | {2 * x}
            if not (new_sums & sums):
                s.append(x)
                sums |= new_sums
                if len(s) == max_possible_size:
                    return s
        return s

    best_set = greedy_construct()
    best_size = len(best_set)
    
    if best_size == max_possible_size:
        return best_set

    # Simulated Annealing to try to reach max size
    # We try sizes from max_possible_size down to best_size + 1
    
    # SA Parameters
    # We want to minimize cost to 0.
    
    def run_sa(target_size, time_limit):
        num_vars = target_size - len(fixed_elements)
        if num_vars < 0:
            return None
        
        end_time = time.time() + time_limit
        
        # Multiple restarts
        while time.time() < end_time:
            # Initialize
            # Start with fixed + random sample of vars
            # Sort to maintain consistency? Not strictly necessary for cost, but good for output
            random_vars = random.sample(candidates, num_vars)
            s = fixed_elements + random_vars
            random.shuffle(s) # Shuffle positions? Cost is independent of order.
            # Actually, for mutation, we need to know which are fixed.
            # Let's maintain a list where we know indices.
            # Or simpler: s is a list. We pick an index. If it's in fixed_set, we skip or don't mutate.
            
            current_cost = calc_cost(s)
            if current_cost == 0:
                return sorted(set(s))
                
            T = 5.0
            T_min = 0.01
            cooling = 0.9995
            steps = 5000 # Limit steps per restart
            
            for step in range(steps):
                if time.time() > end_time:
                    return None
                if time.time() > deadline:
                    return None
                    
                # Pick a random index to mutate
                # We need to pick a variable that is NOT in fixed_set
                # To be efficient, we can just pick an index, and if it's fixed, pick another.
                # Or we can separate fixed and variable parts.
                # Let's stick to the list and check.
                
                idx = random.randint(0, target_size - 1)
                # Ensure we pick a mutable element
                while s[idx] in fixed_set:
                    idx = random.randint(0, target_size - 1)
                    
                old_val = s[idx]
                # Pick a new value
                # Must be in candidates, not in s (except at idx), and not in fixed_set
                new_val = random.choice(candidates)
                # Check if new_val is already in s (excluding idx)
                # O(K) check. K=20, so this is fine.
                if new_val == old_val:
                    continue
                if new_val in s:
                    continue
                
                s[idx] = new_val
                new_cost = calc_cost(s)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    current_cost = new_cost
                    if current_cost == 0:
                        return sorted(set(s))
                else:
                    s[idx] = old_val
                    
                T *= cooling
                if T < T_min:
                    break
        
        return None

    # Allocate time for different sizes
    # Priority is 20.
    # Give 90s to size 20.
    time_for_20 = 90
    s20 = run_sa(max_possible_size, time_for_20)
    if s20:
        return s20
        
    # If 20 failed, try 19 with remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        s19 = run_sa(19, remaining_time - 5)
        if s19:
            return s19
        # If 19 failed, try 18? 
        # The prompt says maximize size. 
        # If we can't get 20, 19 is better than the greedy result if greedy < 19.
        
    return best_set
