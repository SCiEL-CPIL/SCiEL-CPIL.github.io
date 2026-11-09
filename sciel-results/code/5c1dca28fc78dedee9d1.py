import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    fixed_elements = [1, 2, 5, 11]
    N = 284
    max_size = 20
    
    # Helper to calculate cost (number of duplicate sums)
    def calc_cost(s):
        sums = {}
        for i in range(len(s)):
            for j in range(i, len(s)):
                val = s[i] + s[j]
                sums[val] = sums.get(val, 0) + 1
        # Cost is the number of sums that appear more than once
        # If a sum appears k times, it contributes (k-1) to the "excess"
        # But we just need to know if it's 0.
        # For SA, a smooth cost is better.
        # Let's return the number of pairs of sums that collide.
        # If sum 'v' appears 'c' times, it contributes c*(c-1)/2 collisions.
        collisions = 0
        for c in sums.values():
            if c > 1:
                collisions += c * (c - 1) // 2
        return collisions

    # Greedy solution as fallback
    def greedy_solution():
        a = list(fixed_elements)
        sums = set()
        for i, x in enumerate(a):
            for y in a[i:]:
                sums.add(x + y)
        
        for x in range(12, N + 1):
            new_sums = {x + y for y in a} | {2 * x}
            if len(new_sums) == len(a) + 1 and not (new_sums & sums):
                a.append(x)
                sums |= new_sums
        return a

    best_set = greedy_solution()
    best_size = len(best_set)
    
    # Try to find a larger set using Simulated Annealing
    # We will try sizes from 20 down to best_size + 1
    for target_size in range(max_size, best_size, -1):
        if time.time() > deadline:
            break
            
        # Number of variable elements
        num_vars = target_size - len(fixed_elements)
        if num_vars < 0:
            continue
            
        # Available values for variable elements
        available = [x for x in range(1, N + 1) if x not in fixed_elements]
        
        # Multiple restarts
        restarts = 50
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            # Initialize random set
            s = list(fixed_elements)
            s += random.sample(available, num_vars)
            random.shuffle(s)
            
            current_cost = calc_cost(s)
            if current_cost == 0:
                return s
                
            T = 10.0
            T_min = 0.01
            cooling_rate = 0.999
            steps = 2000
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                    
                # Pick a random variable index
                # Indices 0-3 are fixed (assuming fixed_elements are at start, but they might be shuffled)
                # Let's keep fixed elements at specific indices to simplify?
                # No, the set is unordered.
                # We can just pick any index, but if it's a fixed element, we skip or don't change it.
                # Better: maintain a list where first 4 are fixed?
                # The cost function doesn't care about order.
                # Let's just pick a random index. If it's in fixed_elements, try another.
                idx = random.randint(0, target_size - 1)
                if s[idx] in fixed_elements:
                    # Find a non-fixed index
                    # This is inefficient if many are fixed, but only 4 are fixed.
                    while s[idx] in fixed_elements:
                        idx = random.randint(0, target_size - 1)
                        
                old_val = s[idx]
                # Pick a new value from available
                new_val = random.choice(available)
                if new_val == old_val:
                    continue
                # Ensure new_val is not already in the set (distinct integers)
                # Since we sample from available which excludes fixed, we must check against other vars
                # This check is O(K).
                if new_val in s:
                    continue
                    
                s[idx] = new_val
                new_cost = calc_cost(s)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    current_cost = new_cost
                    if current_cost == 0:
                        return s
                else:
                    s[idx] = old_val
                    
                T *= cooling_rate
                if T < T_min:
                    break

    return best_set
