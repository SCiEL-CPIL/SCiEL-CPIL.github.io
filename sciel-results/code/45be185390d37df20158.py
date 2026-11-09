<<<<<<< SEARCH
import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    fixed_elements = frozenset([1, 2, 5, 11])
    N = 284
    max_size = 20
    
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
        restarts = 200
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            # Initialize random set: fixed elements first, then variables
            s = list(fixed_elements)
            s += random.sample(available, num_vars)
            
            # Build sum frequency dictionary incrementally
            sum_counts = {}
            for i in range(len(s)):
                for j in range(i, len(s)):
                    val = s[i] + s[j]
                    sum_counts[val] = sum_counts.get(val, 0) + 1
            
            # Compute initial cost
            current_cost = 0
            for c in sum_counts.values():
                if c > 1:
                    current_cost += c * (c - 1) // 2
            
            if current_cost == 0:
                return s
                
            T = 10.0
            T_min = 0.01
            cooling_rate = 0.999
            steps = 5000
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                    
                # Pick a random variable index (indices 4..target_size-1 are variables)
                idx = random.randint(4, target_size - 1)
                
                old_val = s[idx]
                # Pick a new value from available
                new_val = random.choice(available)
                if new_val == old_val:
                    continue
                # Ensure new_val is not already in the set (distinct integers)
                if new_val in s:
                    continue
                    
                # Compute delta cost incrementally
                # Remove contributions of old_val with all other elements
                delta = 0
                for j in range(target_size):
                    if j == idx:
                        continue
                    sv = old_val + s[j]
                    c = sum_counts[sv]
                    delta -= c * (c - 1) // 2
                    sum_counts[sv] = c - 1
                    delta += (c - 1) * (c - 2) // 1
                
                # Add contributions of new_val with all other elements
                for j in range(target_size):
                    if j == idx:
                        continue
                    sv = new_val + s[j]
                    c = sum_counts.get(sv, 0)
                    delta -= c * (c - 1) // 2
                    sum_counts[sv] = c + 1
                    delta += c * (c + 1) // 1
                
                # Accept or reject
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    s[idx] = new_val
                    current_cost += delta
                    if current_cost == 0:
                        return s
                else:
                    # Roll back sum_counts
                    for j in range(target_size):
                        if j == idx:
                            continue
                        sv = new_val + s[j]
                        c = sum_counts[sv]
                        sum_counts[sv] = c - 1
                        delta -= (c - 1) * (c - 2) // 1
                    
                    for j in range(target_size):
                        if j == idx:
                            continue
                        sv = old_val + s[j]
                        c = sum_counts.get(sv, 0)
                        sum_counts[sv] = c + 1
                        delta += c * (c + 1) // 1
                    
                T *= cooling_rate
                if T < T_min:
                    break

    return best_set
=======
import random
import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    fixed_elements = frozenset([1, 2, 5, 11])
    N = 284
    max_size = 20
    
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
        restarts = 200
        for r in range(restarts):
            if time.time() > deadline:
                break
                
            # Initialize random set: fixed elements first, then variables
            s = list(fixed_elements)
            s += random.sample(available, num_vars)
            
            # Build sum frequency dictionary
            sum_counts = {}
            for i in range(len(s)):
                for j in range(i, len(s)):
                    val = s[i] + s[j]
                    sum_counts[val] = sum_counts.get(val, 0) + 1
            
            # Compute initial cost
            current_cost = 0
            for c in sum_counts.values():
                if c > 1:
                    current_cost += c * (c - 1) // 2
            
            if current_cost == 0:
                return s
                
            T = 10.0
            T_min = 0.01
            cooling_rate = 0.999
            steps = 5000
            
            for step in range(steps):
                if time.time() > deadline:
                    break
                    
                # Pick a random variable index (indices 4..target_size-1 are variables)
                idx = random.randint(4, target_size - 1)
                
                old_val = s[idx]
                # Pick a new value from available
                new_val = random.choice(available)
                if new_val == old_val:
                    continue
                # Ensure new_val is not already in the set (distinct integers)
                if new_val in s:
                    continue
                    
                # Compute delta cost incrementally
                # When replacing old_val with new_val at position idx:
                # For each j != idx, the sum old_val+s[j] is removed and new_val+s[j] is added
                delta = 0
                for j in range(target_size):
                    if j == idx:
                        continue
                    sv = old_val + s[j]
                    c = sum_counts[sv]
                    # Removing one occurrence: cost change = -c*(c-1)/2 + (c-1)*(c-2)/2 = -(c-1)
                    delta -= (c - 1)
                    sum_counts[sv] = c - 1
                
                for j in range(target_size):
                    if j == idx:
                        continue
                    sv = new_val + s[j]
                    c = sum_counts.get(sv, 0)
                    # Adding one occurrence: cost change = (c+1)*c/2 - c*(c-1)/2 = c
                    delta += c
                    sum_counts[sv] = c + 1
                
                # Accept or reject
                if delta < 0 or random.random() < pow(2.71828, -delta / T):
                    s[idx] = new_val
                    current_cost += delta
                    if current_cost == 0:
                        return s
                else:
                    # Roll back sum_counts
                    for j in range(target_size):
                        if j == idx:
                            continue
                        sv = new_val + s[j]
                        c = sum_counts[sv]
                        sum_counts[sv] = c - 1
                    
                    for j in range(target_size):
                        if j == idx:
                            continue
                        sv = old_val + s[j]
                        c = sum_counts.get(sv, 0)
                        sum_counts[sv] = c + 1
                    
                T *= cooling_rate
                if T < T_min:
                    break

    return best_set
>>>>>>> REPLACE
