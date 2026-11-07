import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 256
    
    # Precompute prime factors
    def prime_factors(x):
        factors = set()
        d = 2
        while d * d <= x:
            while x % d == 0:
                factors.add(d)
                x //= d
            d += 1
        if x > 1:
            factors.add(x)
        return factors
    
    factors = [set() for _ in range(n + 1)]
    for i in range(2, n + 1):
        factors[i] = prime_factors(i)
    
    # Build bitmask
    all_primes = sorted(set(p for i in range(2, n + 1) for p in factors[i]))
    prime_to_bit = {p: i for i, p in enumerate(all_primes)}
    
    bitmask = [0] * (n + 1)
    for i in range(2, n + 1):
        mask = 0
        for p in factors[i]:
            mask |= (1 << prime_to_bit[p])
        bitmask[i] = mask

    # Candidates: 256 down to 2
    candidates = list(range(n, 1, -1))
    
    # Greedy Initial Solution
    best_sum = 1
    best_solution = [1]
    selected_mask = 0
    
    for c in candidates:
        if (bitmask[c] & selected_mask) == 0:
            best_solution.append(c)
            selected_mask |= bitmask[c]
            best_sum += c
            
    # Branch and Bound
    # To speed up, we can order candidates by value (already done).
    # We pass indices into 'candidates'.
    
    def dfs(avail_indices, current_sum, remaining_sum, current_solution):
        nonlocal best_sum, best_solution
        
        if time.time() > deadline:
            return
            
        if not avail_indices:
            if current_sum > best_sum:
                best_sum = current_sum
                best_solution = current_solution[:]
            return
            
        # Pruning
        if current_sum + remaining_sum <= best_sum:
            return
            
        i0 = avail_indices[0]
        x = candidates[i0]
        
        # Branch 1: Include x
        # Filter remaining indices
        new_avail = []
        removed_sum = 0
        # Optimization: Use list comprehension
        for j in avail_indices[1:]:
            cj = candidates[j]
            if (bitmask[x] & bitmask[cj]) == 0:
                new_avail.append(j)
            else:
                removed_sum += cj
                
        new_current_sum = current_sum + x
        new_remaining_sum = remaining_sum - x - removed_sum
        
        current_solution.append(x)
        dfs(tuple(new_avail), new_current_sum, new_remaining_sum, current_solution)
        current_solution.pop()
        
        # Branch 2: Exclude x
        dfs(tuple(avail_indices[1:]), current_sum, remaining_sum - x, current_solution)

    initial_indices = tuple(range(len(candidates)))
    initial_remaining_sum = sum(candidates)
    
    dfs(initial_indices, 1, initial_remaining_sum, [1])
    
    # Ensure types
    result = [int(x) for x in best_solution]
    return result
