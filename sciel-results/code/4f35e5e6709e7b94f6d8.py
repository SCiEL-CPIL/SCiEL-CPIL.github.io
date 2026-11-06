import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    N = 511
    M = N  # Number of values 1..511
    weights = [0.0] * (M + 1)  # 1-indexed for easier mapping
    for i in range(1, M + 1):
        weights[i] = 1.0 / (512 - i)
        
    # Sort indices by weight descending
    sorted_indices = sorted(range(1, M + 1), key=lambda x: weights[x], reverse=True)
    vals = sorted_indices
    
    # Precompute prime masks
    def get_prime_factors(n):
        factors = set()
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.add(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.add(n)
        return factors

    # Sieve for primes < 512
    is_prime = [True] * 512
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(512**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, 512, i):
                is_prime[j] = False
    prime_list = [i for i in range(2, 512) if is_prime[i]]
    prime_to_bit = {p: 1 << i for i, p in enumerate(prime_list)}
    
    prime_masks = [0] * (M + 1)
    for i in range(1, M + 1):
        pf = get_prime_factors(i)
        mask = 0
        for p in pf:
            if p in prime_to_bit:
                mask |= prime_to_bit[p]
        prime_masks[i] = mask
        
    # Conflict masks
    conflict_mask = [0] * M
    for i in range(M):
        v_i = vals[i]
        mask = 0
        for j in range(M):
            if i == j:
                continue
            v_j = vals[j]
            if prime_masks[v_i] & prime_masks[v_j]:
                mask |= (1 << j)
        conflict_mask[i] = mask
        
    best_sum = 0.0
    best_set = []
    
    def update_best(current_sum, selected_indices):
        nonlocal best_sum, best_set
        if current_sum > best_sum:
            best_sum = current_sum
            best_set = list(selected_indices)
            
    def sum_available(mask):
        total = 0.0
        m = mask
        while m:
            lsb = m & -m
            pos = lsb.bit_length() - 1
            total += weights[vals[pos]]
            m ^= lsb
        return total

    # Greedy initialization
    greedy_selected = []
    greedy_mask = (1 << M) - 1
    for i in range(M):
        if (greedy_mask & (1 << i)) != 0:
            greedy_selected.append(vals[i])
            greedy_mask &= ~(1 << i)
            greedy_mask &= ~conflict_mask[i]
            
    greedy_score = sum(weights[v] for v in greedy_selected)
    if greedy_score > best_sum:
        best_sum = greedy_score
        best_set = list(greedy_selected)
        
    # DFS
    def dfs(idx, current_sum, available_mask, selected):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
            
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(selected)
            return

        bound = current_sum + sum_available(available_mask)
        if bound <= best_sum:
            return
            
        if idx >= M:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = list(selected)
            return
            
        if (available_mask & (1 << idx)) == 0:
            # Exclude idx
            next_mask = available_mask & ~(1 << idx)
            dfs(idx + 1, current_sum, next_mask, selected)
            return
            
        # Include idx
        forward_mask = ((1 << M) - 1) ^ ((1 << (idx + 1)) - 1)
        next_mask_include = (available_mask & ~(1 << idx) & ~conflict_mask[idx]) & forward_mask
        
        selected.append(vals[idx])
        dfs(idx + 1, current_sum + weights[vals[idx]], next_mask_include, selected)
        selected.pop()
        
        if time.time() > deadline:
            return
            
        # Exclude idx
        next_mask_exclude = (available_mask & ~(1 << idx)) & forward_mask
        dfs(idx + 1, current_sum, next_mask_exclude, selected)

    full_mask = (1 << M) - 1
    dfs(0, 0.0, full_mask, [])
    
    return best_set
