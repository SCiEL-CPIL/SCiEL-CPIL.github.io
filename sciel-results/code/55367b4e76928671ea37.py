import time
import random
import math

def construct(n):
    # For n <= 6, the recursive construction is optimal.
    if n <= 6:
        word = [1]
        for m in range(2, n + 1):
            perms, seen = [], set()
            for i in range(len(word) - m + 2):
                window = tuple(word[i:i + m - 1])
                if len(set(window)) == m - 1 and window not in seen:
                    seen.add(window)
                    perms.append(window)
            new = []
            for p in perms:
                block = list(p) + [m] + list(p)
                overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
                new += block[overlap:]
            word = new
        return "".join(map(str, word))

    # For n = 7, use Simulated Annealing on the permutation order
    # 1. Generate seed word
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        for i in range(len(word) - m + 2):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            overlap = next(o for o in range(min(len(new), len(block)), -1, -1) if new[len(new) - o:] == block[:o])
            new += block[overlap:]
        word = new
    
    # Extract order of permutations from seed word
    perms_in_order = []
    seen = set()
    for i in range(len(word) - n + 1):
        window = tuple(word[i:i + n])
        if len(set(window)) == n and window not in seen:
            seen.add(window)
            perms_in_order.append(window)
            
    # perms_in_order should have 5040 elements
    if len(perms_in_order) != 5040:
        # Fallback if extraction failed
        return "".join(map(str, word))

    order = perms_in_order
    N = len(order)
    
    # Precompute overlap costs? No, too big.
    # Calculate initial total length
    def get_overlap(a, b):
        # a, b are tuples of length n
        max_k = n
        for k in range(max_k, -1, -1):
            if a[n-k:] == b[:k]:
                return k
        return 0

    def calc_cost(i, j):
        # Cost of transition from i to j
        # Length added is n - overlap
        return n - get_overlap(order[i], order[j])

    # Initial total length
    total_len = n
    for i in range(N - 1):
        total_len += calc_cost(i, i + 1)
        
    best_order = order[:]
    best_len = total_len
    
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    T = 1.0
    cooling = 0.999
    
    # Simple SA loop
    # We swap adjacent elements
    # To speed up, we can iterate through random indices
    
    while time.time() - start_time < timeout:
        # Random swap
        i = random.randint(0, N - 2)
        j = i + 1
        
        # Calculate delta
        # Old costs: (i-1, i), (i, i+1), (i+1, i+2)
        # New costs: (i-1, i+1), (i+1, i), (i, i+2)
        
        # Handle boundaries
        c1 = calc_cost(i-1, i) if i > 0 else 0
        c2 = calc_cost(i, i+1)
        c3 = calc_cost(i+1, i+2) if i+2 < N else 0
        
        # Perform swap temporarily
        order[i], order[j] = order[j], order[i]
        
        nc1 = calc_cost(i-1, i) if i > 0 else 0 # Now order[i] is old j
        nc2 = calc_cost(i, i+1) # Now order[i] is old j, order[i+1] is old i
        nc3 = calc_cost(i+1, i+2) if i+2 < N else 0 # Now order[i+1] is old i
        
        # Wait, indices:
        # Original: ... A, B, C, D ...
        # Swap B, C: ... A, C, B, D ...
        # Old: cost(A,B) + cost(B,C) + cost(C,D)
        # New: cost(A,C) + cost(C,B) + cost(B,D)
        
        # My calc_cost(i, j) uses current order array.
        # So I must be careful.
        
        # Let's revert and calculate explicitly
        order[i], order[j] = order[j], order[i]
        
        # Old
        old_c1 = calc_cost(i-1, i) if i > 0 else 0
        old_c2 = calc_cost(i, i+1)
        old_c3 = calc_cost(i+1, i+2) if i+2 < N else 0
        old_sum = old_c1 + old_c2 + old_c3
        
        # Swap
        order[i], order[j] = order[j], order[i]
        
        # New
        new_c1 = calc_cost(i-1, i) if i > 0 else 0
        new_c2 = calc_cost(i, i+1)
        new_c3 = calc_cost(i+1, i+2) if i+2 < N else 0
        new_sum = new_c1 + new_c2 + new_c3
        
        delta = new_sum - old_sum
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            # Accept
            total_len += delta
            if total_len < best_len:
                best_len = total_len
                best_order = order[:]
        else:
            # Revert
            order[i], order[j] = order[j], order[i]
            
        T *= cooling
        
    # Reconstruct word from best_order
    final_word = list(best_order[0])
    for i in range(1, N):
        prev = best_order[i-1]
        curr = best_order[i]
        o = get_overlap(prev, curr)
        final_word += list(curr[o:])
        
    return "".join(map(str, final_word))
