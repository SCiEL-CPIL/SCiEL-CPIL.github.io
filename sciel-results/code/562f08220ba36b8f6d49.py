import time
from math import lcm as math_lcm

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    N = 128
    # Precompute bad masks
    # bad_mask[a-1][b-1] is a bitmask of c (1..128) such that (a,b,c) is a bad triple
    bad_mask = [[0] * N for _ in range(N)]
    
    for a in range(1, N + 1):
        for b in range(1, N + 1):
            if a == b:
                continue
            mask = 0
            # Optimization: If lcm(a,b) is fixed, we can check c faster?
            # lcm(a,b) == lcm(a,c) == lcm(b,c)
            # Let L = lcm(a,b). Then L must be a multiple of a, b, c.
            # Also lcm(a,c) = L implies c divides L and lcm(a,c)=L.
            # Actually, just iterating c is fine for N=128.
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if math_lcm(a, b) == math_lcm(a, c) == math_lcm(b, c):
                    mask |= (1 << (c - 1))
            bad_mask[a - 1][b - 1] = mask
            bad_mask[b - 1][a - 1] = mask

    def calc_conflict_mask(S):
        """Calculate the union of bad masks for all pairs in S."""
        mask = 0
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                mask |= bad_mask[a - 1][b - 1]
        return mask

    def try_add(x, S, conflict_mask):
        """
        Try to add x to S.
        Returns (success, new_conflict_mask).
        """
        # Check if x conflicts with any pair in S
        if (conflict_mask >> (x - 1)) & 1:
            return False, conflict_mask
        
        # If not, calculate new conflict mask
        # New conflicts come from pairs (x, y) for y in S
        new_mask = conflict_mask
        for y in S:
            new_mask |= bad_mask[x - 1][y - 1]
        
        return True, new_mask

    def num_distinct_primes(n):
        count = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                count += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            count += 1
        return count

    best_set = set()
    best_size = 0
    
    # Generate a few different orders
    # 1. Ascending
    # 2. Descending
    # 3. By number of distinct prime factors (desc)
    # 4. Random (fixed seed for reproducibility, or just use a few fixed permutations)
    
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: (-num_distinct_primes(x), x)),
        sorted(range(1, N + 1), key=lambda x: (num_distinct_primes(x), x)),
        sorted(range(1, N + 1), key=lambda x: (x % 2, -x)), # Evens then odds desc?
    ]
    
    # Add a few random-like orders using simple permutations
    # Since we can't use random without importing, we can use a simple LCG or just shuffle manually if we import random.
    # The prompt says NumPy and SciPy are available, and standard library. random is standard library.
    import random
    
    for _ in range(5):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orders.append(random_order)

    for order in orders:
        if time.time() - start_time > deadline - 5:
            break
            
        S = set()
        conflict_mask = 0
        
        # Greedy Phase
        for x in order:
            ok, new_mask = try_add(x, S, conflict_mask)
            if ok:
                S.add(x)
                conflict_mask = new_mask
        
        # Local Search Phase
        improved = True
        while improved:
            improved = False
            if time.time() - start_time > deadline - 5:
                break
            
            # 1. Try to add any remaining element
            for x in range(1, N + 1):
                if x not in S:
                    ok, new_mask = try_add(x, S, conflict_mask)
                    if ok:
                        S.add(x)
                        conflict_mask = new_mask
                        improved = True
                        break
            
            if improved:
                continue
            
            # 2. Try 1-for-2 swaps
            # For each a in S, remove it, then try to add two new elements
            S_list = list(S)
            for a in S_list:
                if time.time() - start_time > deadline - 5:
                    break
                    
                S_temp = S - {a}
                mask_temp = calc_conflict_mask(S_temp)
                
                # Try to add first element x
                added_x = False
                for x in range(1, N + 1):
                    if x not in S_temp:
                        ok, new_mask = try_add(x, S_temp, mask_temp)
                        if ok:
                            S_temp.add(x)
                            mask_temp = new_mask
                            added_x = True
                            break
                
                if not added_x:
                    continue
                
                # Try to add second element y
                for y in range(1, N + 1):
                    if y not in S_temp:
                        ok, new_mask = try_add(y, S_temp, mask_temp)
                        if ok:
                            # Success: replace a with x, y
                            S = S_temp
                            conflict_mask = new_mask
                            improved = True
                            break
                
                if improved:
                    break
            
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    return list(best_set)
