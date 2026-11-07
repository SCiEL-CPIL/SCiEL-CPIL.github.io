import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    N = 256
    candidates = list(range(2, N + 1))
    
    # 1. Identify primes
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N+1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N+1) if is_prime[i]]
    
    # Strategy: Large Primes + Small Smooth Composites
    # We keep all large primes (p > 13) because they are "independent" from small smooth numbers
    # in terms of the divisibility constraint a | b*c, provided a is smooth and b,c are large primes.
    # If a is 13-smooth and b,c are primes > 13, then gcd(a, b*c) = 1, so a cannot divide b*c.
    # Thus, we can safely take the union of:
    # 1. All primes > 13
    # 2. A valid subset of 13-smooth numbers (including small primes 2,3,5,7,11,13)
    
    small_primes = [2, 3, 5, 7, 11, 13]
    large_primes = [p for p in primes if p > 13]
    
    # Generate 13-smooth numbers
    smooth_candidates = []
    for n in range(2, N+1):
        if is_prime[n]:
            if n in small_primes:
                smooth_candidates.append(n)
        else:
            m = n
            smooth = True
            for p in small_primes:
                while m % p == 0:
                    m //= p
            if m == 1:
                smooth_candidates.append(n)
            
    # Helper: Check if adding y to set S is valid
    # S is a valid set. y not in S.
    # Condition: For all distinct a,b,c in S U {y}, a does not divide b*c.
    # This breaks down into:
    # 1. y does not divide u*v for any u,v in S, u!=v
    # 2. u does not divide y*v for any u,v in S, u!=v
    # (The case where a is in S and b,c are in S is already handled by S being valid)
    # (The case where a is in S and b=y, c in S is covered by 2. u=a, v=c, check a | y*c)
    # (The case where a is in S and b in S, c=y is covered by 2. u=a, v=b, check a | y*b)
    # (The case where a=y and b,c in S is covered by 1. u=b, v=c, check y | b*c)
    # Note: The condition "distinct" for a,b,c means we don't need to check a|a*a etc.
    
    def is_valid_addition(S, y):
        S_list = list(S)
        n = len(S_list)
        
        # 1. y does not divide u*v for any u,v in S, u!=v
        for i in range(n):
            u = S_list[i]
            for j in range(n):
                if i == j: continue
                v = S_list[j]
                if (u * v) % y == 0:
                    return False
                    
        # 2. u does not divide y*v for any u,v in S, u!=v
        for i in range(n):
            u = S_list[i]
            for j in range(n):
                if i == j: continue
                v = S_list[j]
                if (y * v) % u == 0:
                    return False
                    
        return True

    def is_valid_subset(subset):
        s = set(subset)
        items = list(s)
        n = len(items)
        for i in range(n):
            a = items[i]
            for j in range(n):
                if i == j: continue
                b = items[j]
                for k in range(n):
                    if i == k or j == k: continue
                    c = items[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Initial Greedy: Try multiple orderings
    best_small_set = []
    best_size = 0
    
    # Ordering 1: Descending value (larger numbers are "safer" to add? or "harder" to divide?)
    # Actually, if a is large, it's harder for a to divide b*c.
    # If a is small, it's easier.
    # So adding large numbers first might block fewer small numbers?
    # Or adding small numbers first might block many?
    # Let's try both.
    
    orderings = [
        sorted(smooth_candidates, reverse=True),
        sorted(smooth_candidates),
        sorted(smooth_candidates, key=lambda x: -len([d for d in range(2, x+1) if x%d==0])) # Fewest divisors first?
    ]
    
    for order in orderings:
        curr_set = []
        for n in order:
            if is_valid_addition(curr_set, n):
                curr_set.append(n)
        if len(curr_set) > best_size:
            best_size = len(curr_set)
            best_small_set = curr_set
    
    # Local Search: 2-for-1 Swaps
    # Try to replace one element x in S with two elements y, z from C (complement)
    # such that S \ {x} U {y, z} is valid.
    
    S = set(best_small_set)
    C = [x for x in smooth_candidates if x not in S]
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(S)
        found_swap = False
        
        # Iterate over elements to remove
        # Sort by some heuristic? Maybe remove "heavy" elements first?
        # For now, just iterate.
        for x in S_list:
            if time.time() > deadline:
                break
            S_minus_x = S - {x}
            
            # Find valid y in C to add to S_minus_x
            valid_ys = []
            for y in C:
                if is_valid_addition(S_minus_x, y):
                    valid_ys.append(y)
            
            # For each valid y, try to find valid z in C \ {y} to add to S_minus_x U {y}
            for y in valid_ys:
                S_prime = S_minus_x | {y}
                for z in C:
                    if z == y:
                        continue
                    if is_valid_addition(S_prime, z):
                        # Success: Swap x for y, z
                        S = S_prime | {z}
                        C = [c for c in C if c not in {y, z}]
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            if found_swap:
                break
                
    best_small_set = list(S)
    
    # Combine
    final_set = list(set(large_primes + best_small_set))
    
    # Verify final set
    if not is_valid_subset(final_set):
        # Fallback to just large primes
        final_set = large_primes
    else:
        pass

    # Ensure return type is list of ints
    return [int(x) for x in final_set]
