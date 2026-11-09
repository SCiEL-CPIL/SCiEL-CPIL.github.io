def construct():
    import random
    from math import gcd
    from itertools import combinations
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute GCD matrix
    gcd_mat = [[1] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g
            
    # Precompute Omega(n) - total number of prime factors with multiplicity
    omega = [0] * (N + 1)
    for i in range(2, N + 1):
        if omega[i] == 0: # i is prime
            for j in range(i, N + 1, i):
                k = j
                while k % i == 0:
                    omega[j] += 1
                    k //= i
    # Note: The above loop is slightly inefficient for omega if not careful, but N=128 is tiny.
    # Let's do a simpler factorization for clarity and correctness for small N.
    omega = [0] * (N + 1)
    for i in range(2, N + 1):
        n = i
        d = 2
        while d * d <= n:
            while n % d == 0:
                omega[i] += 1
                n //= d
            d += 1
        if n > 1:
            omega[i] += 1
    omega[1] = 0

    def is_valid_set(S):
        """Check if a set of numbers has no bad triples."""
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                g_ab = gcd_mat[a][b]
                for k in range(j + 1, len(S_list)):
                    c = S_list[k]
                    if g_ab == gcd_mat[a][c] == gcd_mat[b][c]:
                        return False
        return True

    def can_add(S, x):
        """Check if x can be added to S without creating a bad triple."""
        # S is a set for O(1) lookup, but we need to iterate pairs in S
        # Since N is small, converting to list is fine, or we can keep a list parallel
        # For efficiency in local search, we might want a list.
        # Let's assume S is a set here for the initial greedy, but for local search we use lists.
        pass # Re-implementing below for list input

    def can_add_list(S_list, x):
        """Check if x can be added to S_list without creating a bad triple."""
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                a, b = S_list[i], S_list[j]
                if gcd_mat[a][b] == gcd_mat[a][x] == gcd_mat[b][x]:
                    return False
        return True

    # Strategy 1: Start with Even Omega numbers
    even_omega_nums = [n for n in nums if omega[n] % 2 == 0]
    # Verify even_omega_nums is valid
    # It should be, based on the heuristic.
    # Let's just run a greedy on all numbers, but bias towards even omega?
    # Or just start with even_omega_nums and try to add odd ones.
    
    best = []
    
    # Candidate 1: Even Omega set
    cand1 = even_omega_nums[:]
    if is_valid_set(cand1):
        best = cand1
    else:
        # Fallback: Greedy
        selected = []
        for n in nums:
            if can_add_list(selected, n):
                selected.append(n)
        best = selected

    # Candidate 2: Greedy with specific order (Primes first? No, primes are bad together. 
    # Maybe numbers with few factors first?)
    # The current best is likely close to the Even Omega set size.
    
    # Local Search Improvement
    # We will try to swap elements in `best` with elements not in `best` to increase size or maintain size while allowing further additions?
    # Actually, we want to maximize cardinality.
    # If we have a valid set of size K, we try to find a valid set of size K+1.
    
    current_best = best[:]
    improved = True
    while improved:
        improved = False
        current_set = set(current_best)
        # Try adding any missing element
        for x in nums:
            if x in current_set:
                continue
            if can_add_list(current_best, x):
                current_best.append(x)
                current_set.add(x)
                improved = True
                break # Restart scan as set changed
        
        if not improved:
            # Try swapping: remove y, add x. This keeps size same, but might allow adding z later?
            # Or just try to find a larger set by swapping?
            # A swap (remove y, add x) results in same size. 
            # To increase size, we need to add x without removing.
            # If we can't add x, maybe we can swap y for x, and then add z?
            # This is a 2-opt style move.
            # Given N=128, simple hill climbing on size is sufficient if we start good.
            
            # Let's try random restarts with the Even Omega bias if we are stuck
            # Or just break and try a different heuristic.
            break

    # Let's also run a few random restarts to ensure we didn't get stuck in a local minimum
    # of the "add only" phase.
    random.seed(42)
    for _ in range(100):
        # Shuffle nums, but keep even_omega first?
        # Or just shuffle all.
        shuffled = nums[:]
        random.shuffle(shuffled)
        selected = []
        for x in shuffled:
            if can_add_list(selected, x):
                selected.append(x)
        if len(selected) > len(current_best):
            current_best = selected
            # Try to expand this new best
            current_set = set(current_best)
            expanded = True
            while expanded:
                expanded = False
                for x in nums:
                    if x in current_set:
                        continue
                    if can_add_list(current_best, x):
                        current_best.append(x)
                        current_set.add(x)
                        expanded = True
                        break

    return current_best
