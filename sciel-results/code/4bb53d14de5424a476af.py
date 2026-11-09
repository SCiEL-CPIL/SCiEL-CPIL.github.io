"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = P['n']
    candidates = list(range(2, n + 1))
    
    # Precompute prime factorizations for all numbers
    def prime_factorize(num):
        factors = {}
        d = 2
        while d * d <= num:
            while num % d == 0:
                factors[d] = factors.get(d, 0) + 1
                num //= d
            d += 1
        if num > 1:
            factors[num] = factors.get(num, 0) + 1
        return factors
    
    factorizations = {}
    for i in range(2, n + 1):
        factorizations[i] = prime_factorize(i)
    
    # Precompute divides_product table for all triples (a, b, c) where a|b*c
    # Using a 2D array: dp[a-2][b-2] = set of c values where a | b*c
    # Actually, let's use a different approach: precompute for each pair (a,b) 
    # whether a divides b*c for each c
    # dp[a][b] = bitmask of c values (0-indexed in candidates) where a | b*c
    # But that's 255*255*255 bits which is too much.
    
    # Better: precompute for each a, the set of (b,c) pairs where a|b*c
    # Use a list of sets: conflicts[a] = set of frozenset({b,c}) where a|b*c
    # Actually for can_add, we need: for new_elem x, check if any a in S conflicts with x
    # a conflicts with x if there exists b in S (b != a, b != x) such that a|x*b or x|a*b
    
    # Let me precompute: for each pair (a, b), the set of c such that a|b*c
    # div_table[a][b] = set of c where a | b*c
    # Index by value-2 (0-indexed)
    size = n - 1  # indices 0..254 for values 2..256
    
    # Precompute div_table: div_table[a_idx][b_idx] = bitmask of c_idx where a|b*c
    # Use Python int as bitmask (255 bits)
    div_table = [[0] * size for _ in range(size)]
    
    for a in range(2, n + 1):
        a_idx = a - 2
        for b in range(2, n + 1):
            b_idx = b - 2
            mask = 0
            for c in range(2, n + 1):
                if c == a or c == b:
                    continue
                c_idx = c - 2
                # Check if a divides b*c
                if (b * c) % a == 0:
                    mask |= (1 << c_idx)
            div_table[a_idx][b_idx] = mask
    
    def can_add_fast(S_set, new_elem):
        """Fast check using precomputed table. S_set is a set of values."""
        if not S_set:
            return True
        new_idx = new_elem - 2
        S_list = list(S_set)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            a_idx = a - 2
            for j in range(m):
                if j == i:
                    continue
                b = S_list[j]
                b_idx = b - 2
                # Check a | new_elem * b: need new_elem in div_table[a][b]
                if div_table[a_idx][b_idx] & (1 << new_idx):
                    return False
                # Check new_elem | a * b: need b in div_table[new_elem][a]
                if div_table[new_idx][a_idx] & (1 << b_idx):
                    return False
        return True
    
    def is_valid_set_fast(S):
        """Check validity using precomputed table"""
        S_list = list(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            a_idx = a - 2
            for j in range(m):
                if j == i:
                    continue
                b = S_list[j]
                b_idx = b - 2
                for k in range(m):
                    if k == i or k == j:
                        continue
                    c = S_list[k]
                    c_idx = c - 2
                    if div_table[a_idx][b_idx] & (1 << c_idx):
                        return False
        return True
    
    # Compute conflict count efficiently using div_table
    conflict_count = {}
    for a in range(2, n + 1):
        a_idx = a - 2
        cnt = 0
        for b in range(2, n + 1):
            if b == a:
                continue
            b_idx = b - 2
            mask = div_table[a_idx][b_idx]
            # Count bits in mask, but exclude b itself (already excluded in precompute)
            cnt += bin(mask).count('1')
        conflict_count[a] = cnt
    
    if time.time() > deadline - 15:
        # Fallback to primes
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        if n >= 1:
            is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    best_set = []
    
    # Strategy 1: Sort by conflict count (ascending)
    sorted_by_conflict = sorted(candidates, key=lambda x: conflict_count[x])
    selected = set()
    for a in sorted_by_conflict:
        if time.time() > deadline - 10:
            break
        if can_add_fast(selected, a):
            selected.add(a)
    if len(selected) > len(best_set):
        best_set = list(selected)
    
    # Strategy 2: Sort by number of prime factors (ascending)
    sorted_by_pf = sorted(candidates, key=lambda x: sum(factorizations[x].values()))
    selected = set()
    for a in sorted_by_pf:
        if time.time() > deadline - 10:
            break
        if can_add_fast(selected, a):
            selected.add(a)
    if len(selected) > len(best_set):
        best_set = list(selected)
    
    # Strategy 3: Sort by value descending
    selected = set()
    for a in sorted(candidates, reverse=True):
        if time.time() > deadline - 10:
            break
        if can_add_fast(selected, a):
            selected.add(a)
    if len(selected) > len(best_set):
        best_set = list(selected)
    
    # Strategy 4: Sort by value ascending
    selected = set()
    for a in sorted(candidates):
        if time.time() > deadline - 10:
            break
        if can_add_fast(selected, a):
            selected.add(a)
    if len(selected) > len(best_set):
        best_set = list(selected)
    
    # Strategy 5-10: Randomized restarts with perturbation
    random.seed(42)
    conflict_values = [conflict_count[x] for x in candidates]
    median_conflict = sorted(conflict_values)[len(conflict_values) // 2]
    for restart in range(6):
        if time.time() > deadline - 8:
            break
        # Create a randomized ordering with some structure
        order = list(candidates)
        if restart == 0:
            # Mostly by conflict count with random tie-breaking
            order.sort(key=lambda x: (conflict_count[x], random.random()))
        elif restart == 1:
            # Mostly by prime factor count
            order.sort(key=lambda x: (sum(factorizations[x].values()), random.random()))
        elif restart == 2:
            # Random
            random.shuffle(order)
        elif restart == 3:
            # Descending value with random perturbation
            order.sort(key=lambda x: (-x + random.random() * 10, random.random()))
        elif restart == 4:
            # Ascending value with random perturbation
            order.sort(key=lambda x: (x + random.random() * 10, random.random()))
        else:
            # Mix: half by conflict, half random
            low_conflict = [x for x in order if conflict_count[x] <= median_conflict]
            high_conflict = [x for x in order if conflict_count[x] > median_conflict]
            random.shuffle(low_conflict)
            random.shuffle(high_conflict)
            order = low_conflict + high_conflict
        
        selected = set()
        for a in order:
            if time.time() > deadline - 8:
                break
            if can_add_fast(selected, a):
                selected.add(a)
        if len(selected) > len(best_set):
            best_set = list(selected)
    
    # Local improvement: try swapping elements
    best_set_set = set(best_set)
    improved = True
    while improved:
        if time.time() > deadline - 4:
            break
        improved = False
        for remove_elem in list(best_set_set):
            if time.time() > deadline - 4:
                break
            candidate_set = best_set_set - {remove_elem}
            for add_elem in candidates:
                if add_elem in candidate_set:
                    continue
                if can_add_fast(candidate_set, add_elem):
                    new_set = candidate_set | {add_elem}
                    if is_valid_set_fast(new_set):
                        best_set_set = new_set
                        improved = True
                        break
    
    best_set = list(best_set_set)
    
    # Verify the result
    if not is_valid_set_fast(best_set):
        # Fallback to primes
        is_prime = [True] * (n + 1)
        is_prime[0] = False
        if n >= 1:
            is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i in range(2, n + 1) if is_prime[i]]
    
    # Ensure all elements are Python ints
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
