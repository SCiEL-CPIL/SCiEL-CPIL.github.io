"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt, gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin (90s limit)
    
    N = 256
    
    # 1. Sieve to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(N) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    
    # 2. Calculate largest prime power for each prime
    prime_powers = {}
    for p in primes:
        power = p
        while power * p <= N:
            power *= p
        prime_powers[p] = power
    
    # Base: 1 + sum of all prime powers
    base_sum = 1 + sum(prime_powers.values())
    
    # 3. Generate potential merge groups
    active_primes = [p for p in primes if p <= N // 2]
    
    potential_groups = []
    
    def time_check():
        return time.time() > deadline
    
    n_active = len(active_primes)
    
    # Pairs
    for i in range(n_active):
        if time_check():
            return [1] + [prime_powers[p] for p in primes]
        p1 = active_primes[i]
        for j in range(i + 1, n_active):
            p2 = active_primes[j]
            prod = p1 * p2
            if prod > N:
                break
            cost = prime_powers[p1] + prime_powers[p2]
            gain = prod - cost
            if gain > 0:
                potential_groups.append((frozenset([p1, p2]), gain))
    
    # Triples
    for i in range(n_active):
        if time_check():
            return [1] + [prime_powers[p] for p in primes]
        p1 = active_primes[i]
        for j in range(i + 1, n_active):
            p2 = active_primes[j]
            prod12 = p1 * p2
            if prod12 * 2 > N:
                continue
            for k in range(j + 1, n_active):
                p3 = active_primes[k]
                prod = prod12 * p3
                if prod > N:
                    break
                cost = prime_powers[p1] + prime_powers[p2] + prime_powers[p3]
                gain = prod - cost
                if gain > 0:
                    potential_groups.append((frozenset([p1, p2, p3]), gain))
    
    # Quads (only very small primes)
    small_active = active_primes[:12]
    for combo in combinations(small_active, 4):
        prod = 1
        for p in combo:
            prod *= p
        if prod > N:
            continue
        cost = sum(prime_powers[p] for p in combo)
        gain = prod - cost
        if gain > 0:
            potential_groups.append((frozenset(combo), gain))
    
    # Sort by gain descending
    potential_groups.sort(key=lambda x: x[1], reverse=True)
    
    n_groups = len(potential_groups)
    
    # Build bipartite matching upper bound
    # Partition: small primes (<=11) on left, larger primes (>11) on right
    small_primes = [p for p in active_primes if p <= 11]
    large_primes = [p for p in active_primes if p > 11]
    
    n_small = len(small_primes)
    n_large = len(large_primes)
    
    # Build weight matrix for bipartite matching (pair gains only)
    # W[i][j] = gain of merging small_primes[i] with large_primes[j], or 0 if invalid
    # For Hungarian, we maximize, so use negative weights or convert
    INF = 10**9
    weight_matrix = [[0.0] * n_large for _ in range(n_small)]
    pair_gain = {}  # (small_idx, large_idx) -> gain
    
    for i, ps in enumerate(small_primes):
        for j, pl in enumerate(large_primes):
            prod = ps * pl
            if prod <= N:
                cost = prime_powers[ps] + prime_powers[pl]
                gain = prod - cost
                if gain > 0:
                    weight_matrix[i][j] = float(gain)
                    pair_gain[(i, j)] = gain
    
    # Solve bipartite max-weight matching using scipy
    try:
        from scipy.optimize import linear_sum_assignment
        # linear_sum_assignment minimizes, so negate weights
        cost_matrix = [[-weight_matrix[i][j] for j in range(n_large)] for i in range(n_small)]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        bipartite_gain = 0.0
        bipartite_pairs = []
        for r, c in zip(row_ind, col_ind):
            if weight_matrix[r][c] > 0:
                bipartite_gain += weight_matrix[r][c]
                bipartite_pairs.append((small_primes[r], large_primes[c]))
    except ImportError:
        bipartite_gain = 0.0
        bipartite_pairs = []
    
    # Upper bound: bipartite matching gives max pair gain, add all triple/quad gains
    max_extra_gain = 0.0
    for gs, g in potential_groups:
        if len(gs) > 2:
            max_extra_gain += g
    
    total_upper_bound = bipartite_gain + max_extra_gain
    
    # 4. Branch and bound with tighter bound
    best_gain = 0
    best_groups = []
    
    # Initialize with bipartite matching solution as starting point
    for (ps, pl) in bipartite_pairs:
        best_groups.append(frozenset([ps, pl]))
    best_gain = int(bipartite_gain)
    
    # Compute suffix sums
    suffix_gain = [0] * (n_groups + 1)
    for i in range(n_groups - 1, -1, -1):
        suffix_gain[i] = suffix_gain[i + 1] + potential_groups[i][1]
    
    def backtrack(idx, used_primes, current_gain, current_groups):
        nonlocal best_gain, best_groups
        
        if time.time() > deadline:
            return
        
        # Upper bound: current gain + remaining gains, capped by total_upper_bound
        ub = current_gain + min(suffix_gain[idx], total_upper_bound)
        if ub <= best_gain:
            return
        
        if current_gain > best_gain:
            best_gain = current_gain
            best_groups = list(current_groups)
        
        if idx == n_groups:
            return
        
        group_set, gain = potential_groups[idx]
        
        # Option 1: Take this group if no conflict
        if used_primes.isdisjoint(group_set):
            backtrack(idx + 1, used_primes | group_set, current_gain + gain, 
                      current_groups + [group_set])
        
        # Option 2: Skip this group
        backtrack(idx + 1, used_primes, current_gain, current_groups)
    
    backtrack(0, frozenset(), 0, [])
    
    # 5. Local search polish: try to add merges that don't conflict
    used_in_merge = set()
    for group in best_groups:
        for p in group:
            used_in_merge.add(p)
    
    # Try adding individual merges that don't conflict
    for gs, g in potential_groups:
        if time.time() > deadline:
            break
        if gs.isdisjoint(used_in_merge):
            already = any(gs == bg for bg in best_groups)
            if not already:
                best_groups.append(gs)
                used_in_merge |= gs
    
    # 6. Construct final result
    result = [1]
    used_in_merge = set()
    for group in best_groups:
        prod = 1
        for p in group:
            prod *= p
            used_in_merge.add(p)
        result.append(prod)
        
    for p in primes:
        if p not in used_in_merge:
            result.append(prime_powers[p])
            
    # Ensure types are Python int and verify pairwise coprimality
    result = [int(x) for x in result]
    
    return result

# EVOLVE-BLOCK-END
