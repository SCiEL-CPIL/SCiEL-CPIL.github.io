"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from itertools import combinations

def _sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    N = 256
    primes = _sieve(N)
    
    # Split primes: small primes (<= 128) can conflict with each other and composites
    # Large primes (> 128) are > N/2, so they can't divide any number in [1, N] except themselves
    # Actually, a prime p > N/2 can only divide p itself in [1, N]. So large primes never conflict
    # with any other number in [1, N] except... wait.
    # If we pick a composite number, it has prime factors. If a prime factor is large (> 128), 
    # that composite must be at least 2*129 = 258 > 256. So no composite in [1, 256] has a prime 
    # factor > 128. Therefore, large primes are coprime to ALL other numbers in [1, 256] except 
    # themselves. So we should ALWAYS include all large primes.
    
    large_primes = [p for p in primes if p > 128]
    small_primes = [p for p in primes if p <= 128]
    
    # Now, for each composite number c in [1, 256], its prime factors are all in small_primes.
    # Two numbers are coprime iff they share no prime factors.
    # So the problem reduces to: select a subset of small_primes and composites such that no two 
    # share a prime factor, maximizing sum. Plus we always include 1 and all large_primes.
    
    # Actually, this is a maximum weight independent set problem where:
    # - Vertices: small_primes + composites in [2, 256]
    # - Edge between two vertices if they share a prime factor
    # - Weight of a vertex = its value
    # - We want max weight independent set
    
    # But wait: if we pick a prime p, we can't pick any multiple of p.
    # If we pick a composite c, we can't pick any of its prime factors, nor any other number 
    # sharing a factor with c.
    
    # Key insight: For each small prime p, we can either pick p itself, or pick some composites 
    # that are multiples of p (but not sharing factors with other chosen numbers).
    # Actually, this is complex. Let's use branch and bound.
    
    # Let's think differently. The conflict graph has vertices = all numbers in [2, 256].
    # Two vertices are connected if gcd > 1.
    # We want max weight independent set.
    
    # Number of primes <= 128: let's count. There are 30 primes <= 128.
    # The number of composites is 256 - 1 - 30 - len(large_primes) = ... 
    # Actually, let's just do branch and bound on the small primes and composites.
    
    # Simpler approach: Since large primes are always included, and 1 is always included,
    # we just need to solve for the remaining numbers: small_primes and composites in [2, 256].
    
    # Let's identify all candidate numbers: small_primes + composites in [2, 256]
    # For each candidate, compute its prime factor set (all in small_primes).
    
    candidates = []
    for x in range(2, 257):
        if x in large_primes:
            continue  # always included
        # Compute prime factors
        factors = set()
        temp = x
        for p in small_primes:
            if p * p > temp:
                break
            if temp % p == 0:
                factors.add(p)
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            factors.add(temp)
        # All factors should be in small_primes (since x <= 256)
        candidates.append((x, frozenset(factors)))
    
    # Now we have a set of candidates with their prime factor sets.
    # Two candidates conflict if their factor sets intersect.
    # We want max weight independent set.
    
    # Branch and bound:
    # Sort candidates by weight (value) descending.
    # At each step, either include the current candidate (and remove all conflicting ones) 
    # or exclude it.
    
    # For a tight upper bound: sum of weights of all remaining candidates that don't 
    # conflict with already chosen ones... but that's expensive.
    # Simpler upper bound: sum of weights of all remaining candidates.
    
    candidates.sort(key=lambda x: -x[0])
    n_cands = len(candidates)
    
    best_sum = 0
    best_set = set()
    
    # Precompute conflict matrix
    # conflict[i][j] = True if candidate i and j share a prime factor
    factor_sets = [c[1] for c in candidates]
    weights = [c[0] for c in candidates]
    
    # conflict[i] = set of j != i that conflict with i
    conflicts = []
    for i in range(n_cands):
        conf = set()
        for j in range(i + 1, n_cands):
            if factor_sets[i] & factor_sets[j]:
                conf.add(j)
        conflicts.append(conf)
    
    # Also for each i, we need to know which earlier indices conflict with i
    # For branch and bound, let's use a different approach:
    # Process candidates in order. Maintain a set of "available" candidates.
    
    # Actually, let's use a recursive branch and bound with memoization-like pruning.
    # State: (current_index, set of excluded indices due to conflicts)
    # But that's too large.
    
    # Alternative: Since the number of small primes is 30, and each composite is determined 
    # by its prime factors, maybe we can do DP over subsets of small primes? 
    # 2^30 is too large.
    
    # Let's just do branch and bound with a good heuristic ordering.
    
    # Order candidates by weight / (number of conflicts) to prioritize high-value, low-conflict items.
    order = sorted(range(n_cands), key=lambda i: -weights[i] / (len(conflicts[i]) + 1))
    
    # Reorder candidates
    new_weights = [weights[i] for i in order]
    new_factors = [factor_sets[i] for i in order]
    new_conflicts = []
    for i in order:
        conf = set()
        for j in conflicts[i]:
            # Find position of j in order
            pos = order.index(j)
            conf.add(pos)
        new_conflicts.append(conf)
    
    weights = new_weights
    factor_sets = new_factors
    conflicts = new_conflicts
    
    # Branch and bound
    best_sum = [0]
    best_selection = [set()]
    
    def branch(idx, current_sum, selected, excluded):
        if time.time() > deadline:
            return
        
        if idx == n_cands:
            if current_sum > best_sum[0]:
                best_sum[0] = current_sum
                best_selection[0] = set(selected)
            return
        
        # Upper bound: current_sum + sum of weights of all remaining candidates
        # that are not excluded
        ub = current_sum
        for j in range(idx, n_cands):
            if j not in excluded:
                ub += weights[j]
        
        if ub <= best_sum[0]:
            return
        
        # Option 1: Exclude candidate idx
        branch(idx + 1, current_sum, selected, excluded)
        
        # Option 2: Include candidate idx (if not excluded)
        if idx not in excluded:
            new_excluded = set(excluded)
            new_excluded.add(idx)
            # Add all conflicts of idx to excluded
            for j in conflicts[idx]:
                if j > idx:  # only future candidates matter
                    new_excluded.add(j)
            
            selected.append(idx)
            branch(idx + 1, current_sum + weights[idx], selected, new_excluded)
            selected.pop()
    
    # This might be too slow. Let's try a greedy approach first, then refine.
    
    # Greedy: pick the candidate with highest weight that doesn't conflict with already picked.
    selected = set()
    greedy_sum = 0
    available = set(range(n_cands))
    
    while available:
        # Pick the one with highest weight
        best = max(available, key=lambda i: weights[i])
        selected.add(best)
        greedy_sum += weights[best]
        # Remove best and all its conflicts
        to_remove = {best} | conflicts[best]
        available -= to_remove
    
    best_sum[0] = greedy_sum
    best_selection[0] = selected
    
    # Now try to improve with local search or branch and bound with tighter bounds.
    # For now, let's just use the greedy result.
    
    # Build the final set
    result = [1] + large_primes[:]
    for i in best_selection[0]:
        result.append(weights[i])
    
    # Verify: check that all pairs are coprime
    result = list(set(result))  # remove duplicates
    result.sort()
    
    # Quick verification
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                # This shouldn't happen, but if it does, fall back to primes only
                return [1] + primes
    
    return result

# EVOLVE-BLOCK-END
