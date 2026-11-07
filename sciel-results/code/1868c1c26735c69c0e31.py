"""Rewritten solution: maximum-sum pairwise-coprime set from 1..256."""
import math
import time

def _sieve(limit):
    """Return list of primes <= limit."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _factorize(n, primes):
    """Return sorted list of distinct prime factors of n (n >= 2)."""
    factors = []
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            factors.append(p)
            while temp % p == 0:
                temp //= p
    if temp > 1:
        factors.append(temp)
    return factors

def construct():
    start_time = time.time()
    DEADLINE = 80  # safety margin before 90s kill

    N = 256
    primes = _sieve(N)
    
    # All numbers 2..N that are not 1
    candidates = list(range(2, N + 1))
    
    # Precompute prime factorization for each candidate
    # factor_map: list of lists of prime factors
    factor_map = [None] * (N + 1)
    for i in range(2, N + 1):
        factor_map[i] = _factorize(i, primes)
    
    # Build conflict graph: conflict[a][b] = True if gcd(a,b) > 1
    # For efficiency, use a set of sets or bitsets
    # Since N=256, we can use a 2D boolean array or sets
    n_cand = len(candidates)
    cand_index = {val: idx for idx, val in enumerate(candidates)}
    
    # For each number, the set of candidates that conflict with it
    conflicts = [set() for _ in range(n_cand)]
    for i in range(n_cand):
        a = candidates[i]
        fa = factor_map[a]
        if not fa:
            continue
        for j in range(i + 1, n_cand):
            b = candidates[j]
            fb = factor_map[b]
            # Check if they share a prime factor
            shared = False
            for p in fa:
                # p is a prime factor of a; check if p divides b
                if b % p == 0:
                    shared = True
                    break
            if shared:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    # We want to select a subset S of {1} U candidates such that:
    # - 1 can always be included (it's coprime with everything)
    # - Every pair in S is coprime
    # - Sum is maximized
    
    # Since 1 is coprime with everything, we always include it.
    # So we maximize sum of selected candidates + 1.
    
    # Greedy approach: process candidates in descending order of value.
    # For each candidate, include it if it doesn't conflict with any already selected.
    
    # But greedy may not be optimal. Let's try a smarter approach.
    # 
    # Key insight: This is a maximum weight independent set in a conflict graph.
    # With N=256, the graph is dense. But we can exploit structure.
    #
    # Alternative: Since each number's conflicts are determined by its prime factors,
    # we can think of this as: for each prime p, we can select at most one number
    # divisible by p. But a number divisible by multiple primes consumes all of them.
    #
    # This is equivalent to: assign each selected number to its set of prime factors,
    # and no two selected numbers can share a prime factor.
    #
    # Greedy by value with backtracking might work. Let's try a heuristic:
    # Sort candidates by value descending. Use a priority queue / greedy with 
    # conflict checking.
    
    # First, try simple greedy descending:
    selected = set()
    selected_values = []
    for val in sorted(candidates, reverse=True):
        idx = cand_index[val]
        # Check if val conflicts with any selected
        ok = True
        for sel in selected:
            if sel in conflicts[idx]:
                ok = False
                break
        if ok:
            selected.add(val)
            selected_values.append(val)
    
    greedy_sum = 1 + sum(selected_values)
    
    # Now try to improve: local search / swap-based improvement
    # Current selection: selected (set of values)
    # Try: for each unselected number u, and each selected number s,
    #       if we remove s and add u, and the new set is still valid, do it if it improves sum.
    
    # But this is O(n^2) per iteration and may take many iterations.
    # Let's do a few passes of best-improvement.
    
    def is_valid_with(selected_set, add_val, remove_val=None):
        """Check if adding add_val and optionally removing remove_val keeps validity."""
        if remove_val is not None:
            # Remove remove_val, add add_val
            new_set = set(selected_set)
            new_set.discard(remove_val)
            new_set.add(add_val)
        else:
            new_set = set(selected_set)
            new_set.add(add_val)
        
        # Check all pairs are coprime
        new_list = list(new_set)
        for i in range(len(new_list)):
            for j in range(i + 1, len(new_list)):
                if math.gcd(new_list[i], new_list[j]) != 1:
                    return False
        return True
    
    # Better: use the conflict structure for faster checks
    def can_add(selected_set, val):
        idx = cand_index[val]
        for s in selected_set:
            sidx = cand_index[s]
            if sidx in conflicts[idx]:
                return False
        return True
    
    def can_swap(selected_set, remove_val, add_val):
        idx_add = cand_index[add_val]
        idx_rem = cand_index[remove_val]
        # After removing remove_val, add_val must not conflict with any remaining
        for s in selected_set:
            if s == remove_val:
                continue
            sidx = cand_index[s]
            if sidx in conflicts[idx_add]:
                return False
        return True
    
    # Iterative improvement
    improved = True
    max_iters = 50
    iteration = 0
    while improved and iteration < max_iters:
        if time.time() > DEADLINE:
            break
        improved = False
        iteration += 1
        
        # Try all single removals that allow adding a larger unselected number
        unselected = set(candidates) - selected
        for add_val in sorted(unselected, reverse=True):
            # For each selected value that conflicts with add_val, try removing it
            idx_add = cand_index[add_val]
            for rem_val in list(selected):
                if rem_val >= add_val:
                    # Removing a value >= add_val won't improve sum
                    continue
                if can_swap(selected, rem_val, add_val):
                    new_sum = sum(selected) - rem_val + add_val
                    if new_sum > sum(selected):
                        selected.discard(rem_val)
                        selected.add(add_val)
                        improved = True
                        break  # Restart scanning for this add_val
    
    final_sum = 1 + sum(selected)
    
    # Return the list
    result = [1] + sorted(selected)
    
    # Type check: ensure all are Python int
    assert all(isinstance(x, int) for x in result)
    assert len(result) >= 1
    # Verify pairwise coprimality
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            if math.gcd(result[i], result[j]) != 1:
                raise ValueError(f"Invalid: gcd({result[i]}, {result[j]}) != 1")
    
    return result
