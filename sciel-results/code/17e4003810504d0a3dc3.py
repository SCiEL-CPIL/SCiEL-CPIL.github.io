"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import factorial
from functools import lru_cache
from itertools import product

TARGET = factorial(12)  # 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11
PRIMES = [2, 3, 5, 7, 11]
MAX_EXP = [10, 5, 2, 1, 1]

def _all_divisors_gt12():
    """Generate all divisors of TARGET that are > 12, sorted ascending."""
    divs = []
    for e2 in range(11):
        for e3 in range(6):
            for e5 in range(3):
                for e7 in range(2):
                    for e11 in range(2):
                        v = (2**e2) * (3**e3) * (5**e5) * (7**e7) * (11**e11)
                        if v > 12:
                            divs.append(v)
    divs.sort()
    return divs

def _factorize_to_exp(n):
    """Return exponent tuple for n w.r.t. PRIMES, or None if not a divisor."""
    exps = []
    for i, p in enumerate(PRIMES):
        c = 0
        while n % p == 0:
            n //= p
            c += 1
        if c > MAX_EXP[i]:
            return None
        exps.append(c)
    if n != 1:
        return None
    return tuple(exps)

def _exp_tuple_to_val(exp):
    v = 1
    for e, p in zip(exp, PRIMES):
        v *= p ** e
    return v

def construct():
    start = time.time()
    deadline = start + 80.0

    # Precompute all divisors > 12
    all_divs = _all_divisors_gt12()
    n_divs = len(all_divs)

    # Fallback: single factor
    best_result = [TARGET]
    best_max = TARGET

    # We want to find a partition into distinct divisors > 12 with product TARGET,
    # minimizing the max factor.
    #
    # Strategy: binary search on the bound B. For a given B, check if there exists
    # a subset of distinct divisors from {d in all_divs : d <= B} whose product is TARGET.
    #
    # The subset product check is NP-hard in general, but our structure helps:
    # each divisor corresponds to an exponent vector, and we need the sum of exponent
    # vectors to equal MAX_EXP. This is a multi-dimensional subset sum.
    #
    # Better approach: DFS with memoization on the remaining exponent vector.
    # State: (remaining_exp_tuple, min_index_in_divs, factors_remaining)
    # But that's still large. Let's use a different approach.
    #
    # Key optimization: since we want to MINIMIZE the max factor, and we're checking
    # feasibility for bound B, we should try to use as many small factors as possible.
    #
    # Let's do: for each candidate bound (from small to large), run a DFS that tries
    # to decompose TARGET into distinct divisors <= bound. Use memoization on
    # (remaining_target, start_index, factors_used).
    #
    # But remaining_target can be large. Instead, use exponent vector as state.
    #
    # Let me try a cleaner DFS:
    # dfs(remaining_exp, start_idx, count) -> bool
    # where remaining_exp is a tuple of 5 exponents, start_idx is index into all_divs,
    # count is number of factors already used.
    #
    # We try each divisor d >= start_idx with d <= bound. If d divides remaining (in exp terms),
    # recurse with remaining_exp - exp(d), start_idx+1, count+1.
    #
    # Memoize on (remaining_exp, start_idx, count).

    # Precompute exponent vectors for all divisors
    div_exp = []
    for d in all_divs:
        e = _factorize_to_exp(d)
        if e is not None:
            div_exp.append(e)
        else:
            div_exp.append(None)

    # Binary search on bound. The bound must be a divisor of TARGET (since the max factor
    # in any valid partition is a divisor). So we search over all_divs values.
    #
    # For efficiency, we'll try bounds in increasing order and stop at the first feasible one.
    #
    # For each bound index k, the allowed divisors are all_divs[0..k].
    # We check: is there a subset of distinct indices from [0..k] whose exponent vectors
    # sum to MAX_EXP?
    #
    # DFS with memoization. State: (rem_exp_tuple, next_idx, used_count)
    # rem_exp starts at MAX_EXP, and we subtract each chosen divisor's exp.

    # To limit search: used_count <= 12, and rem_exp components must be >= 0.

    # Let's implement the feasibility check.
    def feasible(k):
        """Check if there's a subset of all_divs[0..k] with product TARGET."""
        bound = all_divs[k]
        # Allowed: indices 0..k
        # DFS state: (rem_exp, next_idx, used)
        # rem_exp is a tuple of 5 ints
        # We want rem_exp to reach (0,0,0,0,0)
        
        from functools import lru_cache
        
        @lru_cache(maxsize=None)
        def dfs(rem_exp, next_idx, used):
            if used == 0:
                # No more factors allowed; check if rem_exp is zero
                return all(r == 0 for r in rem_exp)
            if next_idx > k:
                return False
            # Pruning: if the sum of all remaining exponents (from next_idx to k) 
            # can't cover rem_exp, return False. (Expensive to compute each time; skip for now)
            # Try not taking all_divs[next_idx]
            # Try taking all_divs[next_idx]
            # We iterate: for each i from next_idx to k, try including or skipping.
            # Standard subset sum DFS:
            for i in range(next_idx, k + 1):
                e = div_exp[i]
                if e is None:
                    continue
                # Check if we can subtract e from rem_exp
                can_take = True
                new_rem = []
                for j in range(5):
                    if rem_exp[j] < e[j]:
                        can_take = False
                        break
                    new_rem.append(rem_exp[j] - e[j])
                if not can_take:
                    continue
                new_rem = tuple(new_rem)
                if dfs(new_rem, i + 1, used - 1):
                    return True
            return False
        
        return dfs(tuple(MAX_EXP), 0, 12)

    # Binary search: find smallest k such that feasible(k) is True.
    # But feasible() might be slow for large k. Let's try from small k upward.
    # Start from a reasonable lower bound.
    
    # Lower bound: with m factors each <= B, we need B^m >= TARGET (roughly).
    # With 12 factors: B >= TARGET^(1/12) ≈ 479001600^(1/12) ≈ 10.5, so B >= 11, but B > 12.
    # With 10 factors: B >= TARGET^(1/10) ≈ 22.
    # With 8 factors: B >= TARGET^(1/8) ≈ 44.
    # With 6 factors: B >= TARGET^(1/6) ≈ 115.
    # With 4 factors: B >= TARGET^(1/4) ≈ 262.
    # With 3 factors: B >= TARGET^(1/3) ≈ 782.
    # With 2 factors: B >= TARGET^(1/2) ≈ 21886.
    # With 1 factor: B = TARGET.
    
    # So the minimum possible max factor is somewhere around 22-100 range likely.
    # Let's try bounds from the smallest divisor > 12 upward.
    
    # But calling feasible(k) for small k will quickly return False (not enough divisors).
    # Let's find a reasonable starting point.
    
    # Actually, let's just iterate k from 0 upward, but skip k where there aren't
    # enough divisors. We need at least 1 divisor, so k >= 0 always.
    
    # The issue is feasible(k) could be slow for large k. Let's add a time check.
    
    for k in range(n_divs):
        if time.time() > deadline - 5.0:
            break
        
        # Quick prune: if k is very small, there aren't enough divisors to form a product
        # of TARGET. The product of the largest (k+1) divisors must be >= TARGET.
        # But that's expensive. Let's just try.
        
        # Actually, a simpler prune: the maximum product we can get with (k+1) distinct
        # divisors from all_divs[0..k] is the product of the largest min(k+1, 12) of them.
        # If that product < TARGET, skip.
        if k + 1 < 1:
            continue
        
        # Check feasibility
        result = feasible(k)
        
        if result:
            # Found a feasible bound. Now we need to actually extract the partition.
            # Let's redo the DFS to record the actual divisors.
            
            @lru_cache(maxsize=None)
            def dfs_record(rem_exp, next_idx, used):
                if used == 0:
                    if all(r == 0 for r in rem_exp):
                        return []
                    return None
                if next_idx > k:
                    return None
                for i in range(next_idx, k + 1):
                    e = div_exp[i]
                    if e is None:
                        continue
                    can_take = True
                    new_rem = []
                    for j in range(5):
                        if rem_exp[j] < e[j]:
                            can_take = False
                            break
                        new_rem.append(rem_exp[j] - e[j])
                    if not can_take:
                        continue
                    new_rem = tuple(new_rem)
                    sub = dfs_record(new_rem, i + 1, used - 1)
                    if sub is not None:
                        return [all_divs[i]] + sub
                return None
            
            partition = dfs_record(tuple(MAX_EXP), 0, 12)
            if partition is not None:
                actual_max = max(partition)
                if actual_max < best_max:
                    best_max = actual_max
                    best_result = partition
                # Since we're iterating k in increasing order, the first feasible k
                # gives the minimal possible max factor. Return immediately.
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
