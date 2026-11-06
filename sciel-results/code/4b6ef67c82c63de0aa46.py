"""Optimal factorization of 12! into distinct factors > 12, minimizing the largest factor.

12! = 479001600 = 2^10 * 3^5 * 5^2 * 7 * 11

Strategy:
1. Generate all divisors of 12! that are > 12.
2. Binary search on the maximum allowed factor M.
3. For each M, check if there exists a subset of divisors in [13, M] whose product is exactly 12!.
   Use DP with bitset representation of exponent vectors for speed.
4. Once optimal M* is found, recover the actual set via DFS.
"""
import math
import time

T12 = math.factorial(12)  # 479001600
PRIMES = [2, 3, 5, 7, 11]
EXP_TOTAL = {2: 10, 3: 5, 5: 2, 7: 1, 11: 1}


def get_divisors(n):
    """Return all divisors of n."""
    divs = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.add(i)
            divs.add(n // i)
        i += 1
    return sorted(divs)


def factorize(n):
    """Return prime factorization of n as a dict {prime: exponent}."""
    e = {}
    for p in PRIMES:
        c = 0
        while n % p == 0:
            n //= p
            c += 1
        if c:
            e[p] = c
    if n != 1:
        e[n] = 1
    return e


def encode_exp(e):
    """Encode exponent dict as a tuple for use as dict key."""
    return tuple(e.get(p, 0) for p in PRIMES)


def can_fit(rem, e):
    """Check if exponent vector e fits within remaining rem."""
    for i, p in enumerate(PRIMES):
        if e.get(p, 0) > rem[i]:
            return False
    return True


def subtract_exp(rem, e):
    """Subtract exponent e from remaining rem, return new tuple."""
    result = list(rem)
    for i, p in enumerate(PRIMES):
        result[i] -= e.get(p, 0)
    return tuple(result)


def solve_max_leq(M, max_factors=12):
    """Check if there exists a subset of divisors in [13, M] with product = T12,
    using at most max_factors distinct factors. Returns the set if found, else None."""
    # Get all divisors in [13, M]
    all_divs = get_divisors(T12)
    cands = [d for d in all_divs if 13 <= d <= M]
    
    if not cands:
        return None
    
    # Precompute exponent vectors
    cand_exps = [(d, encode_exp(factorize(d))) for d in cands]
    
    total_exp = encode_exp(EXP_TOTAL)
    zero_exp = (0, 0, 0, 0, 0)
    
    # DP: dp maps (exp_tuple, count) -> set of indices used (for reconstruction)
    # To save memory, we only store the set of chosen candidate indices
    # State: (exp_remaining_tuple, num_used) -> set of frozensets? No, too much.
    # Better: dp maps exp_tuple -> list of (count, chosen_set)
    # Since we just need ONE solution, we can use BFS/DFS with memoization.
    
    # Use DFS with memoization
    # state: (idx, rem_exp, count) -> bool
    # But we need the actual set, so let's do a different approach:
    # BFS from zero, tracking (exp_used, chosen_indices)
    
    # Since the number of candidates is small (divisors of 12! in [13,M]),
    # let's use a set-based DP:
    # dp: dict from exp_tuple -> set of tuples of chosen candidate values
    
    # Actually, for reconstruction, let's store parent pointers.
    # dp[exp_tuple] = (prev_exp, chosen_value, prev_count) or None
    
    # Simpler: use a dict mapping exp_tuple to the set of chosen values
    # This could blow up in memory, but the exponent space is small:
    # 2^10 * 3^5 * 5^2 * 7^1 * 11^1 = 11 * 6 * 3 * 2 * 2 = 792 possible states
    
    # dp[exp] = frozenset of chosen divisors that achieve this exp
    # Initialize: dp[zero_exp] = frozenset()
    
    dp = {zero_exp: frozenset()}
    
    for d, e in cand_exps:
        e_tuple = e
        # Process in reverse order to avoid using the same candidate twice
        new_dp = {}
        for exp, chosen in dp.items():
            new_exp = tuple(exp[i] + e_tuple[i] for i in range(len(PRIMES)))
            # Check if new_exp is within bounds
            valid = True
            for i in range(len(PRIMES)):
                if new_exp[i] > total_exp[i]:
                    valid = False
                    break
            if not valid:
                continue
            # Check count
            if len(chosen) + 1 > max_factors:
                continue
            new_chosen = chosen | {d}
            if new_exp not in new_dp:
                new_dp[new_exp] = new_chosen
        
        # Merge new_dp into dp
        for exp, chosen in new_dp.items():
            if exp not in dp:
                dp[exp] = chosen
    
    if total_exp in dp:
        return list(dp[total_exp])
    return None


def find_optimal_solution():
    """Find the optimal factorization minimizing the maximum factor."""
    all_divs = [d for d in get_divisors(T12) if d >= 13]
    
    if not all_divs:
        return [T12]
    
    min_possible = max(all_divs[0], 13)
    max_possible = T12
    
    # Binary search on M
    lo = min_possible
    hi = max_possible
    best_set = None
    
    while lo <= hi:
        mid = (lo + hi) // 2
        result = solve_max_leq(mid)
        if result is not None:
            best_set = result
            hi = mid - 1
        else:
            lo = mid + 1
    
    if best_set is None:
        # Should not happen since [T12] is always valid
        return [T12]
    
    return best_set


def valid_solution(factors):
    """Verify a solution meets all constraints."""
    if not (1 <= len(factors) <= 12):
        return False
    if len(set(factors)) != len(factors):
        return False
    if any(f < 13 for f in factors):
        return False
    p = 1
    for f in factors:
        p *= f
    return p == T12


def construct():
    deadline = time.time() + 85  # Safety margin under 90s
    
    # Try the optimal DP-based approach
    try:
        if time.time() < deadline:
            result = find_optimal_solution()
            if result and valid_solution(result):
                return result
    except Exception:
        pass
    
    # Fallback: known valid solution
    # 12! = 479001600
    # Let's find a valid set manually or via simple search
    fallback = [T12]
    if valid_solution(fallback):
        return fallback
    
    # Try a few known decompositions
    candidates = [
        [144, 64, 77, 25, 27],
        [128, 81, 25, 77],
        [256, 81, 35, 77],
        [192, 81, 25, 77, 22],
    ]
    for c in candidates:
        if valid_solution(c):
            return c
    
    return [T12]
