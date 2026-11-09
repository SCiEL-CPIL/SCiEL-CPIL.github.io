"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from math import gcd
from functools import lru_cache

# EVOLVE-BLOCK-START
P = {'divisors': [6, 10, 14, 15, 21, 22, 26, 33, 35, 39], 'maximum': 1000000}

def _lcm(a, b):
    return a * b // gcd(a, b)

def _solve_window(L, W, divisors, max_val):
    """
    Try to assign distinct multiples of each divisor within [L, L+W].
    Uses backtracking with MRV heuristic.
    Returns list of chosen values in order of divisors, or None.
    """
    U = L + W
    if U > max_val:
        return None
    
    # Precompute candidate lists for each divisor
    candidates = []
    for d in divisors:
        # First multiple of d >= L
        start = ((L + d - 1) // d) * d
        cands = []
        v = start
        while v <= U and v <= max_val:
            cands.append(v)
            v += d
        if not cands:
            return None
        candidates.append(cands)
    
    # Order indices by MRV (fewest candidates first)
    order = list(range(len(divisors)))
    order.sort(key=lambda i: len(candidates[i]))
    
    used = set()
    assignment = [None] * len(divisors)
    
    def backtrack(idx):
        if idx == len(order):
            return True
        
        i = order[idx]
        d = divisors[i]
        
        for c in candidates[i]:
            if c not in used:
                used.add(c)
                assignment[i] = c
                if backtrack(idx + 1):
                    return True
                used.remove(c)
                assignment[i] = None
        
        return False
    
    if backtrack(0):
        return assignment
    return None

def _binary_search_window(divisors, max_val):
    """
    Find the minimum window width W such that there exists L with
    a valid assignment in [L, L+W].
    """
    # Lower bound: 0 (though distinctness means >= 1 effectively, but let's start low)
    # Upper bound: max_val
    lo = 0
    hi = max_val
    
    best_sol = None
    best_W = None
    
    # We can also do a direct search for small W first, then binary search
    # But binary search on W is valid if feasibility is monotonic?
    # If a solution exists in window W, does it exist in W+1?
    # Yes, because we can just keep the same L and same assignments.
    # So feasibility is monotonic in W.
    
    while lo <= hi:
        mid = (lo + hi) // 2
        # Check if any window of width mid is feasible
        found = False
        sol = None
        
        # We need to check if there exists L such that _solve_window(L, mid, ...) works.
        # L can range from 1 to max_val - mid.
        # This is expensive if we scan all L.
        # However, for small mid, we can scan L.
        # For large mid, the probability of success is high, so we might find it quickly.
        
        # Optimization: Only check L values that are "aligned" with the divisors?
        # Actually, L can be any integer.
        # Let's scan L from 1 upwards, but step by 1.
        # To speed up, we can limit the number of L checks per mid.
        # Or, we can observe that if a solution exists, it likely exists for small L too?
        # Not necessarily.
        
        # Alternative: For a given W, the set of valid L is the union over all assignments A of
        # [max(A) - W, min(A)].
        # This is hard to compute directly.
        
        # Let's try scanning L from 1 to some limit, or all L if W is small.
        # Since max_val is 1e6, scanning all L for each mid in binary search is too slow (20 * 1e6 * backtrack).
        
        # We need a smarter way to check feasibility for a given W.
        # Idea: For a fixed W, we want to know if there is an L.
        # This is equivalent to: Do there exist distinct multiples m_i of d_i such that
        # max(m_i) - min(m_i) <= W?
        # Let min_m = min(m_i). Then all m_i are in [min_m, min_m + W].
        # So we can iterate over possible min_m?
        # min_m must be a multiple of one of the divisors.
        
        # Let's iterate over which divisor provides the minimum.
        # For each divisor d_k, assume m_k is the minimum.
        # Then m_k must be a multiple of d_k.
        # And all other m_i must be in [m_k, m_k + W] and be multiples of d_i.
        
        # This reduces the search space for L to multiples of the divisors.
        
        for k in range(len(divisors)):
            d_k = divisors[k]
            # m_k ranges from d_k to max_val - W
            # Step by d_k
            # But we can prune: m_k + W >= max(m_i) >= d_i for all i.
            # So m_k >= max_i(d_i) - W.
            min_start = max(d_k, max(divisors) - W)
            max_end = max_val - W
            
            start = ((min_start + d_k - 1) // d_k) * d_k
            if start > max_end:
                continue
                
            # We don't need to check every multiple, just enough to find one solution.
            # Let's check a limited number of starts, or all if range is small.
            # To keep it fast, let's limit the number of L checks to 1000 per k?
            # Or just check the first few.
            
            # Actually, let's just check all multiples of d_k in the range, but break early on success.
            # The number of multiples is roughly (max_end - start) / d_k.
            # For small W, this is small. For large W, it's large, but we find a solution quickly.
            
            m_k = start
            while m_k <= max_end:
                # Try to solve with L = m_k, W = W
                # But wait, the window is [m_k, m_k + W].
                # The minimum value in the assignment must be m_k?
                # No, m_k is just the candidate for the minimum.
                # If we find a solution in [m_k, m_k+W] where the actual minimum is > m_k,
                # it will be found when we iterate to that higher minimum.
                # So we can just call _solve_window(m_k, W, ...) and check if the min of the solution is m_k?
                # No, _solve_window finds ANY solution in the window.
                # If it finds one, we are done.
                sol = _solve_window(m_k, W, divisors, max_val)
                if sol is not None:
                    # Verify that min(sol) is indeed m_k?
                    # Not strictly necessary for validity, but for the "minimum is m_k" assumption?
                    # No, we just need A valid assignment in SOME window of width W.
                    # If _solve_window(m_k, W) succeeds, we have a valid assignment.
                    return True, sol
                m_k += d_k
                
                # Safety break to prevent infinite loops or too long runs
                if m_k > max_end:
                    break
        
        # If we didn't find a solution for this W
        return False, None
    
    # This binary search structure is flawed because the inner loop is expensive.
    # Let's restructure.
    
    # Better approach:
    # 1. Iterate W from 0 upwards.
    # 2. For each W, check if a solution exists.
    # 3. To check if a solution exists for W:
    #    - Iterate over k (which divisor is the min).
    #    - Iterate over m_k (multiples of d_k).
    #    - Call _solve_window(m_k, W).
    #    - If success, return solution.
    
    # How high does W go?
    # The divisors are around 10-40.
    # The LCM is large.
    # A reasonable guess is W < 1000.
    # Let's try W from 0 to 5000.
    
    for W in range(0, 5001):
        for k in range(len(divisors)):
            d_k = divisors[k]
            min_start = max(d_k, max(divisors) - W)
            max_end = max_val - W
            
            start = ((min_start + d_k - 1) // d_k) * d_k
            if start > max_end:
                continue
                
            m_k = start
            while m_k <= max_end:
                sol = _solve_window(m_k, W, divisors, max_val)
                if sol is not None:
                    return sol
                m_k += d_k
                # Optimization: if W is large, we might find it quickly.
                # If W is small, the range of m_k is small.
                # Let's add a break if we've tried too many m_k for this W.
                # But for small W, there aren't many.
                # For large W, we find it fast.
                # So this should be efficient.
    
    # Fallback: return the divisors themselves (which is valid but not optimal)
    return divisors[:]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin
    
    divisors = P['divisors']
    max_val = P['maximum']
    
    # Try to find optimal solution
    sol = _binary_search_window(divisors, max_val)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen if _binary_search_window is fast, but just in case
        return divisors[:]
    
    # Validate
    if not isinstance(sol, list):
        return divisors[:]
    if len(sol) != len(divisors):
        return divisors[:]
    if len(set(sol)) != len(sol):
        return divisors[:]
    for v, d in zip(sol, divisors):
        if not isinstance(v, int) or v % d != 0 or v < 1 or v > max_val:
            return divisors[:]
            
    return sol
# EVOLVE-BLOCK-END
