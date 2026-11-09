"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
N = len(DIVISORS)
MAX_VAL = 1000000
DEADLINE_MARGIN = 5.0  # seconds before the 90s wall-clock limit

def _lcm(a, b):
    return a * b // gcd(a, b)

def _all_lcm():
    return reduce(_lcm, DIVISORS)

L = _all_lcm()

def _check_window(L_start, W):
    """
    Check if there exists a window [L_start, L_start + W] such that for each divisor d_i,
    there is a multiple of d_i in the window, and all chosen multiples are distinct.
    
    Returns True if such a valid packing exists, else False.
    Uses backtracking with constraint propagation.
    """
    # For each divisor, find all multiples in [L_start, L_start + W]
    candidates = []
    for d in DIVISORS:
        first = ((L_start + d - 1) // d) * d
        if first > L_start + W:
            return False
        mults = []
        m = first
        while m <= L_start + W:
            mults.append(m)
            m += d
        candidates.append(mults)
    
    # Backtracking search for distinct assignments
    assigned = [False] * (W + 1)
    order = list(range(N))
    
    # Sort divisors by number of candidates (most constrained first)
    order.sort(key=lambda i: len(candidates[i]))
    
    def backtrack(idx):
        if idx == N:
            return True
        i = order[idx]
        for m in candidates[i]:
            offset = m - L_start
            if not assigned[offset]:
                assigned[offset] = True
                if backtrack(idx + 1):
                    return True
                assigned[offset] = True  # undo
        return False
    
    return backtrack(0)

def _find_optimal_window():
    """
    Search for the minimum window width W such that there exists a starting position L_start
    where all divisors have a multiple in [L_start, L_start + W] with distinct values.
    """
    start_time = time.time()
    
    # W must be at least N-1 = 9 since we need N distinct integers
    min_W = N - 1
    
    # Upper bound: the divisors themselves fit in [0, max(DIVISORS)] if we start at 0,
    # but they must be distinct multiples. Actually, the divisors are already distinct,
    # so W <= max(DIVISORS) - min(DIVISORS) = 39 - 6 = 33 is a valid upper bound.
    # But we can do better. Let's try from min_W upward.
    
    # Periodicity: if a window at position L works, then L + k*L also works for any k.
    # So we only need to check L_start in [0, L-1].
    
    for W in range(min_W, 200):  # Try small windows first
        # Check all possible starting positions modulo L
        # But L is huge (lcm of all divisors). Instead, we can reason:
        # A window [s, s+W] works iff for each d_i, there's a multiple of d_i in [s, s+W].
        # This is a system of congruence-like constraints.
        
        # For efficiency, note that the pattern of which multiples fall in the window
        # repeats with period lcm of the divisors involved. But since L is huge,
        # let's use a different approach: iterate over possible s values that are "interesting".
        
        # Key insight: the critical values of s are those where some multiple of some d_i
        # enters or leaves the window. These are s = k*d_i - W and s = k*d_i for various k.
        # But this is complex. Instead, let's just try s in a smart range.
        
        # Actually, let's use the fact that if we shift the window, the relative positions
        # of multiples change periodically. Let's just brute force s in [0, max_div*2] or so,
        # since the structure repeats with period L but L is too large.
        
        # Better approach: for a given W, check if there exists s such that all constraints are met.
        # This is a hitting set / interval covering problem.
        
        # Let's try a direct approach: for each divisor, the set of valid s values is a union of intervals.
        # For divisor d, multiples are at ..., m, m+d, m+2d, ...
        # A multiple k*d is in [s, s+W] iff s <= k*d <= s+W, i.e., k*d - W <= s <= k*d.
        # So for each d, valid s is union over k of [k*d - W, k*d].
        
        # We need s that is in the intersection over all d of these unions.
        # Let's collect all interval endpoints and check candidate s values.
        
        # Collect all critical s values: for each d, for each k such that [k*d - W, k*d] might be relevant
        # Since s >= 0 (we can assume w.l.o.g. by periodicity), k*d - W >= 0 => k >= W/d
        # and k*d <= s_max. What's s_max? We can bound it.
        
        # Actually, by periodicity with period L, we only need to check s in [0, L-1], but L is huge.
        # However, the structure of the union of intervals for each d has period d, and the combined
        # structure has period L. But for small W, maybe we can just check s in [0, max_div*2].
        
        # Let's just try s from 0 to max_div*3, which should be enough for small W.
        max_d = max(DIVISORS)
        
        for s in range(0, max_d * 3 + W + 1):
            if _check_window(s, W):
                return W, s
    
    # If we get here, try larger W with more s values
    for W in range(200, 500):
        max_d = max(DIVISORS)
        for s in range(0, max_d * 3 + W + 1):
            if _check_window(s, W):
                return W, s
    
    return None, None

def _get_solution(W, s):
    """
    Given a valid window width W and start s, find distinct multiples for each divisor.
    Uses backtracking to assign multiples.
    """
    candidates = []
    for d in DIVISORS:
        first = ((s + d - 1) // d) * d
        mults = []
        m = first
        while m <= s + W:
            mults.append(m)
            m += d
        candidates.append(mults)
    
    order = list(range(N))
    order.sort(key=lambda i: len(candidates[i]))
    
    assigned = [False] * (W + 1)
    assignment = [None] * N
    
    def backtrack(idx):
        if idx == N:
            return True
        i = order[idx]
        for m in candidates[i]:
            offset = m - s
            if not assigned[offset]:
                assigned[offset] = True
                assignment[i] = m
                if backtrack(idx + 1):
                    return True
                assigned[offset] = False
                assignment[i] = None
        return False
    
    backtrack(0)
    
    # Convert to Python ints
    return [int(x) for x in assignment]

def construct():
    start_time = time.time()
    
    # Find optimal window
    W, s = _find_optimal_window()
    
    if W is None:
        # Fallback: just use the divisors themselves
        return [int(d) for d in DIVISORS]
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > 60.0 - DEADLINE_MARGIN:
        # Try to get a solution quickly
        try:
            sol = _get_solution(W, s)
            if all(isinstance(x, int) and not isinstance(x, bool) for x in sol):
                return sol
        except:
            pass
        return [int(d) for d in DIVISORS]
    
    sol = _get_solution(W, s)
    
    # Verify
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in sol):
        return [int(d) for d in DIVISORS]
    
    if len(set(sol)) != N:
        return [int(d) for d in DIVISORS]
    
    for i, d in enumerate(DIVISORS):
        if sol[i] % d != 0 or sol[i] <= 0 or sol[i] > MAX_VAL:
            return [int(d) for d in DIVISORS]
    
    return sol
# EVOLVE-BLOCK-END
