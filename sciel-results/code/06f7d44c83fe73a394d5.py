"""Rewritten program for packing distinct multiples."""
import time
import sys
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    n = len(DIVISORS)
    
    # For each L, we need to find if there exists a window [m, m+L]
    # where we can assign distinct multiples.
    # Key structural change: use the fact that the problem is periodic.
    # The pattern of candidates repeats with period P = LCM of all divisors.
    # But P is huge, so instead, for each L, we only need to check
    # m in [0, P-1] but that's infeasible.
    
    # Better structural approach: For small L, each divisor has very few
    # candidates. Use constraint propagation with forward checking.
    # Also, note that if a solution exists for window [m, m+L], then
    # solutions exist for infinitely many m, but we just need one.
    
    # The critical insight: for a given L, the set of valid m values
    # where each divisor has at least one multiple in [m, m+L] is
    # a union of intervals. We can find these intervals efficiently.
    
    # For divisor d, m is valid iff there exists k such that
    # m <= k*d <= m+L, i.e., k*d - L <= m <= k*d.
    # So valid m for divisor d is union over k of [k*d - L, k*d].
    # The intersection over all divisors gives valid m for the window.
    
    # But we also need distinctness, which complicates things.
    
    # New structural approach: enumerate L from 1 upward.
    # For each L, use a smarter search:
    # 1. Determine the set of m values where each divisor has >= 1 candidate.
    # 2. For those m, run backtracking with strong pruning.
    
    # Even better: since L is small, just scan m but with much faster
    # feasibility checks and backtracking.
    
    # Precompute: for each divisor index i, the divisor value
    divs = DIVISORS
    
    for L in range(1, 100):
        if time.time() > deadline:
            break
        
        # For this L, find all m in a reasonable range where each divisor
        # has at least one multiple in [m, m+L].
        # The valid m for divisor d is: m in [k*d - L, k*d] for some k.
        # For m in [1, M], k ranges from ceil(1/d) to floor((M+L)/d).
        
        # Instead of scanning all m, note that the pattern repeats every
        # lcm of divisors, but that's too big. So scan m from 1 to some bound.
        # For small L, a bound of 2000 should suffice.
        
        # Use a faster inner loop:
        # For each m, quickly check if each divisor has a candidate.
        # If yes, run backtracking.
        
        # Optimization: precompute for each divisor d, the residues
        # m mod d for which there's a multiple in [m, m+L].
        # m mod d = r. The next multiple of d >= m is m + (d - r) % d.
        # This is <= m+L iff (d - r) % d <= L.
        # So valid residues for divisor d are those r where (d - r) % d <= L.
        
        # For each divisor, compute valid residues:
        valid_residues = []
        for d in divs:
            residues = set()
            for r in range(d):
                # next multiple of d >= m where m % d = r
                offset = (d - r) % d
                if offset <= L:
                    residues.add(r)
            valid_residues.append(residues)
        
        # Now scan m. For each m, check if m % d is in valid_residues[i] for all i.
        # This is a quick filter.
        
        # To speed up, we can iterate m and check residues.
        # The combined pattern has period P = lcm of all divisors, but that's huge.
        # So just scan m from 1 to a bound.
        
        m_bound = 3000
        found = False
        
        for m in range(1, m_bound + 1):
            if time.time() > deadline:
                found = True  # break out
                break
            
            # Quick feasibility: check residues
            feasible = True
            for i, d in enumerate(divs):
                if m % d not in valid_residues[i]:
                    feasible = False
                    break
            
            if not feasible:
                continue
            
            # Get candidates for each divisor
            candidates = []
            for i, d in enumerate(divs):
                k_min = (m + d - 1) // d
                k_max = (m + L) // d
                vals = tuple(k * d for k in range(k_min, k_max + 1))
                candidates.append((i, vals))
            
            # Sort by number of candidates (most constrained first)
            candidates.sort(key=lambda x: len(x[1]))
            
            # Backtracking with forward checking
            assignment = [None] * n
            used = set()
            
            # Use iterative approach to avoid recursion depth issues
            # and for speed
            def solve(idx):
                if idx == n:
                    return True
                orig_idx, vals = candidates[idx]
                for v in vals:
                    if v not in used:
                        used.add(v)
                        assignment[orig_idx] = v
                        # Forward checking: ensure remaining variables have candidates
                        feasible = True
                        for j in range(idx + 1, n):
                            oj, ov = candidates[j]
                            if not any(ovv not in used for ovv in ov):
                                feasible = False
                                break
                        if feasible:
                            if solve(idx + 1):
                                return True
                        used.remove(v)
                        assignment[orig_idx] = None
                return False
            
            if solve(0):
                result = []
                for i in range(n):
                    x = assignment[i]
                    if x is None or x % divs[i] != 0 or x < 1 or x > MAX_VAL:
                        return divs[:]
                    result.append(int(x))
                if len(set(result)) != n:
                    return divs[:]
                return result
        
        if found:
            break
    
    # Fallback
    return divs[:]
# EVOLVE-BLOCK-END
