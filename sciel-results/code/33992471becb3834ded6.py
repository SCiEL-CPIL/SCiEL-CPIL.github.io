"""Rewritten program: pack distinct multiples of a fixed integer set."""
# EVOLVE-BLOCK-START
import math
from itertools import combinations

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000


def _find_optimal_window():
    """
    Find the minimum window length L such that there exists a start S
    where for each divisor d_i, there is a distinct multiple m_i in [S, S+L]
    divisible by d_i.
    
    We use a greedy interval covering approach:
    For a given window length L, we check if we can assign distinct multiples.
    This is a bipartite matching problem, but we can use a greedy approach
    sorted by the 'tightness' of constraints.
    
    Actually, a simpler approach: for each divisor d, the multiples in [S, S+L]
    are ceil(S/d)*d, ceil(S/d)*d + d, ... <= S+L.
    We need to pick one from each such that all are distinct.
    
    Since the number of divisors is small (10), we can try to find a valid S
    by scanning S in a smart way. But scanning all S up to 1e6 might be slow.
    
    Alternative: Note that the LCM of all divisors is large, but the pattern of
    available multiples repeats with period LCM. However, we only need a small window.
    
    Let's think differently: The minimum possible window is at least 9 (since we need
    10 distinct integers, the span is at least 9). Let's try to find a window of
    length L starting from L=9 upwards, and for each L, try to find an S that works.
    
    For a fixed L, the condition is: there exist distinct integers x_1,...,x_10
    such that x_i is divisible by d_i and all x_i in [S, S+L].
    
    Equivalently, for each d_i, the set M_i(S) = {k*d_i : S <= k*d_i <= S+L} is non-empty,
    and we can pick one element from each M_i(S) such that all picks are distinct.
    
    Since L is small (hopefully), each M_i(S) has at most floor(L/d_i)+1 elements.
    For d_i >= 6 and L small, this is at most 2 or 3.
    
    We can use backtracking to check if a valid assignment exists for a given S and L.
    But we need to find S. 
    
    Insight: The optimal S will be such that the window [S, S+L] contains at least one
    multiple of each divisor. As S increases by 1, the sets M_i shift.
    
    Let's try a different approach: generate candidate windows.
    For each divisor d, consider its multiples. The "gaps" between multiples of d are d.
    To minimize the window, we want to align the multiples as tightly as possible.
    
    Heuristic: Try windows centered around multiples of the LCM of a subset, or just
    do a local search.
    
    Actually, let's just do a brute-force over S for small L. Since L starts at 9,
    and we can stop once we find a valid S, and the number of S to check might be bounded.
    
    But worst case, we might need to check many S. Let's limit the search.
    
    Better idea: For each divisor d_i, the multiples form an arithmetic progression.
    We can model this as: choose integers k_i such that x_i = k_i * d_i, and
    min(x_i) <= x_j <= min(x_i) + L for all j, and all x_i distinct.
    
    Let m = min(x_i). Then x_i - m is in [0, L] and x_i - m is divisible by d_i? No,
    x_i is divisible by d_i, but m may not be.
    
    Let's just use a randomized/local search approach:
    1. Start with L = 9.
    2. For each L, try random S values and check if a valid assignment exists via backtracking.
    3. If found, return the assignment.
    4. Increment L.
    
    To make it efficient, we can restrict S to be a multiple of some base, or just
    try S in a range.
    
    Actually, since the problem is small, let's do a deterministic search:
    For L from 9 to some max:
      For S in range(0, MAX_VAL - L, step=1):  # This is too slow
    
    We need a smarter way. Let's use the fact that the assignment depends on S modulo
    the LCM of the divisors. The LCM of [6,10,14,15,21,22,26,33,35,39] is:
    6=2*3, 10=2*5, 14=2*7, 15=3*5, 21=3*7, 22=2*11, 26=2*13, 33=3*11, 35=5*7, 39=3*13
    LCM = 2 * 3 * 5 * 7 * 11 * 13 = 30030.
    
    So the pattern repeats every 30030. We can just check S in [0, 30030) for each L.
    That's 30030 * (L - 9) checks, each with a backtracking of depth 10. Should be fast.
    """
    lcm_all = 30030  # Precomputed LCM of the divisors
    
    for L in range(9, 1000):  # Window length from 9 to 999
        # Try all S in one period
        for S in range(0, lcm_all):
            if S + L > MAX_VAL:
                continue
            # Check if we can assign distinct multiples
            # For each divisor, find all multiples in [S, S+L]
            candidates = []
            for d in DIVISORS:
                # First multiple >= S
                k_start = math.ceil(S / d)
                m = k_start * d
                mults = []
                while m <= S + L:
                    mults.append(m)
                    m += d
                if not mults:
                    break
                candidates.append(mults)
            else:
                # All divisors have at least one candidate
                # Now use backtracking to find distinct assignment
                assignment = [None] * len(DIVISORS)
                used = set()
                
                def backtrack(idx):
                    if idx == len(DIVISORS):
                        return True
                    for m in candidates[idx]:
                        if m not in used:
                            used.add(m)
                            assignment[idx] = m
                            if backtrack(idx + 1):
                                return True
                            used.remove(m)
                            assignment[idx] = None
                    return False
                
                if backtrack(0):
                    return assignment, L
    
    # Fallback: just return the divisors themselves (not optimal but valid)
    return DIVISORS[:], max(DIVISORS) - min(DIVISORS)


# Precompute the optimal solution
_optimal_solution = None

def _get_solution():
    global _optimal_solution
    if _optimal_solution is None:
        _optimal_solution = _find_optimal_window()
    return _optimal_solution


def construct():
    sol, _ = _get_solution()
    # Ensure we return Python ints in a list
    return [int(x) for x in sol]

# EVOLVE-BLOCK-END
