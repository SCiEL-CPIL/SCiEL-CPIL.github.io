"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import heapq
from itertools import combinations

DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000

def _next_multiple(n, d):
    """Return the smallest multiple of d that is >= n."""
    if n % d == 0:
        return n
    return ((n // d) + 1) * d

def _try_window(L, W):
    """
    Try to find a valid assignment of 10 distinct multiples within [L, L+W].
    Returns a list of 10 ints (one per divisor) if successful, else None.
    Uses backtracking with forward checking.
    """
    # Precompute for each divisor d and each candidate value v in [L, L+W]:
    # whether v is a multiple of d.
    # To save memory/time, we just check on the fly.
    
    # We'll use a simple backtracking search.
    # Order divisors by number of available multiples in the window (ascending) to prune early.
    # But computing that for every window is expensive. Let's just use a fixed order or sort once.
    # Actually, the number of multiples of d in [L, L+W] is roughly (W+1)/d.
    # So larger divisors have fewer options. Let's sort divisors by descending value (fewer options first).
    
    # We need to map back to the original order for the result.
    # indices: 0..9 corresponding to DIVISORS
    # Let's create a list of (divisor, index) and sort by divisor descending.
    order = sorted(range(10), key=lambda i: -DIVISORS[i])
    
    assigned = [None] * 10
    used_values = set()
    
    def backtrack(k):
        if k == 10:
            return True
        
        idx = order[k]
        d = DIVISORS[idx]
        
        # Find all multiples of d in [L, L+W] that are not in used_values
        start = _next_multiple(L, d)
        end = L + W
        
        # Iterate through candidates
        # To improve pruning, we might want to try values that are more "central" or just in order.
        # In order is fine.
        
        for v in range(start, end + 1, d):
            if v in used_values:
                continue
            
            # Forward checking: for all remaining divisors, is there at least one available multiple?
            # This can be expensive. Let's try without it first, or with a lightweight check.
            # Given the small size (10 items), simple backtracking might be fast enough if the window is tight.
            
            assigned[idx] = v
            used_values.add(v)
            
            if backtrack(k + 1):
                return True
            
            assigned[idx] = None
            used_values.remove(v)
        
        return False
    
    if backtrack(0):
        return assigned
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_result = None
    best_span = float('inf')
    
    # Strategy:
    # 1. Find a good initial window using a greedy or random approach to get a feasible solution quickly.
    # 2. Then search for smaller windows.
    
    # First, let's find a feasible solution with a reasonably small window.
    # We can estimate the minimum possible span.
    # The divisors are [6, 10, 14, 15, 21, 22, 26, 33, 35, 39].
    # LCM of these is large.
    # Let's just try a range of window sizes.
    
    # We'll iterate window size W from small to large.
    # For each W, we'll try several starting positions L.
    
    # To be efficient, we can sample L values.
    # Or, we can use a smarter search.
    
    # Let's try a specific heuristic:
    # The values must be distinct multiples.
    # Let's try to cluster them around a common base.
    
    # Alternative approach:
    # Generate a pool of candidate values.
    # For each divisor d, generate multiples.
    # We want to pick 10 values (one for each d) that are distinct and close to each other.
    
    # Let's try a random restart hill climbing or local search.
    # But backtracking on a window is deterministic and fast if the window is small.
    
    # Let's start with W = 100 and increase if no solution found.
    # Actually, the previous best was around 20. Let's start small.
    
    # We'll search W from 1 upwards.
    # For each W, we'll try a few L values.
    # Which L values?
    # If we have a solution for W, the values are in [L, L+W].
    # We can try L values that are multiples of the LCM of the divisors? No, that's huge.
    # We can try L values modulo something.
    
    # Let's just try a set of L values.
    # Since the divisors are small, the pattern of multiples repeats every LCM.
    # But LCM is huge.
    
    # Let's try a different approach:
    # Randomly pick a center C.
    # For each divisor d, pick a multiple of d close to C.
    # Ensure distinctness.
    # This is a local search.
    
    # Let's combine:
    # 1. Try small W with specific L values derived from the divisors.
    # 2. If that fails, use local search.
    
    # For now, let's stick to the window search but optimize it.
    # We'll try W from 1 to 1000.
    # For each W, we'll try L in a set of candidate starts.
    # Candidate starts: multiples of the divisors, or values near multiples.
    
    # Actually, a simpler and effective method:
    # Iterate L from 1 to some limit.
    # For each L, find the minimal W such that a solution exists in [L, L+W].
    # This is hard to do directly.
    
    # Let's go back to:
    # For W in range(1, 1000):
    #   For L in candidate_starts:
    #     if _try_window(L, W):
    #       return result
    #   if time.time() > deadline:
    #       break
    
    # What are good candidate_starts?
    # If we have a solution, min(value) is one of the values.
    # So L can be any of the values.
    # But we don't know the values.
    
    # Let's try a heuristic:
    # The values are multiples of the divisors.
    # So the values are in the set S = {k*d | k>=1, d in DIVISORS}.
    # We want to find 10 distinct values in S, one for each d, minimizing max-min.
    
    # Let's generate a sorted list of all possible values up to some limit (e.g., 100000).
    # Then use a sliding window over this list?
    # No, because each value must be assigned to a specific divisor.
    
    # Let's try this:
    # Generate candidates for each divisor.
    # For d in DIVISORS:
    #   candidates[d] = [d, 2d, 3d, ...] up to some limit.
    
    # Then, we want to pick v_d from candidates[d] such that all v_d are distinct and max(v_d) - min(v_d) is minimized.
    
    # This is a constraint satisfaction problem.
    # We can use a branch and bound search.
    # Order divisors by number of candidates (ascending) or by value.
    
    # Let's implement a branch and bound search.
    # State: (k, current_min, current_max, assigned_values)
    # k: index of next divisor to assign (in a fixed order).
    # assigned_values: set of used values.
    
    # To prune:
    # 1. If current_max - current_min >= best_span, prune.
    # 2. If even the smallest possible remaining values cannot fit within best_span, prune.
    
    # Let's define the order of divisors.
    # Sort by divisor value descending (fewer options first).
    
    # Precompute candidates for each divisor up to a limit.
    # Limit: If the best span is S, and the min value is M, then max value is M+S.
    # So we only need candidates up to M+S.
    # But M is unknown.
    # We can set a global limit, e.g., 10000.
    # If the optimal solution is beyond 10000, we might miss it, but it's unlikely for small divisors.
    
    LIMIT = 20000
    candidates = {}
    for d in DIVISORS:
        candidates[d] = list(range(d, LIMIT + 1, d))
    
    # Sort divisors by descending value
    order = sorted(range(10), key=lambda i: -DIVISORS[i])
    
    # Branch and bound
    # We'll use a recursive function with memoization or just recursion.
    # Since the depth is only 10, recursion is fine.
    
    best_span = float('inf')
    best_assignment = None
    
    # To speed up, we can use a priority queue (best-first search) instead of depth-first.
    # State: (span, k, assigned_values_tuple)
    # But assigned_values is a set, which is not hashable. We can use a frozenset or a tuple of sorted values.
    # However, the number of states might be large.
    
    # Let's stick to depth-first with pruning.
    
    def search(k, current_min, current_max, assigned):
        nonlocal best_span, best_assignment
        
        if k == 10:
            span = current_max - current_min
            if span < best_span:
                best_span = span
                best_assignment = assigned[:]
            return
        
        idx = order[k]
        d = DIVISORS[idx]
        
        # Pruning: if current_max - current_min >= best_span, no need to continue
        if current_max - current_min >= best_span:
            return
        
        # Try each candidate for this divisor
        # We can limit the candidates to those that don't immediately violate the span bound.
        # Also, we can sort candidates by value to find tight clusters early.
        
        for v in candidates[d]:
            # If v is already assigned, skip
            if v in assigned:
                continue
            
            new_min = min(current_min, v)
            new_max = max(current_max, v)
            
            # Pruning: if the new span is already >= best_span, skip
            if new_max - new_min >= best_span:
                # Since candidates are sorted, if v is too large, we can break?
                # Not necessarily, because a smaller v might have been skipped? No, we iterate in order.
                # If v is increasing, and new_max - new_min >= best_span, then for larger v, new_max will be even larger.
                # But new_min might decrease? No, v is increasing, so new_min stays the same or increases.
                # So if new_max - new_min >= best_span, then for all larger v, the span will be >= best_span.
                # So we can break.
                break
            
            assigned[idx] = v
            search(k + 1, new_min, new_max, assigned)
            assigned[idx] = None
    
    # Initial call
    # current_min = infinity, current_max = -infinity
    # But we need to handle the first assignment carefully.
    # Let's modify the search to handle the first step.
    
    def search_init():
        nonlocal best_span, best_assignment
        
        # Try each candidate for the first divisor
        idx = order[0]
        d = DIVISORS[idx]
        
        for v in candidates[d]:
            assigned = [None] * 10
            assigned[idx] = v
            search(1, v, v, assigned)
            
            # Pruning: if the span for the first element is already large? No, span is 0.
            # But if v is very large, and we know the other values must be close, we might prune.
            # For now, just try all.
            
            if time.time() > deadline:
                return
    
    search_init()
    
    if best_assignment is not None:
        # Convert to list of ints in the original order
        result = [int(best_assignment[i]) for i in range(10)]
        return result
    else:
        # Fallback: return a valid but possibly suboptimal solution
        # Just pick the smallest multiple for each divisor
        # This might not be distinct, so we need to handle that.
        # For safety, return the divisors themselves (which are distinct? 6,10,14,15,21,22,26,33,35,39 are distinct)
        return DIVISORS[:]

# EVOLVE-BLOCK-END
