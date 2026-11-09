import time
import sys
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START
DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
MAX_VAL = 1000000
DEADLINE = 85.0  # Seconds, with safety margin

def _lcm(a, b):
    return a * b // gcd(a, b)

def _all_lcm():
    return reduce(_lcm, DIVISORS)

def _greedy_assign(divisors, L, W):
    """
    Greedy assignment: for each divisor (in order of fewest candidates),
    pick the smallest available multiple in [L, L+W].
    Returns list of values or None if greedy fails.
    """
    n = len(divisors)
    candidates = []
    for d in divisors:
        start_k = (L + d - 1) // d
        end_k = (L + W) // d
        cands = []
        for k in range(start_k, end_k + 1):
            val = k * d
            if val <= MAX_VAL:
                cands.append(val)
        candidates.append(cands)
    
    # Order by number of candidates (most constrained first)
    order = sorted(range(n), key=lambda i: len(candidates[i]))
    
    assigned = [None] * n
    used = set()
    
    for idx in order:
        found = False
        for val in candidates[idx]:
            if val not in used:
                assigned[idx] = val
                used.add(val)
                found = True
                break
        if not found:
            return None
    return assigned

def _backtrack_assign(divisors, L, W):
    """
    Full backtracking with MRV heuristic.
    Returns list of values or None.
    """
    n = len(divisors)
    candidates = []
    for d in divisors:
        start_k = (L + d - 1) // d
        end_k = (L + W) // d
        cands = []
        for k in range(start_k, end_k + 1):
            val = k * d
            if val <= MAX_VAL:
                cands.append(val)
        candidates.append(cands)
    
    # Precompute conflict structure: which divisors share multiples
    # For backtracking, use MRV dynamically
    
    assigned = [None] * n
    used = set()
    
    def dfs(pos):
        if pos == n:
            return True
        
        # MRV: pick unassigned variable with fewest available candidates
        best_idx = -1
        best_count = float('inf')
        for i in range(n):
            if assigned[i] is None:
                avail = sum(1 for v in candidates[i] if v not in used)
                if avail < best_count:
                    best_count = avail
                    best_idx = i
                if avail == 0:
                    return False
        
        if best_idx == -1:
            return True  # all assigned
        
        for val in candidates[best_idx]:
            if val not in used:
                used.add(val)
                assigned[best_idx] = val
                if dfs(pos + 1):
                    return True
                used.remove(val)
                assigned[best_idx] = None
        return False
    
    if dfs(0):
        return assigned
    return None

def construct():
    start_time = time.time()
    lcm_period = _all_lcm()
    
    # Strategy: for each width W from 9 upward,
    # check all residue classes r mod lcm_period for L = r + k*lcm_period
    # But lcm is huge, so instead:
    # For small W, the window [L, L+W] intersects at most W+1 residue positions.
    # Key insight: the relative positions of multiples within a window depend on L mod lcm.
    # But lcm is ~3.6e8, too large to enumerate.
    
    # Alternative structural approach:
    # Use the fact that for width W, we need 10 distinct values in an interval of length W.
    # The minimum possible width is 9. Try W = 9, 10, 11, ... 
    # For each W, search over L efficiently.
    
    # Efficient search: For a given W, the assignment feasibility depends on L mod g 
    # where g relates to the divisors' gcd structure. But more practically:
    # For small W, each divisor has very few candidates (1 or 2).
    # We can enumerate L in a smart way.
    
    # For width W, divisor d has candidates at positions:
    # L + offset where offset makes L+offset divisible by d
    # The number of candidates is floor((L+W)/d) - ceil(L/d) + 1
    
    # Let's try a different structural approach:
    # Model as exact cover / bipartite matching for each window.
    # For a given L and W, build bipartite graph: divisors on left, values in [L,L+W] on right.
    # Edge if value is divisible by divisor. Check if perfect matching exists.
    
    # But values in [L, L+W] are at most W+1 values, and we have 10 divisors.
    # Use Hopcroft-Karp or simple DFS matching.
    
    def _has_perfect_matching(divisors, L, W):
        """Check if there's a perfect matching from divisors to distinct values in [L, L+W]."""
        n = len(divisors)
        # Get all values in [L, L+W]
        values = list(range(L, L + W + 1))
        
        # For each divisor, find which values it can match to
        adj = [[] for _ in range(n)]
        for i, d in enumerate(divisors):
            for j, v in enumerate(values):
                if v % d == 0:
                    adj[i].append(j)
        
        # If any divisor has no candidates, fail
        for i in range(n):
            if not adj[i]:
                return None
        
        # Try to find perfect matching using DFS (n is small: 10)
        match_to = [-1] * len(values)  # value index -> divisor index
        match_from = [-1] * n  # divisor index -> value index
        
        def bpm(u, seen):
            for v in adj[u]:
                if seen[v]:
                    continue
                seen[v] = True
                if match_to[v] == -1 or bpm(match_to[v], seen):
                    match_to[v] = u
                    match_from[u] = v
                    return True
            return False
        
        for u in range(n):
            seen = [False] * len(values)
            if not bpm(u, seen):
                return None
        
        # Reconstruct
        result = [None] * n
        for u in range(n):
            result[u] = values[match_from[u]]
        return result
    
    # Search strategy:
    # For each W starting from 9:
    #   For L in a reasonable range, check if perfect matching exists.
    #   To make this efficient, note that the pattern repeats with period lcm,
    #   but we can also observe that for small W, the constraints are tight.
    
    # Better: for each W, the feasible L values are those where the "coverage" 
    # pattern allows a matching. Let's just search L up to a bound.
    
    # Bound for L: Since values must be <= MAX_VAL, L <= MAX_VAL - W.
    # But we can stop early. For small W, solutions exist for small L.
    
    # Use the fact that if a solution exists for some L, then for L' = L + lcm_period,
    # the same relative pattern holds. But lcm is huge.
    
    # Practical approach: search L from 1 to some limit, increasing as W increases.
    # For W=9, each divisor has at most 2 candidates (since W+1=10 values, 
    # and for divisor d, roughly 10/d candidates).
    
    # Let's use a hybrid: greedy first, then backtracking if greedy fails.
    
    for W in range(9, 50):
        # Determine search range for L
        # For small W, we can search a smaller range
        # The maximum L we need to check: since the pattern is periodic with lcm,
        # but lcm is too large, we use a heuristic bound.
        # In practice, for W < 30, solutions are found for L < 100000.
        
        # Use adaptive bound: start small, expand if needed
        search_bounds = [5000, 20000, 100000, 500000]
        
        for bound in search_bounds:
            for L in range(1, min(bound, MAX_VAL - W) + 1):
                if time.time() - start_time > DEADLINE - 3:
                    break
            
            if time.time() - start_time > DEADLINE - 3:
                break
            
            # Check if we broke due to time
            if time.time() - start_time > DEADLINE - 3:
                break
            
            # For efficiency, first try greedy (fast), then backtracking (slower)
            for L in range(1, min(bound, MAX_VAL - W) + 1):
                if time.time() - start_time > DEADLINE - 3:
                    return DIVISORS[:]
                
                # Quick check: does each divisor have at least one candidate?
                feasible = True
                for d in DIVISORS:
                    start_k = (L + d - 1) // d
                    end_k = (L + W) // d
                    if start_k > end_k:
                        feasible = False
                        break
                if not feasible:
                    continue
                
                # Try greedy first
                res = _greedy_assign(DIVISORS, L, W)
                if res is not None:
                    return [int(x) for x in res]
                
                # Try backtracking
                res = _backtrack_assign(DIVISORS, L, W)
                if res is not None:
                    return [int(x) for x in res]
        
        if time.time() - start_time > DEADLINE - 3:
            break
    
    # Fallback: return divisors themselves (valid but not optimal)
    return [int(d) for d in DIVISORS]
# EVOLVE-BLOCK-END
