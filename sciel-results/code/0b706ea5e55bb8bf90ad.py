import math
from functools import reduce

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst)

_DIVISORS = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
_LCM = _lcm_list(_DIVISORS)
_MAX = 1000000

def _has_perfect_matching(adj):
    n = len(adj)
    if n == 0:
        return []
    
    match_right = {}
    match_left = [-1] * n
    
    def dfs(u, visited):
        for v in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_right or dfs(match_right[v], visited):
                match_right[v] = u
                match_left[u] = v
                return True
        return False
    
    for u in range(n):
        visited = set()
        if not dfs(u, visited):
            return None
            
    return match_left

def _check_window(X, L, divisors):
    """
    Check if there are distinct multiples for all divisors in [X, X+L].
    Returns list of values if found, else None.
    """
    n = len(divisors)
    adj = []
    val_to_idx = {}
    right_nodes = []
    
    for i, d in enumerate(divisors):
        k_start = (X + d - 1) // d
        k_end = (X + L) // d
        
        adj_i = []
        for k in range(k_start, k_end + 1):
            val = k * d
            if val not in val_to_idx:
                idx = len(right_nodes)
                right_nodes.append(val)
                val_to_idx[val] = idx
            adj_i.append(val_to_idx[val])
        adj.append(adj_i)
        
    if any(len(a) == 0 for a in adj):
        return None
        
    match = _has_perfect_matching(adj)
    if match is not None:
        return [right_nodes[match[i]] for i in range(n)]
        
    return None

def construct():
    divisors = _DIVISORS
    R = _LCM
    
    best_L = float('inf')
    best_vals = None
    
    # We only need to check X up to R
    # The optimal L is expected to be small
    for X in range(1, R + 1):
        # Binary search for minimal L
        lo = 0
        hi = 1000  # Upper bound for L
        
        # Quick check: if X + hi > MAX, we might need to adjust, 
        # but since we just need *a* valid set and we can shift later, 
        # the relative window validity is what matters.
        # However, the values returned must be <= MAX.
        # We will handle shifting after finding the best relative window.
        
        while lo <= hi:
            mid = (lo + hi) // 2
            vals = _check_window(X, mid, divisors)
            if vals is not None:
                if mid < best_L:
                    best_L = mid
                    best_vals = vals
                hi = mid - 1
            else:
                lo = mid + 1
                
    if best_vals is None:
        # Fallback, shouldn't happen
        return divisors[:]
        
    # best_vals are in [X, X+best_L].
    # They are distinct and divisible by corresponding divisors.
    # Check if they are within [1, MAX].
    # Since X >= 1 and best_L is small, min >= 1.
    # Max <= R + 1000 < MAX.
    # So they are valid.
    
    return [int(v) for v in best_vals]
# EVOLVE-BLOCK-END
