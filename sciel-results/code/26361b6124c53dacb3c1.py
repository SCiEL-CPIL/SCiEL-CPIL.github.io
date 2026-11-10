"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = 27720
N = 12
MAX_L = 200

def _has_matching(m, L):
    """
    Checks if there is a perfect matching between indices 1..N and positions 1..L
    in the interval (m, m+L] such that i divides m+j.
    """
    # Build adjacency list: for each index i, list of positions j
    adj = [[] for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, L + 1):
            if (m + j) % i == 0:
                adj[i].append(j)
    
    # Check if any index has no candidates
    for i in range(1, N + 1):
        if not adj[i]:
            return False
            
    # Order indices by constraint (fewest candidates first)
    indices = list(range(1, N + 1))
    indices.sort(key=lambda x: len(adj[x]))
    
    assigned = [0] * (L + 1) # assigned[j] = i if j is assigned to i
    used_indices = [False] * (N + 1)
    
    def backtrack(idx_pos):
        if idx_pos == N:
            return True
        i = indices[idx_pos]
        for j in adj[i]:
            if assigned[j] == 0:
                # Assign j to i
                assigned[j] = i
                used_indices[i] = True
                if backtrack(idx_pos + 1):
                    return True
                # Undo
                assigned[j] = 0
                used_indices[i] = False
        return False

    return backtrack(0)

def _compute_f(m):
    """
    Computes f(N, m) using binary search on L.
    """
    # Lower bound: at least N
    # Upper bound: MAX_L
    lo = 1
    hi = MAX_L
    ans = -1
    
    while lo <= hi:
        mid = (lo + hi) // 2
        if _has_matching(m, mid):
            ans = mid
            hi = mid - 1
        else:
            lo = mid + 1
            
    return ans if ans != -1 else MAX_L # Fallback, shouldn't happen

def construct():
    start_time = time.time()
    deadline = start_time + 55.0 # Safety margin
    
    best_m = 0
    best_f = -1
    
    # Search over one period
    for m in range(P):
        if time.time() > deadline:
            break
            
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            
    return int(best_m)
# EVOLVE-BLOCK-END
