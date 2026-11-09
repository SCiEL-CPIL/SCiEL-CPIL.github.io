# EVOLVE-BLOCK-START
import math

P = {'n': 12, 'maximum_start': 1000000}

# LCM of 1..12
def _lcm(a, b):
    return a // math.gcd(a, b) * b

_LCM = 1
for _i in range(2, 13):
    _LCM = _lcm(_LCM, _i)

def _max_bipartite_matching_fast(adj):
    """
    adj: list of lists of right node indices.
    Returns size of max matching.
    Uses bitmask for visited to speed up DFS.
    """
    n_left = len(adj)
    match_r = [-1] * 144  # Max L is 144, so max right index is 143
    
    # We only care about right nodes 0..L-1
    # We will pass L to limit the range if needed, but adj only contains valid nodes.
    
    def dfs(u, visited_mask1, visited_mask2):
        # visited_mask1 covers bits 0-63, visited_mask2 covers bits 64-127
        # L is at most 144, so we might need a third mask? 144 > 128.
        # Let's use a list of booleans for simplicity and speed if L is small, 
        # or just a set. Actually, a list of booleans is faster than a set for small sizes.
        pass

    # Let's stick to the list of booleans approach which is often faster than sets for small N in Python
    # But we can reuse the array.
    
    match_r = [-1] * 144
    result = 0
    for u in range(n_left):
        visited = [False] * 144
        if _dfs(u, adj, match_r, visited):
            result += 1
    return result

def _dfs(u, adj, match_r, visited):
    for v in adj[u]:
        if visited[v]:
            continue
        visited[v] = True
        if match_r[v] == -1 or _dfs(match_r[v], adj, match_r, visited):
            match_r[v] = u
            return True
    return False

def _compute_f_for_residue_fast(r, max_L=144):
    """Compute f(12, r) = least L such that (r, r+L] has valid matching."""
    # Precompute divisibility for r+1..r+144 for all i
    # div[i][j] = 1 if (r+j)%i==0 else 0
    # i in 1..12, j in 1..144
    div = [[0]*145 for _ in range(13)] # 1-indexed for i, 1-indexed for j
    for i in range(1, 13):
        for j in range(1, 145):
            if (r + j) % i == 0:
                div[i][j] = 1
    
    for L in range(1, max_L + 1):
        # Build adj
        adj = [[] for _ in range(12)]
        for i in range(1, 13):
            # adj[i-1] contains j-1 for j in 1..L if div[i][j] is 1
            # We can slice or loop
            # Loop is fine
            for j in range(1, L + 1):
                if div[i][j]:
                    adj[i-1].append(j-1)
        
        if _max_bipartite_matching_fast(adj) == 12:
            return L
    return max_L

def _precompute():
    best_f = 0
    best_r = 0
    for r in range(_LCM):
        f_val = _compute_f_for_residue_fast(r)
        if f_val > best_f:
            best_f = f_val
            best_r = r
    return best_f, best_r

_best_f, _best_r = _precompute()

_k_max = (1000000 - _best_r) // _LCM
if _k_max < 0:
    _best_m = _best_r
else:
    _best_m = _best_r

def construct():
    return int(_best_m)
# EVOLVE-BLOCK-END
