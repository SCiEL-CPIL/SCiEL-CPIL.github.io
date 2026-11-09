"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

P = {'n': 12, 'maximum_start': 1000000}

# LCM of 1..12
def _lcm(a, b):
    return a // math.gcd(a, b) * b

_LCM = 1
for _i in range(2, 13):
    _LCM = _lcm(_LCM, _i)

# Precompute positions for each modulus and remainder
# positions_for_mod[i][rem] contains j in 1..144 such that j % i == rem
# We store 0-indexed positions (j-1) for the matching algorithm
MAX_L = 144
positions_for_mod = {}
for i in range(1, 13):
    positions_for_mod[i] = [[] for _ in range(i)]
    for j in range(1, MAX_L + 1):
        rem = j % i
        positions_for_mod[i][rem].append(j - 1) # 0-indexed

def _max_bipartite_matching(adj):
    """adj[i] = list of right nodes adjacent to left node i (0-indexed left, 0-indexed right).
    Returns size of maximum matching."""
    n_left = len(adj)
    match_r = {}  # right -> left
    
    def dfs(u, visited):
        for v in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            if v not in match_r or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    result = 0
    for u in range(n_left):
        if dfs(u, set()):
            result += 1
    return result

def _compute_f_for_residue(r):
    """Compute f(12, r) = least L such that (r, r+L] has valid matching."""
    # We need to check L from 1 to MAX_L
    # For each L, we need adj[i] to only contain positions < L
    # Instead of rebuilding adj from scratch, we can filter the precomputed lists?
    # Or just build adj once for MAX_L and then check if a matching exists using only first L nodes?
    # The matching algorithm checks against the full adj list.
    # We can modify the matching algorithm to take a limit L, or just filter adj.
    
    # Filtering is O(12 * L) which is small.
    # But we can be smarter: 
    # If we precompute adj_full[i] for all 144 positions, 
    # then for a specific L, adj_L[i] = [v for v in adj_full[i] if v < L]
    
    adj_full = [[] for _ in range(12)]
    for i in range(1, 13):
        rem = (-r) % i
        # Copy the list to avoid mutation issues if we were modifying, but here we just read
        # Actually, we can just reference the precomputed list and filter in the matching loop?
        # No, the matching loop iterates over adj[u].
        # Let's just build adj_full once.
        adj_full[i-1] = positions_for_mod[i][rem]
        
    for L in range(1, MAX_L + 1):
        # Create filtered adjacency lists
        adj = []
        for i in range(12):
            # Filter positions < L
            # Since positions are sorted (j increases), we can use bisect or just list comprehension
            # List comprehension is fast for small lists
            adj.append([v for v in adj_full[i] if v < L])
            
        if _max_bipartite_matching(adj) == 12:
            return L
            
    return MAX_L

# Precompute
def _precompute():
    best_f = 0
    best_r = 0
    for r in range(_LCM):
        f_val = _compute_f_for_residue(r)
        if f_val > best_f:
            best_f = f_val
            best_r = r
    return best_f, best_r

_best_f, _best_r = _precompute()

# Find m in 0..1000000 with m % LCM == _best_r
_k_max = (1000000 - _best_r) // _LCM
if _k_max < 0:
    _best_m = _best_r
else:
    _best_m = _best_r

def construct():
    return int(_best_m)
# EVOLVE-BLOCK-END
