"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def _lcm(a, b):
    return a // math.gcd(a, b) * b

_LCM = 1
for _i in range(2, 13):
    _LCM = _lcm(_LCM, _i)

# Precompute for each residue r in 0..LCM-1, the mask of which i in 1..12 divide r+j
# for j = 1, 2, ..., 144. Then compute f(12, r) = min L such that positions 1..L have perfect matching.
#
# Key insight: for a given r, the divisibility pattern of r+1, r+2, ..., r+144 is fixed.
# We can encode for each position j (1-indexed) a bitmask of which i in 1..12 divide r+j.
# Then we need to find the smallest L such that there's a perfect matching between i=1..12 and positions 1..L
# where position j is assigned to i only if i's bit is set in mask[j].
#
# To avoid runtime matching, we precompute for all 27720 residues the value f(12, r).
# Since LCM = 27720 and max L = 144, this is 27720 * 144 matching checks.
# Each matching check is on a graph with 12 left nodes and up to 144 right nodes.
# We use a simple DFS-based bipartite matching.

def _has_perfect_matching(adj, n_left, n_right):
    """adj[u] = list of right nodes adjacent to left node u.
    Returns True if there's a perfect matching (all n_left left nodes matched)."""
    match_r = [-1] * n_right  # match_r[v] = u or -1
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                return True
        return False
    
    for u in range(n_left):
        if not dfs(u, [False] * n_right):
            return False
    return True

def _compute_f_for_residue(r, max_L=144):
    """Compute f(12, r) = least L such that (r, r+L] has valid matching."""
    # Precompute masks for positions 1..max_L
    # mask[j] = bitmask where bit (i-1) is set iff (r+j) % i == 0
    masks = [0] * (max_L + 1)  # 1-indexed
    for j in range(1, max_L + 1):
        m = 0
        for i in range(1, 13):
            if (r + j) % i == 0:
                m |= (1 << (i - 1))
        masks[j] = m
    
    for L in range(1, max_L + 1):
        # Build adjacency: left node i-1 (0-indexed) connects to right node j-1 (0-indexed)
        # if bit (i-1) is set in masks[j]
        adj = [[] for _ in range(12)]
        for j in range(1, L + 1):
            m = masks[j]
            for i in range(1, 13):
                if m & (1 << (i - 1)):
                    adj[i - 1].append(j - 1)
        if _has_perfect_matching(adj, 12, L):
            return L
    return max_L

# Precompute best residue
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
