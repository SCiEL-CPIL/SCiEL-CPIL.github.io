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

# Precompute for each residue r in 0..LCM-1, for each i in 1..12:
#   can_match[r][i] = list of offsets d in [1, LCM] such that (r + d) % i == 0
# Actually we need to think about it differently.
# 
# f(12, m) is the least L such that in (m, m+L] there exist distinct b_1,...,b_12 with i | b_i.
# By periodicity mod LCM, the pattern of divisibility depends only on m mod LCM.
# But L can be up to 144, which is larger than LCM=27720? No, LCM=27720 is large.
# Wait, L is at most 144. So we only need to check windows of length up to 144.
#
# For a given m, we want the maximum L such that in EVERY window of length L starting at some point... 
# No wait. f(12,m) is the LEAST positive L such that (m, m+L] contains a valid matching.
# So f(12,m) = minimum over L of: L is valid if there exists a matching in (m, m+L].
# We want to MAXIMIZE f(12,m), i.e., make the smallest valid L as large as possible.
#
# So for each m, f(12,m) = the minimum L >= 1 such that there exist distinct b_i in (m, m+L] with i | b_i.
# Equivalently, f(12,m) = 1 + max over all "blocked" prefixes... no.
#
# Let's think: for a fixed m, define g(L) = True if (m, m+L] contains a valid matching.
# g(L) is monotone non-decreasing in L. f(12,m) = min{L : g(L) = True}.
# To maximize f(12,m), we want the smallest L for which g(L) becomes True to be as large as possible.
# That means we want to delay the first point where a matching exists.
#
# For a window (m, m+L], a matching exists iff we can assign each i in 1..12 to a distinct number in (m,m+L] divisible by i.
# This is a bipartite matching problem.
#
# Key insight: since L <= 144, and the divisibility pattern repeats every LCM, but L is small,
# the structure depends on m mod LCM but only through which numbers in the next 144 positions are divisible by which i.
#
# For each residue r = m % LCM, the sequence of numbers r+1, r+2, ..., r+144 has a fixed divisibility pattern.
# So f(12, m) depends only on r = m % LCM.
#
# There are LCM = 27720 possible residues. For each residue, we can precompute f(12, r).
# Then at runtime, construct() returns an m in 0..1000000 that maximizes f(12, m).
# Since m can be any value 0..1000000, and f depends on m % LCM, we just need the best residue.
# But wait: m is in 0..1000000, so not all residues are reachable? 
# 1000000 / 27720 ≈ 36, so all residues are reachable (since 27720 < 1000001).
#
# So: precompute best_f = max over r in 0..LCM-1 of f(12, r).
# Then find any m in 0..1000000 with m % LCM = r_best. Return that m.
#
# Now, how to compute f(12, r) efficiently?
# For a given r, the numbers in (r, r+L] are r+1, ..., r+L.
# For each position j in 1..L, the number is r+j, and it's divisible by i iff (r+j) % i == 0.
# We need the minimum L such that there's a matching.
#
# We can binary search on L for each r, but that's LCM * log(144) * matching_time.
# Better: for each r, just iterate L from 1 to 144 and check if matching exists.
# The first L where matching exists is f(12, r).
#
# Matching check: bipartite graph with left nodes 1..12, right nodes 1..L (positions).
# Edge from i to position j iff (r+j) % i == 0.
# We need a perfect matching.
#
# Since L <= 144 and left side is 12, we can use a simple backtracking or Hall's theorem check,
# or just use a simple DFS-based bipartite matching (Hopcroft-Karp overkill, simple DFS works for size 12).
#
# Let's precompute for all r.

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

def _compute_f_for_residue(r, max_L=144):
    """Compute f(12, r) = least L such that (r, r+L] has valid matching."""
    for L in range(1, max_L + 1):
        # Build adjacency: left node i (1-indexed, so i-1 for 0-indexed) connects to position j (1-indexed)
        # if (r + j) % i == 0
        adj = [[] for _ in range(12)]
        for i in range(1, 13):
            for j in range(1, L + 1):
                if (r + j) % i == 0:
                    adj[i-1].append(j-1)  # 0-indexed position
        if _max_bipartite_matching(adj) == 12:
            return L
    return max_L  # should not happen

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
# m = _best_r + k * LCM for some k >= 0
# We need m <= 1000000
_k_max = (1000000 - _best_r) // _LCM
if _k_max < 0:
    _best_m = _best_r  # should not happen
else:
    _best_m = _best_r  # smallest such m is fine

def construct():
    return int(_best_m)
# EVOLVE-BLOCK-END
