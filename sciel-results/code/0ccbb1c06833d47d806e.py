"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    # Precompute modular inverses for all residues 1..p-1
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # Self-inverses in [1, p-1]: x where x*x % p == 1, i.e., x = 1 or x = p-1 = 1008
    # These cannot appear in any valid interval.
    # So valid s range: [2, 1007], valid e range: [s, 1007]
    
    # For each inverse pair (a, b) with a < b (a*b % p == 1):
    #   Constraint: interval [s, e] must not contain both a and b.
    #   Since a < b, if s <= a and b <= e, it's invalid.
    #   For fixed s: e must be < b for all pairs with a >= s.
    #   So e_max(s) = min{b - 1 : (a,b) is a pair, a < b, a >= s}
    #   This is a suffix minimum over a.
    
    INF = float('inf')
    suffix_min = [INF] * (p + 1)
    # suffix_min[a] = min{b - 1 : a' >= a, a' < inv[a'], a'*inv[a'] % p == 1}
    for a in range(p - 1, 0, -1):
        val = INF
        if a < inv[a]:
            val = inv[a] - 1
        suffix_min[a] = min(val, suffix_min[a + 1])
    
    best_len = 0
    best_start = 2
    
    for s in range(2, p - 1):  # s in [2, 1007]
        e_max = min(suffix_min[s], p - 2)  # e can be at most 1007
        if e_max < s:
            continue
        length = e_max - s + 1
        if length > best_len:
            best_len = length
            best_start = s
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
