import time

# EVOLVE-BLOCK-START
P = 27720  # LCM(1..12)

def _precompute_masks():
    """For each residue r in [0, P), compute bitmask of which i (1..12) divide r."""
    masks = [0] * P
    for r in range(P):
        m = 0
        for i in range(1, 13):
            if r % i == 0:
                m |= (1 << (i - 1))
        masks[r] = m
    return masks

MASKS = _precompute_masks()

def _has_matching_residue(r, L):
    """Check if interval starting at residue r (i.e., m such that m+1 ≡ r mod P) 
    with length L contains distinct b_1..b_12 with i | b_i.
    r is (m+1) % P."""
    if L < 12:
        return False
    # adj[i] = list of j in [0, L) where (r + j) % (i+1) == 0
    adj = [[] for _ in range(12)]
    for j in range(L):
        res = (r + j) % P
        mask = MASKS[res]
        mm = mask
        while mm:
            low = mm & (-mm)
            i = low.bit_length() - 1
            adj[i].append(j)
            mm ^= low
    
    # Bipartite matching: left = i (12 nodes), right = j (L nodes)
    match_j = [-1] * L
    
    def dfs(i, visited):
        for j in adj[i]:
            if visited[j]:
                continue
            visited[j] = True
            if match_j[j] == -1 or dfs(match_j[j], visited):
                match_j[j] = i
                return True
        return False
    
    for i in range(12):
        visited = [False] * L
        if not dfs(i, visited):
            return False
    return True

def _f_residue(r):
    """Compute f(12, m) for m where (m+1) % P == r, via binary search on L."""
    lo, hi = 12, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_matching_residue(r, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def _precompute_all_f():
    """Precompute f for all residues r in [0, P)."""
    f_table = [0] * P
    for r in range(P):
        f_table[r] = _f_residue(r)
    return f_table

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    start_time = time.time()
    
    # Precompute f for all residues
    f_table = _precompute_all_f()
    
    # Find the residue with maximum f
    best_f = 0
    best_r = 0
    for r in range(P):
        if f_table[r] > best_f:
            best_f = f_table[r]
            best_r = r
    
    # Now find m in [0, 1000000] such that (m+1) % P == best_r
    # m+1 ≡ best_r (mod P) => m ≡ best_r - 1 (mod P)
    target = (best_r - 1) % P
    # Smallest m >= 0 with m % P == target
    best_m = target
    if best_m > 1000000:
        # Try to find a valid m
        # m = target + k*P for some k >= 0
        # We need m <= 1000000
        # If target > 1000000, no solution... but target < P = 27720 < 1000000, so this won't happen
        best_m = target
    
    elapsed = time.time() - start_time
    if elapsed > 80:
        # Emergency: return a valid m quickly
        return int(best_m)
    
    # Verify and potentially refine: check if there's a better m nearby
    # The f value depends only on (m+1) % P, so all m with the same residue have the same f.
    # But let's double-check our best answer is correct.
    
    return int(best_m)
# EVOLVE-BLOCK-END
