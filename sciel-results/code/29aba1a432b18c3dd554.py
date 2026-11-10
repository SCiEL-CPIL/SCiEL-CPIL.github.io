"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'n': 12, 'maximum_start': 1000000}
LCM_12 = 27720  # LCM(1,2,...,12)

def _has_perfect_matching(m, L):
    """Check if in (m, m+L] we can assign distinct b_i to each i in 1..12 with i|b_i."""
    n = 12
    # Build adjacency: for each i, which positions in (m, m+L] are divisible by i
    adj = [[] for _ in range(n)]
    for i in range(1, n+1):
        first = m + 1
        rem = first % i
        if rem == 0:
            first_mult = first
        else:
            first_mult = first + (i - rem)
        for b in range(first_mult, m + L + 1, i):
            idx = b - m - 1
            if 0 <= idx < L:
                adj[i-1].append(idx)
    
    # Check if any left node has no candidates
    for i in range(n):
        if not adj[i]:
            return False
    
    # Order left nodes by candidate count (most constrained first)
    order = sorted(range(n), key=lambda i: len(adj[i]))
    
    # DFS-based bipartite matching
    match_r = [-1] * L
    match_l = [-1] * n
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited[v]:
                continue
            visited[v] = True
            if match_r[v] == -1 or dfs(match_r[v], visited):
                match_r[v] = u
                match_l[u] = v
                return True
        return False
    
    for u in order:
        visited = [False] * L
        if not dfs(u, visited):
            return False
    return True

def _compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def _precompute_f_table():
    """Precompute f(12, m) for all m in [0, LCM_12) using periodicity."""
    f_table = [0] * LCM_12
    for m in range(LCM_12):
        f_table[m] = _compute_f(m)
    return f_table

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Precompute the full f-table for one period
    # This is the key structural change: exploit full periodicity
    f_table = _precompute_f_table()
    
    # Find the best residue in the table
    best_residue = 0
    best_f = 0
    for r in range(LCM_12):
        if f_table[r] > best_f:
            best_f = f_table[r]
            best_residue = r
    
    # Now find the best m in [0, 1000000] that has residue best_residue mod LCM_12
    # m = k * LCM_12 + best_residue for some k >= 0
    # We want to maximize f(12, m), which is f_table[m % LCM_12]
    # So any m with m % LCM_12 == best_residue gives f = best_f
    
    # But wait - we need to return a specific m, and the evaluator will compute f(12, m)
    # Since f is periodic with period LCM_12, all m with the same residue give the same f
    # So we just need to return any valid m with the best residue
    
    # The smallest such m is best_residue itself (if best_residue <= 1000000)
    # Or we can use any m = k * LCM_12 + best_residue within range
    
    # Let's pick the smallest valid m
    if best_residue <= 1000000:
        best_m = best_residue
    else:
        # This shouldn't happen since LCM_12 = 27720 < 1000000
        best_m = best_residue % LCM_12
    
    # Verify that best_m is in range
    if best_m < 0 or best_m > 1000000:
        best_m = 0
    
    # Double-check: compute f(12, best_m) to make sure it matches
    # (This is just a sanity check; the precomputation should be correct)
    # But let's also check nearby residues in case there's a tie or better option
    
    # Actually, let's be more careful: scan all residues and find the global max
    # Then return the smallest m with that residue
    
    # The precomputation already found the best residue, so best_m should be optimal
    
    # However, let's do a final verification by checking a few candidates
    # to ensure no bugs in the precomputation
    
    # Check best_m and a few neighbors
    candidates = [best_m]
    for dm in range(-5, 6):
        m2 = best_m + dm
        if 0 <= m2 <= 1000000:
            candidates.append(m2)
    
    # Also check m = 0 and m = LCM_12 - 1 as edge cases
    candidates.append(0)
    candidates.append(min(LCM_12 - 1, 1000000))
    
    # Deduplicate
    candidates = list(set(candidates))
    
    best_m_final = best_m
    best_f_final = _compute_f(best_m)
    
    for m2 in candidates:
        f_val = _compute_f(m2)
        if f_val > best_f_final:
            best_f_final = f_val
            best_m_final = m2
    
    # If we have time, do a more thorough scan to confirm
    if time.time() < deadline - 10:
        # Scan all residues again but with a slightly different approach
        # to cross-validate
        for r in range(LCM_12):
            if time.time() > deadline - 5:
                break
            m2 = r
            if m2 <= 1000000:
                f_val = _compute_f(m2)
                if f_val > best_f_final:
                    best_f_final = f_val
                    best_m_final = m2
    
    return int(best_m_final)
# EVOLVE-BLOCK-END
