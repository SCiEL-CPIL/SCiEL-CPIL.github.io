"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 50  # safety margin before 60s limit
    
    # Build adjacency as bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute: for each subset, is it a clique? is it an independent set?
    # A subset S is a clique if for every pair in S, they are adjacent.
    # A subset S is independent if for every pair in S, they are NOT adjacent.
    
    # Efficient check using bitmasks:
    # S is a clique iff for all v in S, S \ {v} is a subset of adj[v]
    # S is independent iff for all v in S, S \ {v} is a subset of (~adj[v]) & full_mask
    
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_clique[0] = True
    is_independent[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be adjacent to all in rest, and rest must be clique
        if rest == 0:
            is_clique[mask] = True
            is_independent[mask] = True
        else:
            # For clique: all vertices in rest must be in adj[v]
            if is_clique[rest] and (rest & ~adj[v] & full_mask) == 0:
                is_clique[mask] = True
            # For independent: all vertices in rest must NOT be in adj[v]
            non_adj_v = (~adj[v]) & full_mask
            if is_independent[rest] and (rest & non_adj_v) == rest:
                is_independent[mask] = True
    
    # DP: dp[mask] = minimum number of parts to partition mask
    # dp[0] = 0
    # dp[mask] = 1 + min over all valid submasks S of mask (clique or independent) of dp[mask ^ S]
    
    INF = 10**9
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    for mask in range(1, 1 << n):
        # Enumerate all submasks of mask
        sub = mask
        while sub > 0:
            if is_clique[sub] or is_independent[sub]:
                val = 1 + dp[mask ^ sub]
                if val < dp[mask]:
                    dp[mask] = val
            sub = (sub - 1) & mask
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    if time.time() > deadline:
        # Fallback: greedy
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                # Check if adding v keeps part as clique or independent
                ok_clique = True
                ok_indep = True
                for u in part:
                    if (adj[u] >> v) & 1:
                        ok_indep = False
                    else:
                        ok_clique = False
                if ok_clique or ok_indep:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts
    
    # Reconstruct partition
    result_parts = []
    mask = full_mask
    while mask > 0:
        # Find a submask S such that dp[mask] = 1 + dp[mask ^ S] and S is valid
        sub = mask
        found = False
        while sub > 0:
            if (is_clique[sub] or is_independent[sub]) and dp[mask ^ sub] == dp[mask] - 1:
                # Extract vertices in sub
                part = []
                s = sub
                while s > 0:
                    lsb2 = s & (-s)
                    v2 = lsb2.bit_length() - 1
                    part.append(v2)
                    s ^= lsb2
                result_parts.append(part)
                mask ^= sub
                found = True
                break
            sub = (sub - 1) & mask
        if not found:
            break
    
    # Verify all vertices are covered
    covered = set()
    for part in result_parts:
        for v in part:
            covered.add(v)
    if covered != set(range(n)):
        # Fallback to greedy
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                ok_clique = True
                ok_indep = True
                for u in part:
                    if (adj[u] >> v) & 1:
                        ok_indep = False
                    else:
                        ok_clique = False
                if ok_clique or ok_indep:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts
    
    # Return as list of lists of ints
    return [[int(v) for v in part] for part in result_parts]

# EVOLVE-BLOCK-END
