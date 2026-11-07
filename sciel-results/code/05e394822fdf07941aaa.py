"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    deadline = time.time() + 85.0
    n = P['n']
    edges = P['edges']

    # Adjacency bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    full_mask = (1 << n) - 1
    size = 1 << n

    # Precompute valid masks (clique or independent set)
    is_clique = [0] * size
    is_indep = [0] * size
    is_clique[0] = 1
    is_indep[0] = 1
    valid = [0] * size
    valid[0] = 1

    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = 1
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = 1
            
        if is_clique[mask] or is_indep[mask]:
            valid[mask] = 1

    # Subset DP: dp[mask] = minimum number of parts to partition mask
    INF = 100
    dp = [INF] * size
    dp[0] = 0
    best_sub = [0] * size

    for mask in range(1, size):
        best = INF
        sub = mask
        best_s = 0
        while sub:
            if valid[sub]:
                val = dp[mask ^ sub] + 1
                if val < best:
                    best = val
                    best_s = sub
                    if best == 1:
                        break
            sub = (sub - 1) & mask
        dp[mask] = best
        best_sub[mask] = best_s
        
        if time.time() > deadline:
            break

    if dp[full_mask] < INF:
        # Reconstruct partition
        parts = []
        mask = full_mask
        while mask:
            sub = best_sub[mask]
            if sub == 0:
                # Fallback if reconstruction fails (should not happen)
                part = []
                m = mask
                while m:
                    lb = m & (-m)
                    v = lb.bit_length() - 1
                    part.append(v)
                    m ^= lb
                parts.append(part)
                break
            
            part = []
            m = sub
            while m:
                lb = m & (-m)
                v = lb.bit_length() - 1
                part.append(v)
                m ^= lb
            parts.append(part)
            mask ^= sub
            
        # Ensure types are Python int and list
        result = []
        for part in parts:
            result.append([int(v) for v in part])
        return result
    else:
        # Fallback: single vertices (always valid)
        return [[i] for i in range(n)]
# EVOLVE-BLOCK-END
