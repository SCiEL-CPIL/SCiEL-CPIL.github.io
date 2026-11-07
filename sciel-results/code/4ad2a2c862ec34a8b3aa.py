"""
Cochromatic partition search using Dynamic Programming over subsets.
"""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Build adjacency masks
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Precompute is_indep and is_clique for all masks
    # is_indep[mask]: 1 if mask is an independent set, 0 otherwise
    # is_clique[mask]: 1 if mask is a clique, 0 otherwise
    size = 1 << n
    is_indep = [0] * size
    is_clique = [0] * size
    is_indep[0] = 1
    is_clique[0] = 1
    
    for mask in range(1, size):
        # Find lowest set bit
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        rest = mask ^ lb
        
        # Check independent set: v must not be adjacent to any vertex in rest
        if is_indep[rest] and (adj[v] & rest) == 0:
            is_indep[mask] = 1
            
        # Check clique: v must be adjacent to ALL vertices in rest
        if is_clique[rest] and (adj[v] & rest) == rest:
            is_clique[mask] = 1

    # DP to find minimum number of parts
    # dp[mask] = min number of parts to cover 'mask'
    # parent[mask] = the subset 'sub' chosen as one part in the optimal solution for 'mask'
    INF = n + 1
    dp = [INF] * size
    parent = [0] * size
    dp[0] = 0
    
    # Deadline guard: keep a safety margin before the 90s limit
    start_time = time.time()
    deadline = start_time + 80.0  # 80s leaves 10s margin
    
    for mask in range(1, size):
        # Iterate over all submasks of mask
        sub = mask
        best_val = INF
        best_sub = 0
        
        while sub:
            # A valid part must be a clique or an independent set
            if is_indep[sub] or is_clique[sub]:
                remaining = mask ^ sub
                val = dp[remaining] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
                    # Early termination: if we found 1, that's optimal
                    if best_val == 1:
                        break
            sub = (sub - 1) & mask
        
        dp[mask] = best_val
        parent[mask] = best_sub
        
        # Check deadline periodically
        if mask % 1024 == 0:
            if time.time() > deadline:
                # Return whatever we have so far (may be incomplete)
                # But we need a valid partition, so let's continue with what we have
                # Actually, if we break here, dp for remaining masks is INF
                # We need to handle this case - but for n=14, this shouldn't happen
                pass
        
    # Reconstruct the partition
    parts = []
    current_mask = full_mask
    while current_mask:
        sub = parent[current_mask]
        if sub == 0:
            # Fallback: if parent is 0 (shouldn't happen for valid masks), use the mask itself
            sub = current_mask
        # Convert mask to list of vertices
        verts = []
        for v in range(n):
            if (sub >> v) & 1:
                verts.append(v)
        parts.append(verts)
        current_mask ^= sub
        
    # Ensure types are correct: list of lists of ints
    # The problem states: "sequences must be list or tuple" and "integers must be Python int"
    # Our construction uses standard Python lists and ints.
    
    return parts

if __name__ == "__main__":
    # Simple self-test to ensure it runs within time
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Time taken: {end - start:.4f}s")
    print(f"Number of parts: {len(res)}")
    print(f"Partition: {res}")
