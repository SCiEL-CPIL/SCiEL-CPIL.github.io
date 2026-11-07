"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    num_subsets = 1 << n
    
    # Precompute validity of each subset: is it a clique or independent set?
    # is_valid[S] = True if S is a clique or S is an independent set
    is_valid = [False] * num_subsets
    is_valid[0] = False # Empty set is not a part
    
    # Helper to check clique/ind set
    # A set S is a clique if for all v in S, (S & ~adj[v] & ~(1<<v)) == 0
    # A set S is an independent set if for all v in S, (S & adj[v]) == 0
    
    # Efficient precomputation using recurrence:
    # A set S is a clique if S \ {v} is a clique and v is adjacent to all in S \ {v}
    # A set S is an independent set if S \ {v} is an independent set and v is not adjacent to any in S \ {v}
    is_clique = [False] * num_subsets
    is_ind = [False] * num_subsets
    is_clique[0] = True
    is_ind[0] = True
    
    for S in range(1, num_subsets):
        # Get lowest set bit
        lsb = S & (-S)
        v = lsb.bit_length() - 1
        S_without_v = S ^ lsb
        
        if is_clique[S_without_v] and (adj[v] & S_without_v) == S_without_v:
            is_clique[S] = True
            is_valid[S] = True
            continue
        
        if is_ind[S_without_v] and (adj[v] & S_without_v) == 0:
            is_ind[S] = True
            is_valid[S] = True
            continue

    # DP
    # dp[S] = min number of parts to partition S
    # parent[S] = the last part T chosen for S, such that dp[S] = 1 + dp[S ^ T]
    INF = 10**9
    dp = [INF] * num_subsets
    parent_part = [0] * num_subsets
    dp[0] = 0
    
    # Iterate through all subsets S
    for S in range(1, num_subsets):
        # If S itself is valid, dp[S] can be 1
        if is_valid[S]:
            dp[S] = 1
            parent_part[S] = S
            continue # No need to check submasks if 1 is the minimum possible
            
        # Otherwise, find min over valid submasks T of S
        # To ensure we find the min, we iterate all submasks.
        # Optimization: Only iterate valid submasks.
        # Generating all submasks of S:
        sub = (S - 1) & S
        while sub:
            if is_valid[sub]:
                rest = S ^ sub
                if dp[rest] + 1 < dp[S]:
                    dp[S] = dp[rest] + 1
                    parent_part[S] = sub
            sub = (sub - 1) & S
            
            # Deadline check
            if time.time() - start_time > 50:
                # This is extremely unlikely to happen for n=14
                break
        # Note: It is guaranteed that dp[S] will be set because single vertices are always valid (clique or ind set)
        # So we will always find at least the single-vertex submasks.

    # Reconstruct the partition
    parts = []
    S = full_mask
    while S:
        part_mask = parent_part[S]
        # Convert mask to list of vertices
        part_verts = [i for i in range(n) if (part_mask >> i) & 1]
        parts.append(part_verts)
        S = S ^ part_mask
        
    # Convert to required types: list of lists of ints
    # Ensure ints are Python ints
    final_parts = []
    for part in parts:
        final_parts.append([int(v) for v in part])
        
    return final_parts

# EVOLVE-BLOCK-END
