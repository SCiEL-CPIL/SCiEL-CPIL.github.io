import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask for each vertex
    # adj[v] is a bitmask of neighbors of v
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Precompute validity of all subsets
    # valid[mask] is True if mask forms a clique or an independent set
    total_masks = 1 << n
    valid = [False] * total_masks
    valid[0] = True
    
    for mask in range(1, total_masks):
        # Check if clique
        is_clique = True
        # Check if independent
        is_indep = True
        
        vertices_in_mask = []
        for i in range(n):
            if mask & (1 << i):
                vertices_in_mask.append(i)
        
        # Check pairs
        for i in range(len(vertices_in_mask)):
            for j in range(i+1, len(vertices_in_mask)):
                u = vertices_in_mask[i]
                v = vertices_in_mask[j]
                if (adj[u] & (1 << v)) == 0:
                    is_clique = False
                else:
                    is_indep = False
                if not is_clique and not is_indep:
                    break
            if not is_clique and not is_indep:
                break
        
        if is_clique or is_indep:
            valid[mask] = True
            
    # List of valid masks for quick iteration
    all_valid_masks = [mask for mask in range(1, total_masks) if valid[mask]]
    
    # Precompute for each vertex v, the list of valid masks containing v
    valid_masks_containing = [[] for _ in range(n)]
    for mask in all_valid_masks:
        for v in range(n):
            if mask & (1 << v):
                valid_masks_containing[v].append(mask)
                
    # Sort masks in each list by size descending to cover more ground early
    for v in range(n):
        valid_masks_containing[v].sort(key=lambda x: bin(x).count('1'), reverse=True)

    # Precompute maximum clique size and maximum independent set size for lower bounds
    # For a given remaining mask, the min number of parts needed is at least:
    # 1. The size of the maximum clique in rem_mask (since each part can contain at most one clique vertex if they're not all in one part... wait no)
    # Actually, each part is either a clique or independent set.
    # If we have a clique of size k in rem_mask, those k vertices could all be in one part (as a clique), so that doesn't give a lower bound > 1.
    # Better lower bound: Consider the complement. If rem_mask has an independent set of size k, those k vertices could all be in one part.
    # Actually, a better approach: For any set S, the min number of parts to cover S is at least the ceiling of |S| / (max clique size in S).
    # Also at least the ceiling of |S| / (max independent set size in S).
    # But these are weak.
    
    # A stronger lower bound: Let's use the fact that each part is either a clique or independent set.
    # For a given rem_mask, let's compute:
    # - max_clique_size: largest clique in rem_mask
    # - max_indep_size: largest independent set in rem_mask
    # Then min parts >= max(ceil(|rem_mask| / max_clique_size), ceil(|rem_mask| / max_indep_size))
    # But this is still weak.
    
    # Alternative: Use a greedy lower bound. Try to greedily pack vertices into parts and count.
    # Or use the following: For each vertex v in rem_mask, it must be in some part. 
    # The minimum number of parts is at least 1 if rem_mask is non-empty.
    
    # Let's use a different approach: precompute for each mask the minimum number of parts needed (using DP)
    # But that's what we're trying to solve.
    
    # Simpler: Use the fact that if rem_mask itself is a valid part, we need 1 more.
    # Otherwise, we need at least 2 more (if we can split into 2 valid parts).
    # This is still weak.
    
    # Let's try: lower_bound = 1 if valid[rem_mask] else 2 (if rem_mask can be split into 2 valid parts) else 3...
    # This is getting complex. Let's just use a simple bound: 
    # min_parts >= 1 if rem_mask != 0
    
    # Better: Precompute the minimum number of valid parts needed to cover each mask using DP.
    # This is essentially solving the problem, but we can do it once for all masks.
    # dp[mask] = min number of valid parts to cover mask
    # dp[0] = 0
    # dp[mask] = min over valid submasks S of mask: dp[mask ^ S] + 1
    
    # This DP will take O(3^n) which for n=14 is about 4.78M, which is fine.
    
    # Let's precompute dp[mask] for all masks
    import math
    INF = float('inf')
    dp = [INF] * total_masks
    dp[0] = 0
    
    for mask in range(1, total_masks):
        # Try all valid submasks of mask
        sub = mask
        while sub:
            if valid[sub]:
                if dp[mask ^ sub] + 1 < dp[mask]:
                    dp[mask] = dp[mask ^ sub] + 1
            sub = (sub - 1) & mask
            
    # Now we have exact minimum for each mask!
    # We can use this to directly return the answer.
    # But we also need the actual partition, not just the count.
    # Let's store the choice as well.
    
    choice = [0] * total_masks
    for mask in range(1, total_masks):
        sub = mask
        best_val = INF
        best_sub = 0
        while sub:
            if valid[sub]:
                val = dp[mask ^ sub] + 1
                if val < best_val:
                    best_val = val
                    best_sub = sub
            sub = (sub - 1) & mask
        choice[mask] = best_sub
        dp[mask] = best_val
    
    # Now reconstruct the partition for target
    target = (1 << n) - 1
    partition_masks = []
    cur = target
    while cur:
        s = choice[cur]
        if s == 0:
            # Should not happen if dp is correct
            break
        partition_masks.append(s)
        cur ^= s
    
    # Convert to list of vertex lists
    final_partition = []
    for mask in partition_masks:
        verts = [v for v in range(n) if mask & (1 << v)]
        final_partition.append(verts)
        
    return final_partition
# EVOLVE-BLOCK-END
