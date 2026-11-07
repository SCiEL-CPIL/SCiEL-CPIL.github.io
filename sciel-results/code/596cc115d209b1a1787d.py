"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmask
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute: for each vertex, the set of vertices it is NOT adjacent to (including itself for independence checks)
    # non_adj_mask[v] has bits set for vertices u where (u,v) is NOT an edge and u != v
    non_adj_mask = [0] * n
    for v in range(n):
        # All vertices except v
        all_others = ((1 << n) - 1) ^ (1 << v)
        # Vertices not adjacent to v
        non_adj_mask[v] = all_others & ~adj_mask[v]
    
    # Precompute clique and independent set checks for all masks using DP
    # is_clique[m] = True if mask m forms a clique
    # is_indep[m] = True if mask m forms an independent set
    full_mask = (1 << n) - 1
    
    is_clique = [False] * (1 << n)
    is_indep = [False] * (1 << n)
    is_clique[0] = True
    is_indep[0] = True
    
    for mask in range(1, 1 << n):
        # Get lowest set bit
        low = mask & (-mask)
        v = low.bit_length() - 1
        rest = mask ^ low
        
        # Check clique: v must be adjacent to all vertices in rest, and rest must be a clique
        if is_clique[rest] and (rest & ~adj_mask[v]) == 0:
            is_clique[mask] = True
        
        # Check independent set: v must NOT be adjacent to any vertex in rest, and rest must be independent
        if is_indep[rest] and (rest & adj_mask[v]) == 0:
            is_indep[mask] = True
    
    # A mask is a valid part if it's a clique or independent set
    is_valid = [False] * (1 << n)
    for mask in range(1 << n):
        is_valid[mask] = is_clique[mask] or is_indep[mask]
    
    # Find all valid submasks of a given mask
    # For a given mask, we want to find valid submasks to use in DP
    # Precompute for each mask the list of valid submasks? That's too much.
    # Instead, in the DP, we enumerate submasks and check validity.
    
    # DP: dp[mask] = minimum number of parts to partition the set of vertices in mask
    # dp[0] = 0
    # dp[mask] = 1 + min(dp[mask ^ submask]) over all valid submasks of mask
    
    # We'll use memoized DFS with bitmask
    dp = [-1] * (1 << n)
    dp[0] = 0
    
    # Store the partition (as list of masks) for reconstruction
    best_partition = [None] * (1 << n)
    
    deadline = time.time() + 80  # Safety margin
    
    def solve(mask):
        if time.time() > deadline:
            return 999
        if dp[mask] != -1:
            return dp[mask]
        
        best = 999
        best_sub = None
        
        # Enumerate all submasks of mask
        # We want to find valid submasks
        # To make this efficient, we can try to find a valid submask that includes a specific vertex
        # Let's pick the lowest set bit to break symmetry
        low = mask & (-mask)
        
        sub = mask
        while sub:
            # We only consider submasks that include 'low' to avoid symmetric duplicates
            if sub & low:
                if is_valid[sub]:
                    rest = mask ^ sub
                    val = 1 + solve(rest)
                    if val < best:
                        best = val
                        best_sub = sub
            # Next submask
            sub = (sub - 1) & mask
        
        dp[mask] = best
        best_partition[mask] = best_sub
        return best
    
    # Solve for the full mask
    full = (1 << n) - 1
    result_count = solve(full)
    
    # Reconstruct the partition
    if result_count >= 999:
        # Fallback to greedy
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                part_mask = 0
                for x in part:
                    part_mask |= (1 << x)
                trial_mask = part_mask | (1 << v)
                if is_valid[trial_mask]:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts
    
    # Reconstruct
    partition_masks = []
    mask = full
    while mask:
        sub = best_partition[mask]
        if sub is None:
            break
        partition_masks.append(sub)
        mask ^= sub
    
    # Convert masks to lists of vertices
    result = []
    for pm in partition_masks:
        verts = []
        m = pm
        while m:
            low = m & (-m)
            v = low.bit_length() - 1
            verts.append(int(v))
            m ^= low
        if verts:
            result.append(verts)
    
    # Verify
    all_vertices = []
    for part in result:
        all_vertices.extend(part)
    if sorted(all_vertices) != list(range(n)) or len(result) == 0:
        # Fallback to greedy
        parts = []
        for v in range(n):
            placed = False
            for part in parts:
                part_mask = 0
                for x in part:
                    part_mask |= (1 << x)
                trial_mask = part_mask | (1 << v)
                if is_valid[trial_mask]:
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return parts
    
    return result
# EVOLVE-BLOCK-END
