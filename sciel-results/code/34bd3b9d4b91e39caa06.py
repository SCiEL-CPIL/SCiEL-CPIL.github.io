"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    deadline = time.time() + 55.0  # safety margin before 60s limit
    
    # Build adjacency bitmask
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute which subsets are cliques or independent sets
    # A subset S is a clique if for every pair in S, they are adjacent
    # A subset S is an independent set if for every pair in S, they are NOT adjacent
    
    # We'll precompute is_valid[mask] for all masks
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_valid = [False] * (1 << n)
    
    is_clique[0] = True
    is_independent[0] = True
    is_valid[0] = True
    
    # For each mask, check if it's a clique or independent set
    # A mask is a clique if every vertex in it is adjacent to every other vertex in it
    # A mask is an independent set if no two vertices in it are adjacent
    
    # Efficient check: for a mask S, pick the lowest bit v. Then S is a clique iff
    # (S without v) is a clique AND v is adjacent to all vertices in (S without v).
    # Similarly for independent sets.
    
    for mask in range(1, 1 << n):
        # Find lowest set bit
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        # Check clique: v must be adjacent to all in rest, and rest must be clique
        # v is adjacent to all in rest iff rest & ~adj[v] == 0
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
        
        # Check independent: v must NOT be adjacent to any in rest, and rest must be independent
        # v is not adjacent to any in rest iff rest & adj[v] == 0
        if is_independent[rest] and (rest & adj[v]) == 0:
            is_independent[mask] = True
        
        is_valid[mask] = is_clique[mask] or is_independent[mask]
    
    # Now solve the minimum partition problem using DP
    # dp[mask] = minimum number of valid parts to partition the set 'mask'
    # dp[0] = 0
    # dp[mask] = min over all valid subsets S of mask: 1 + dp[mask ^ S]
    
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    
    # To speed up, we can iterate over masks and for each valid subset S contained in mask
    # But that's expensive. Instead, use memoized recursion with pruning.
    
    # Alternative: iterate mask from 1 to full_mask, and for each valid subset S of mask,
    # update dp[mask] = min(dp[mask], 1 + dp[mask ^ S])
    # But enumerating all subsets of each mask is 3^n which is ~4.7M for n=14, feasible.
    
    # Actually, let's use a different approach: for each mask, try all valid subsets S that are
    # subsets of mask. We can enumerate subsets of mask efficiently.
    
    # Better approach: memoized DFS with pruning
    best_known = [float('inf')] * (1 << n)
    best_known[0] = 0
    
    @lru_cache(maxsize=None)
    def min_parts(mask):
        if mask == 0:
            return 0
        if best_known[mask] < float('inf'):
            return best_known[mask]
        
        # Lower bound: at least 1 part needed
        # Try all valid subsets S of mask
        # To enumerate subsets of mask: iterate sub = mask; sub = (sub-1) & mask
        best = float('inf')
        sub = mask
        while sub > 0:
            if is_valid[sub]:
                # Check if we can improve
                candidate = 1 + min_parts(mask ^ sub)
                if candidate < best:
                    best = candidate
                    if best == 1:  # can't do better than 1
                        break
            sub = (sub - 1) & mask
        
        best_known[mask] = best
        return best
    
    result_count = min_parts(full_mask)
    
    # Now reconstruct the actual partition
    # We need to find which subsets were used
    def reconstruct(mask, count):
        if mask == 0:
            return []
        # Find a valid subset S of mask such that 1 + min_parts(mask ^ S) == count
        sub = mask
        while sub > 0:
            if is_valid[sub]:
                if 1 + min_parts(mask ^ sub) == count:
                    rest_parts = reconstruct(mask ^ sub, count - 1)
                    return [list(range(n))]  # placeholder, we'll fix
            sub = (sub - 1) & mask
        return []
    
    # Let's do reconstruction more carefully
    def get_partition(mask, count):
        if mask == 0:
            return []
        sub = mask
        while sub > 0:
            if is_valid[sub]:
                if 1 + min_parts(mask ^ sub) == count:
                    rest = get_partition(mask ^ sub, count - 1)
                    part = [i for i in range(n) if (sub >> i) & 1]
                    return [part] + rest
            sub = (sub - 1) & mask
        return []
    
    partition = get_partition(full_mask, result_count)
    
    # Verify the partition is valid
    # Each part must be a nonempty list, vertices occur exactly once, and each part induces clique or independent set
    seen = [False] * n
    for part in partition:
        if not part:
            return [[i] for i in range(n)]  # fallback
        for v in part:
            if seen[v]:
                return [[i] for i in range(n)]
            seen[v] = True
        mask_part = 0
        for v in part:
            mask_part |= (1 << v)
        if not is_valid[mask_part]:
            return [[i] for i in range(n)]
    
    for i in range(n):
        if not seen[i]:
            return [[i] for i in range(n)]
    
    # Ensure types are correct: lists of Python ints
    result = [list(part) for part in partition]
    return result

# EVOLVE-BLOCK-END
