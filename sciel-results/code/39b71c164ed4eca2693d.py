"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    es = {tuple(e) for e in edges}
    
    # Build adjacency sets
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Helper: check if a set of vertices (as a frozenset or tuple) is a clique or independent set
    def is_clique_or_ind(vertices):
        vlist = list(vertices)
        if len(vlist) <= 1:
            return True
        for i in range(len(vlist)):
            for j in range(i+1, len(vlist)):
                if vlist[i] in adj[vlist[j]]:
                    # has an edge
                    return None  # mixed, not clique nor ind
                # else no edge
        # check if all pairs have edges (clique) or none (independent)
        all_edges = all(vlist[i] in adj[vlist[j]] for i in range(len(vlist)) for j in range(i+1, len(vlist)))
        no_edges = all(vlist[i] not in adj[vlist[j]] for i in range(len(vlist)) for j in range(i+1, len(vlist)))
        return all_edges or no_edges
    
    # Better: use bitmask representation for speed
    # Represent each vertex set as a bitmask
    # Precompute for all subsets whether they are clique or independent
    
    # For n=14, there are 2^14 = 16384 subsets, which is manageable
    # But we need to find a partition into minimum number of such subsets
    
    # Let's use a different approach: since n=14 is small, we can use DP over subsets
    # But the state space is large. Let's try a smarter approach.
    
    # Actually, let's use a recursive backtracking with memoization
    # State: (remaining_vertices, current_parts)
    # But this is complex. Let's use a simpler approach:
    
    # First, let's find all maximal cliques and maximal independent sets
    # Then use these as building blocks
    
    # For n=14, let's just enumerate all subsets and check if they are clique or independent
    # There are 2^14 = 16384 subsets, which is feasible
    
    # Precompute: for each subset (as bitmask), is it a clique or independent set?
    valid_mask = {}
    for mask in range(1 << n):
        vertices = [i for i in range(n) if mask & (1 << i)]
        if len(vertices) <= 1:
            valid_mask[mask] = True
            continue
        # Check if clique or independent
        is_c = True
        is_i = True
        for i in range(len(vertices)):
            for j in range(i+1, len(vertices)):
                if vertices[i] in adj[vertices[j]]:
                    is_i = False
                else:
                    is_c = False
        valid_mask[mask] = is_c or is_i
    
    # Now find minimum number of valid masks that partition all vertices
    # This is a set cover problem with the constraint that sets must be disjoint and cover all vertices
    
    # Use DP: dp[mask] = minimum number of parts to cover vertices in 'mask'
    # dp[0] = 0
    # dp[mask] = min over all valid submasks s of mask: dp[mask ^ s] + 1
    
    # But this is O(3^n) which for n=14 is 3^14 ≈ 4.78 million, which might be slow in Python
    # Let's optimize: only consider valid submasks
    
    # Alternative: use the fact that we want minimum parts
    # Let's try a greedy + local search approach first, then refine
    
    # Actually, let's just use the DP approach with optimization
    # dp[mask] = min number of parts to cover 'mask'
    
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    
    # For each mask, find all valid submasks
    # To speed up, we can iterate over submasks of mask
    for mask in range(1, 1 << n):
        # Find all valid submasks of mask
        sub = mask
        while sub:
            if valid_mask[sub]:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
            sub = (sub - 1) & mask
    
    # The answer is dp[(1 << n) - 1]
    full_mask = (1 << n) - 1
    min_parts = dp[full_mask]
    
    # Now reconstruct the partition
    # We need to find which submasks were used
    # Store the choice made at each step
    choice = [0] * (1 << n)
    dp2 = [float('inf')] * (1 << n)
    dp2[0] = 0
    choice[0] = 0
    
    for mask in range(1, 1 << n):
        sub = mask
        while sub:
            if valid_mask[sub]:
                prev = mask ^ sub
                if dp2[prev] + 1 < dp2[mask]:
                    dp2[mask] = dp2[prev] + 1
                    choice[mask] = sub
            sub = (sub - 1) & mask
    
    # Reconstruct
    parts = []
    mask = full_mask
    while mask:
        sub = choice[mask]
        vertices = [i for i in range(n) if sub & (1 << i)]
        parts.append(vertices)
        mask ^= sub
    
    # Convert to list of lists of ints
    result = [list(part) for part in parts]
    return result
# EVOLVE-BLOCK-END
