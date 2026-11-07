"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import product

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    
    # Adjacency as sets
    adj = [set() for _ in range(n)]
    for e in P['edges']:
        adj[e[0]].add(e[1])
        adj[e[1]].add(e[0])
    
    # Bitmask adjacency for fast clique/independent checks
    adj_mask = [0] * n
    for i in range(n):
        for j in adj[i]:
            adj_mask[i] |= (1 << j)
    
    # Non-adjacency mask (including self) for independent set checks
    full_mask = (1 << n) - 1
    non_adj_mask = [full_mask ^ adj_mask[i] for i in range(n)]
    
    deadline = time.time() + 55  # safety margin before 90s kill
    
    # Order vertices by degree descending (more constrained first)
    order = sorted(range(n), key=lambda v: -len(adj[v]))
    
    # Memoization for subproblems: (remaining_mask, k_parts_left) -> True/False
    # But we need to track which types are available, which is complex.
    # Instead, we'll use a different memoization: for a given set of remaining vertices,
    # what is the minimum number of parts needed?
    # min_parts[mask] = minimum number of parts to cover 'mask'
    
    # Precompute all maximal cliques and maximal independent sets
    # A set is a clique if for all v in set, (adj_mask[v] & set_without_v) == set_without_v
    # A set is an independent set if for all v in set, (adj_mask[v] & set_without_v) == 0
    
    # For n=14, there are 2^14 = 16384 possible subsets.
    # We can precompute is_clique[mask] and is_independent[mask] for all masks.
    
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    
    # Base case: empty set is both
    is_clique[0] = True
    is_independent[0] = True
    
    # Single vertices are both
    for v in range(n):
        is_clique[1 << v] = True
        is_independent[1 << v] = True
    
    # Compute for all masks
    # A mask is a clique if it can be formed by adding a vertex v to a clique
    # where v is adjacent to all other vertices in the clique.
    # More efficiently: a mask is a clique if for any v in mask, (mask ^ (1<<v)) is a clique
    # and v is adjacent to all in (mask ^ (1<<v)).
    # Actually, simpler: mask is a clique iff for all v in mask, (adj_mask[v] & mask) == mask (excluding self, but adj_mask doesn't have self)
    # Wait, adj_mask[v] doesn't include v. So (adj_mask[v] & mask) should equal mask without v.
    # Let's just check directly for each mask.
    
    for mask in range(1, 1 << n):
        # Find lowest set bit
        v = (mask & -mask).bit_length() - 1
        rest = mask ^ (1 << v)
        
        # Check if v is adjacent to all in rest
        if (adj_mask[v] & rest) == rest:
            is_clique[mask] = is_clique[rest]
        else:
            is_clique[mask] = False
        
        # Check if v is non-adjacent to all in rest
        if (adj_mask[v] & rest) == 0:
            is_independent[mask] = is_independent[rest]
        else:
            is_independent[mask] = False
    
    # Now, min_parts[mask] = minimum number of parts to partition 'mask'
    # where each part is either a clique or an independent set.
    # min_parts[0] = 0
    # min_parts[mask] = min over all subsets s of mask where is_clique[s] or is_independent[s]
    #                   of (1 + min_parts[mask ^ s])
    
    # But this is O(3^n) which is too slow for n=14 (3^14 ~ 4.8M, might be okay in optimized code, but let's be careful).
    # Actually, we can use a smarter approach:
    # For each mask, try all maximal cliques and maximal independent sets contained in mask.
    
    # Let's precompute the list of all cliques and all independent sets.
    # Actually, we just need to know for each mask, what are the valid parts (subsets) we can use.
    
    # Alternative: use the backtracking approach but with memoization on (remaining_mask, k)
    # where k is the number of parts we are allowed to use.
    # But the type of each part matters.
    
    # Let's stick with the original backtracking but add memoization.
    # State: (idx, parts_tuple, types_tuple) -> too complex.
    
    # Better: Let's compute min_parts[mask] using DP.
    # min_parts[mask] = min(1 + min_parts[mask ^ s] for s in valid_subsets(mask))
    # where valid_subsets(mask) = {s subset of mask : is_clique[s] or is_independent[s]}
    
    # To speed this up, for each mask, we can iterate over its subsets.
    # Number of subsets of mask is 2^(popcount(mask)).
    # Total work is sum over all masks of 2^(popcount(mask)) = 3^n.
    # For n=14, 3^14 = 4,782,969. This is feasible.
    
    INF = float('inf')
    min_parts = [INF] * (1 << n)
    min_parts[0] = 0
    min_parts_choice = [0] * (1 << n)  # stores the subset s that achieves the minimum
    
    # We need to iterate masks in increasing order of popcount
    # Or simply iterate from 1 to 2^n - 1, since mask ^ s is always a proper subset (smaller value)
    # Wait, mask ^ s < mask is not always true. E.g., mask=3 (11), s=1 (01), mask^s=2 (10) < 3.
    # But mask=3 (11), s=2 (10), mask^s=1 (01) < 3.
    # Actually, s is a non-empty subset of mask, so mask ^ s < mask is NOT guaranteed.
    # Example: mask = 0b1000 (8), s = 0b0001 (1) is not a subset.
    # If s is a subset of mask, then mask ^ s = mask - s (in terms of set difference).
    # The numerical value mask ^ s is always < mask?
    # Let mask = 0b1010 (10), s = 0b0010 (2). mask ^ s = 0b1000 (8) < 10.
    # Let mask = 0b1001 (9), s = 0b0001 (1). mask ^ s = 0b1000 (8) < 9.
    # Yes, if s is a non-empty subset of mask, then the highest set bit of s is also set in mask.
    # Removing it from mask will reduce the value.
    # So mask ^ s < mask always holds for non-empty s subset of mask.
    # Therefore, iterating from 1 to 2^n - 1 in increasing order works.
    
    for mask in range(1, 1 << n):
        best = INF
        best_s = 0
        # Iterate over all non-empty subsets s of mask
        s = mask
        while s:
            if is_clique[s] or is_independent[s]:
                val = 1 + min_parts[mask ^ s]
                if val < best:
                    best = val
                    best_s = s
            s = (s - 1) & mask
        
        min_parts[mask] = best
        min_parts_choice[mask] = best_s
    
    # The answer for the full graph is min_parts[full_mask]
    # We need to reconstruct the partition.
    
    if min_parts[full_mask] == INF:
        # Should not happen, fallback
        return [[v] for v in range(n)]
    
    # Reconstruct the partition
    parts = []
    mask = full_mask
    while mask:
        s = min_parts_choice[mask]
        # Convert s to list of vertices
        verts = [v for v in range(n) if (s >> v) & 1]
        parts.append(verts)
        mask ^= s
    
    # Ensure all parts are non-empty and return as list of lists
    final = []
    for p in parts:
        if p:
            final.append(list(p))
    
    return final

# EVOLVE-BLOCK-END
