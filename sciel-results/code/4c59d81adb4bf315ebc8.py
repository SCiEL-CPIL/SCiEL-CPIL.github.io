"""Rewritten program using bitmask-based exact search for minimum cochromatic partition."""
# EVOLVE-BLOCK-START
import sys
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    # Build adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute all maximal cliques and maximal independent sets as bitmasks
    # A set S (bitmask) is a clique if for all u,v in S, u is adjacent to v
    # A set S is an independent set if for all u,v in S, u is NOT adjacent to v
    
    def is_clique(mask):
        # Check if vertices in mask form a clique
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            rest = m & ~lsb
            if rest & ~adj[v]:
                return False
            m = rest
        return True
    
    def is_independent(mask):
        # Check if vertices in mask form an independent set
        m = mask
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            rest = m & ~lsb
            if rest & adj[v]:
                return False
            m = rest
        return True
    
    # Generate all maximal cliques
    # Use Bron-Kerbosch style enumeration or just enumerate all subsets and check maximality
    # Since n=14, we can enumerate all 2^14 = 16384 subsets
    all_cliques = []
    all_independent = []
    
    for mask in range(1, 1 << n):
        if is_clique(mask):
            # Check if maximal: no vertex outside can be added
            # A vertex v outside mask can be added to clique if v is adjacent to all in mask
            outside = (1 << n) - 1 - mask
            maximal = True
            m = outside
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                # v can join clique if adj[v] contains all bits of mask
                if (adj[v] & mask) == mask:
                    maximal = False
                    break
                m = rest = m & ~lsb
            if maximal:
                all_cliques.append(mask)
        if is_independent(mask):
            # Check if maximal: no vertex outside can be added
            # A vertex v outside mask can be added to independent set if v is NOT adjacent to any in mask
            outside = (1 << n) - 1 - mask
            maximal = True
            m = outside
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                # v can join independent set if adj[v] & mask == 0
                if (adj[v] & mask) == 0:
                    maximal = False
                    break
                m = m & ~lsb
            if maximal:
                all_independent.append(mask)
    
    # For efficiency in search, we want to try larger parts first
    all_cliques.sort(key=lambda m: -bin(m).count('1'))
    all_independent.sort(key=lambda m: -bin(m).count('1'))
    
    # We'll use DP with memoization: dp(mask) = minimum number of parts to cover mask
    # where each part is a clique or independent set
    # dp(0) = 0
    # dp(mask) = 1 + min(dp(mask ^ part)) for all valid parts that are subsets of mask
    
    sys.setrecursionlimit(10000)
    
    memo = {}
    
    def solve(mask):
        if mask == 0:
            return 0
        if mask in memo:
            return memo[mask]
        
        best = n + 1  # upper bound
        
        # Try all maximal cliques that are subsets of mask
        for part in all_cliques:
            if (part & mask) == part:
                rest = mask ^ part
                val = 1 + solve(rest)
                if val < best:
                    best = val
                    if best == 1:
                        break
        
        if best == 1:
            memo[mask] = best
            return best
        
        # Try all maximal independent sets that are subsets of mask
        for part in all_independent:
            if (part & mask) == part:
                rest = mask ^ part
                val = 1 + solve(rest)
                if val < best:
                    best = val
                    if best == 1:
                        break
        
        memo[mask] = best
        return best
    
    full_mask = (1 << n) - 1
    min_parts = solve(full_mask)
    
    # Now reconstruct the actual partition
    # We need to find which parts achieve min_parts
    reconstruction = {}
    
    def reconstruct(mask, remaining):
        if mask == 0:
            return []
        if mask in reconstruction:
            return reconstruction[mask]
        
        # Find a part that leads to optimal solution
        for part in all_cliques:
            if (part & mask) == part:
                rest = mask ^ part
                if 1 + solve(rest) == remaining:
                    result = [[v for v in range(n) if (part >> v) & 1]] + reconstruct(rest, remaining - 1)
                    reconstruction[mask] = result
                    return result
        
        for part in all_independent:
            if (part & mask) == part:
                rest = mask ^ part
                if 1 + solve(rest) == remaining:
                    result = [[v for v in range(n) if (part >> v) & 1]] + reconstruct(rest, remaining - 1)
                    reconstruction[mask] = result
                    return result
        
        # Should not reach here if solve was correct
        return []
    
    partition = reconstruct(full_mask, min_parts)
    
    # Ensure types are correct: list of lists of ints
    result = []
    for part in partition:
        result.append([int(v) for v in part])
    
    return result

# EVOLVE-BLOCK-END
