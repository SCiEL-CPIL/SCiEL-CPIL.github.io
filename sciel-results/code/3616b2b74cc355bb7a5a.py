"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = P['n']
    edges = P['edges']
    min_v = P['minimum_vertices']
    
    # Build adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute neighbor count for each vertex (global degree)
    deg = [bin(adj[v]).count('1') for v in range(n)]
    
    # Deadline guard: allow ~50s of work, safety margin before 60s limit
    deadline = time.time() + 50.0
    
    best_score = 0
    best_subset = list(range(min_v))
    
    # Try to find a subset with more distinct degrees than the seed
    # Seed returns list(range(9)) = [0..8], which has some number of distinct degrees.
    # We want to maximize distinct degrees in the induced subgraph.
    
    # First, evaluate the seed to know the baseline
    def evaluate(subset):
        """Return the number of distinct degrees in the induced subgraph on subset."""
        s = set(subset)
        degrees = []
        for v in subset:
            # Count neighbors of v that are in subset
            d = 0
            a = adj[v]
            for u in subset:
                if u != v and (a >> u) & 1:
                    d += 1
            degrees.append(d)
        return len(set(degrees))
    
    seed_score = evaluate(best_subset)
    best_score = seed_score
    
    # Exhaustive search over all subsets of size 8 to 17
    # C(17,8) = 24310, C(17,9)=11440, etc. Total is manageable.
    # But we can be smarter: first try sizes that might give more distinct degrees.
    # With 8 vertices, max distinct degrees is 8 (if all degrees are different).
    # With more vertices, max is the number of vertices.
    
    # Let's enumerate all subsets from size 8 to 17, but with a deadline.
    # For efficiency, use bitmask representation and vectorized evaluation where possible.
    
    # Precompute for each vertex its adjacency as a bitmask
    # For a given subset (as bitmask), compute degrees quickly.
    
    # Let's use a smarter approach: iterate over combinations and compute degrees.
    # Since n=17, the total number of subsets of size >= 8 is:
    # sum_{k=8}^{17} C(17,k) = 2^17 - sum_{k=0}^{7} C(17,k)
    # sum_{k=0}^{7} C(17,k) = 1+17+136+680+2380+6188+12376+19448 = 41226
    # 2^17 = 131072
    # So total = 131072 - 41226 = 89846 subsets. This is very manageable.
    
    # But we can do even better: for each subset, we need to compute degrees.
    # Let's precompute a lookup or use bit operations.
    
    # Alternative: for each subset bitmask, compute degrees for all vertices in the subset.
    # deg_in_subset[v] = popcount(adj[v] & subset_mask)
    
    # Let's iterate over all masks with popcount >= 8.
    # Generate all masks from 0 to 2^17 - 1, filter by popcount >= 8.
    
    # To be efficient, let's iterate by size.
    for k in range(min_v, n + 1):
        if time.time() > deadline:
            break
        for subset in itertools.combinations(range(n), k):
            if time.time() > deadline:
                break
            s = set(subset)
            degrees = []
            for v in subset:
                d = 0
                a = adj[v]
                for u in subset:
                    if u != v and (a >> u) & 1:
                        d += 1
                degrees.append(d)
            score = len(set(degrees))
            if score > best_score:
                best_score = score
                best_subset = list(subset)
                if best_score == k:  # Can't do better than k distinct degrees for k vertices
                    return best_subset
    
    return best_subset
# EVOLVE-BLOCK-END
