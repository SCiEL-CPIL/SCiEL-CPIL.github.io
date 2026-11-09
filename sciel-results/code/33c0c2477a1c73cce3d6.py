"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from math import isqrt

P = {'n': 256}

def construct():
    N = P['n']
    vertices = list(range(2, N + 1))
    n = len(vertices)
    
    # Precompute divisibility: divs[x] is list of y in vertices where x divides y
    # Actually we need for triple {a,b,c}: check if a|bc, b|ac, or c|ab
    # Precompute products? No, bc can be up to 256*256 = 65536, which is fine.
    
    # Build hyperedges: for each triple (i,j,k) with i<j<k, check if it's forbidden
    # Forbidden if any of the three divides the product of the other two.
    
    # To make this faster, let's precompute for each pair (x,y) the set of z such that z divides x*y
    # Or: for each triple, check the three conditions.
    
    # Number of triples: C(255,3) ~ 2.7M. Each check is 3 modulo operations. Should be fine in 60s.
    
    # Let's build the conflict hypergraph as a list of hyperedges (each is a tuple of 3 vertex indices)
    # Then we'll do a greedy maximum independent set via branch and bound or greedy removal.
    
    # Actually, let's use a simpler but effective approach:
    # Greedy: repeatedly pick the vertex that is in the most remaining hyperedges, remove it and all its hyperedges.
    
    # Build hyperedges
    hyperedges = []
    # For each pair (x,y), compute x*y, then for each z, check if z divides x*y
    # This is more efficient: for each pair (x,y), find all z that divide x*y, and those form hyperedges with (x,y,z)
    
    # Let's precompute divisors for numbers up to N*N
    max_prod = N * N
    # divisors[m] = list of d in [2, N] such that d divides m
    divisors = [[] for _ in range(max_prod + 1)]
    for d in range(2, N + 1):
        for m in range(d, max_prod + 1, d):
            divisors[m].append(d)
    
    # Now for each pair (x, y) with x < y, compute p = x*y
    # For each z in divisors[p] where z != x and z != y, we have a forbidden triple
    # We need to store these as hyperedges (sets of 3 vertices)
    
    # Use a set to avoid duplicate hyperedges
    # Each hyperedge is a frozenset of 3 vertex values, or we can use sorted tuples
    
    hyperedge_set = set()
    for x_idx, x in enumerate(vertices):
        for y_idx in range(x_idx + 1, n):
            y = vertices[y_idx]
            p = x * y
            for z in divisors[p]:
                if z == x or z == y:
                    continue
                # z is in vertices (it's in [2, N])
                # Create sorted triple
                triple = tuple(sorted((x, y, z)))
                hyperedge_set.add(triple)
    
    hyperedges = list(hyperedge_set)
    
    # Now we have the hypergraph. We want the maximum independent set.
    # Greedy approach: iteratively remove the vertex that appears in the most hyperedges.
    
    # Map vertex value to index
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # degree[v] = number of hyperedges containing v
    degree = [0] * n
    for he in hyperedges:
        for v in he:
            idx = vertex_to_idx[v]
            degree[idx] += 1
    
    remaining = set(range(n))
    removed_hyperedges = set()
    
    # We'll track which hyperedges are still active
    active_hyperedges = list(range(len(hyperedges)))
    
    # To efficiently find the max-degree vertex and update degrees when removing hyperedges,
    # let's use a different approach: just iterate through hyperedges and count.
    
    # Simpler: just do greedy removal
    active_vertices = set(vertices)
    active_hedge_set = set(hyperedges)
    
    while active_hedge_set:
        # Find vertex in active_vertices that is in the most active hyperedges
        # This is O(n * |H|) per iteration, which might be slow.
        # Let's precompute for each vertex its list of hyperedges.
        pass
    
    # Let's use a more efficient method:
    # vertex_hedges[v] = list of hyperedges (as indices) containing v
    vertex_hedges = {v: [] for v in vertices}
    for idx, he in enumerate(hyperedges):
        for v in he:
            vertex_hedges[v].append(idx)
    
    active_hedges = [True] * len(hyperedges)
    active_vertices_set = set(vertices)
    
    while True:
        # Check if all hyperedges are inactive
        has_active = False
        for idx in range(len(hyperedges)):
            if active_hedges[idx]:
                has_active = True
                break
        if not has_active:
            break
        
        # Find the active vertex with the most active hyperedges
        best_v = None
        best_count = -1
        for v in active_vertices_set:
            count = 0
            for he_idx in vertex_hedges[v]:
                if active_hedges[he_idx]:
                    count += 1
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v is None or best_count == 0:
            break
        
        # Remove best_v and all its hyperedges
        active_vertices_set.discard(best_v)
        for he_idx in vertex_hedges[best_v]:
            if active_hedges[he_idx]:
                active_hedges[he_idx] = False
                # We don't need to update other vertices' counts since we recompute each time
    
    result = list(active_vertices_set)
    result.sort()
    
    # Ensure types are correct
    return [int(x) for x in result]
# EVOLVE-BLOCK-END
