"""Clique partition of split graphs using structural properties."""
import time
from random import Random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def partition_split_graph(n, k, edges):
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Clique part: vertices 0..k-1
        # Independent set: vertices k..n-1
        
        # Strategy:
        # 1. One clique containing all clique-part vertices (0..k-1)
        # 2. For each independent-set vertex v, find the set of clique vertices adjacent to v
        #    That set forms a clique (since clique part is a clique)
        #    But we need to partition edges, not just cover them
        #    
        # Better strategy for edge partition:
        # - The clique on vertices 0..k-1 covers all edges within the clique part
        # - For cross edges (u,v) where u < k <= v: each such edge must be in exactly one clique
        #   A clique containing such an edge must contain u, v, and any other vertices all mutually adjacent
        #   Since v is in the independent set, v is only adjacent to some clique vertices
        #   The maximal clique containing v is {v} ∪ {u in 0..k-1 : (u,v) in edges}
        #   But we need to PARTITION edges, so each cross edge appears in exactly one clique
        #
        # Key insight: We can group independent-set vertices that share the same neighborhood
        # in the clique part. If vertices v1, v2 in independent set have the same set of adjacent
        # clique vertices N, then {v1, v2} ∪ N is not a clique (v1,v2 not adjacent).
        # But each individual {v} ∪ N_v is a clique.
        #
        # For edge partition: each cross edge (u, v) where u < k, v >= k must be in exactly one clique.
        # The clique {v} ∪ N_v contains all cross edges from v to its neighbors.
        # If we use one clique per independent-set vertex, we get n-k cliques for cross edges
        # plus 1 clique for the clique part = n-k+1 total.
        #
        # Can we do better? We can merge independent-set vertices into the same clique only if
        # they are adjacent to each other, but they're not (independent set).
        # However, we can potentially avoid the big clique on 0..k-1 and instead distribute
        # clique-part edges among other cliques.
        #
        # Alternative: For each clique vertex u, consider the set of independent vertices adjacent to u.
        # The set {u} ∪ {v >= k : (u,v) in edges} is a clique (since all v's are non-adjacent to each other
        # but all adjacent to u, and u is the only clique vertex).
        # Wait, that's not a clique unless all v's are mutually adjacent, which they're not.
        #
        # Let me reconsider. A clique in a split graph consists of some subset of the clique part
        # and some subset of the independent set, where every independent vertex is adjacent to
        # every clique vertex in the subset.
        #
        # So a clique is determined by (S, T) where S ⊆ {0..k-1}, T ⊆ {k..n-1}, and every t in T
        # is adjacent to every s in S.
        #
        # For edge partition, we need every edge to appear in exactly one such clique.
        #
        # Simple approach: 
        # - Clique C0 = {0, 1, ..., k-1} covers all clique-part edges
        # - For each v in {k..n-1}, clique Cv = {v} ∪ N(v) where N(v) = {u in 0..k-1 : (u,v) in edges}
        #   This covers all cross edges from v
        # Total: 1 + (n-k) cliques
        #
        # This is valid and simple. Let's compute it.
        
        cliques = []
        
        # Clique on the clique part
        if k >= 2:
            cliques.append(list(range(k)))
        
        # For each independent vertex, form a clique with its neighbors in the clique part
        for v in range(k, n):
            neighbors = sorted([u for u in range(k) if u in adj[v]])
            if neighbors:
                clique = [v] + neighbors
                if len(clique) >= 2:
                    cliques.append(clique)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        results.append(cliques)
    
    return results
# EVOLVE-BLOCK-END
