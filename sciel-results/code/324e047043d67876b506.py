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
        """
        Partition a split graph into cliques.
        
        Structure:
        - Vertices 0..k-1 form a clique (C)
        - Vertices k..n-1 form an independent set (I)
        - Cross edges exist with probability p
        
        Strategy:
        1. The main clique C covers all edges within C.
        2. For cross edges, we group them by the independent vertex.
           For each independent vertex v, let N(v) be its neighbors in C.
           Since I is independent, {v} ∪ N(v) is a clique.
           We can cover all cross edges incident to v with one clique {v} ∪ N(v).
           
        This gives:
        - 1 clique for C (if k >= 2)
        - At most (n-k) cliques for cross edges (one per independent vertex with degree > 0)
        
        Total: at most 1 + (n-k) cliques.
        """
        cliques = []
        
        # 1. The main clique for the clique part
        if k >= 2:
            cliques.append(list(range(k)))
        
        # 2. For each independent vertex, create a clique with it and its neighbors in C
        #    This covers all cross edges incident to that vertex
        #    Since I is independent, no edges between I vertices, so no conflicts
        
        # Build adjacency from independent vertices to clique vertices
        # For each v in I, find all u in C such that (u,v) is an edge
        cross_edges_by_indep = {}
        for u, v in edges:
            if u < k and v >= k:
                if v not in cross_edges_by_indep:
                    cross_edges_by_indep[v] = []
                cross_edges_by_indep[v].append(u)
            elif u >= k and v < k:
                if u not in cross_edges_by_indep:
                    cross_edges_by_indep[u] = []
                cross_edges_by_indep[u].append(v)
        
        # For each independent vertex with at least one cross edge,
        # create a clique containing that vertex and all its neighbors in C
        for v in sorted(cross_edges_by_indep.keys()):
            neighbors = sorted(cross_edges_by_indep[v])
            if len(neighbors) >= 1:
                # The clique is {v} ∪ neighbors
                clique = neighbors + [v]
                # Ensure all are distinct ints
                clique = list(set(clique))
                if len(clique) >= 2:
                    cliques.append(clique)
        
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        if time.time() > deadline:
            break
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = partition_split_graph(n, k, edges)
        # Ensure types are Python int and list
        cleaned_cliques = []
        for c in cliques:
            cleaned_cliques.append([int(x) for x in c])
        results.append(cleaned_cliques)
    
    return results
# EVOLVE-BLOCK-END
