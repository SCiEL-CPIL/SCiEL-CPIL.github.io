"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        cross_edges = []
        for v in range(k, n):
            for u in range(k):
                if rng.random() < p:
                    cross_edges.append((u, v))
        return sorted(edges + cross_edges)

    def clique_partition(edges, n, k):
        # 1. Identify cross edges and build adjacency for I vertices
        # neighbors_in_k[v] = set of u in K such that (u, v) is an edge
        neighbors_in_k = {v: set() for v in range(k, n)}
        cross_edges_list = []
        
        for u, v in edges:
            if u < k and v >= k:
                cross_edges_list.append((u, v))
                neighbors_in_k[v].add(u)
            elif u >= k and v < k:
                # Just in case, though generator produces u<k, v>=k
                cross_edges_list.append((v, u))
                neighbors_in_k[u].add(v)

        cliques = []
        
        # 2. Determine if we can use a large mixed clique K U {v}
        # This requires v to be connected to all u in K.
        best_v = None
        for v in range(k, n):
            if len(neighbors_in_k[v]) == k:
                best_v = v
                break

        if best_v is not None:
            # Use K U {best_v} as the first clique.
            # This covers all internal edges of K and all cross edges to best_v.
            cliques.append(list(range(k)) + [best_v])
            # Remove best_v from consideration for individual I-cliques
            # We will handle other v's next
        else:
            # Use K as the first clique.
            # This covers all internal edges of K.
            if k >= 2:
                cliques.append(list(range(k)))
            elif k == 1:
                # K is a single vertex, no internal edges. 
                # But we still need to cover cross edges.
                # The logic below handles I vertices.
                pass
            else:
                # k=0, no K
                pass

        # 3. Handle each vertex v in I
        # For each v, the cross edges are (u, v) for u in neighbors_in_k[v].
        # We can cover all these edges with a single clique: neighbors_in_k[v] U {v}
        # This is valid because K is a clique, so any subset of K is a clique.
        # And v is connected to all u in neighbors_in_k[v] by definition.
        
        for v in range(k, n):
            if v == best_v:
                continue # Already covered by the large clique
            
            nbrs = neighbors_in_k[v]
            if nbrs:
                # Create the clique: sorted list of neighbors + v
                # Note: The problem requires cliques to be lists of ints.
                # It does not require them to be sorted, but it's good practice.
                clique_v = list(nbrs) + [v]
                # Ensure at least 2 vertices
                if len(clique_v) >= 2:
                    cliques.append(clique_v)
            # If nbrs is empty, v is isolated in the cross graph, 
            # so it has no edges to cover. No clique needed for v.

        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition(edges, n, k)
        
        # Validation step to ensure no errors
        # Check types
        for c in cliques:
            if not isinstance(c, list):
                c = list(c)
            for x in c:
                if not isinstance(x, int):
                    x = int(x)
        
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
