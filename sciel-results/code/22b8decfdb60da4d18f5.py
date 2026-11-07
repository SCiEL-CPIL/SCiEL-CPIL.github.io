# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(edges, n, k):
        cliques = []
        
        # Identify cross edges and build neighborhoods for I
        # K = {0, ..., k-1}, I = {k, ..., n-1}
        # We need to know N_K(v) for each v in I.
        
        # Initialize neighborhoods
        neighborhoods = {v: set() for v in range(k, n)}
        
        # Process edges to find cross edges
        for u, v in edges:
            if u < k and v >= k:
                neighborhoods[v].add(u)
            elif v < k and u >= k:
                neighborhoods[u].add(v)
            # Edges within K or within I (none) are ignored for this step
        
        # 1. Create cliques for each v in I: N_K(v) U {v}
        covered_internal_edges = set()
        
        for v in range(k, n):
            N_v = neighborhoods[v]
            if N_v:
                clique = list(N_v) + [v]
                cliques.append(clique)
                # Mark internal edges covered by this clique
                N_v_list = list(N_v)
                for i in range(len(N_v_list)):
                    for j in range(i + 1, len(N_v_list)):
                        covered_internal_edges.add((min(N_v_list[i], N_v_list[j]), max(N_v_list[i], N_v_list[j])))
            else:
                # If v has no neighbors in K, it is isolated? 
                # But task says split graph, usually connected or p>0.
                # If N_v is empty, we can't form a clique of size >= 2 with v and K.
                # But v must be in a clique.
                # If v is isolated, no edges to cover. But clique must have >= 2 vertices.
                # This case shouldn't happen with p=0.7.
                pass

        # 2. Cover remaining internal edges of K
        # All edges in K
        all_k_edges = set()
        for u in range(k):
            for v in range(u + 1, k):
                all_k_edges.add((u, v))
        
        remaining_edges = all_k_edges - covered_internal_edges
        
        # Partition remaining_edges into cliques
        # Simple strategy: Greedy maximal clique
        # Since k is small, we can just pick edges one by one if needed, 
        # but we can do better by grouping.
        
        # For simplicity and safety, if remaining_edges is not empty,
        # we can cover them with single edges (cliques of size 2).
        # This is valid but might not be minimal.
        # However, given the huge reduction from cross edges, this is likely sufficient.
        
        for u, v in remaining_edges:
            cliques.append([u, v])
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition(edges, n, k)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
