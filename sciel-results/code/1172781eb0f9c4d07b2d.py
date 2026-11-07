"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, k, p, seed):
        import random
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def clique_partition(edges, n, k):
        """
        Optimal clique partition for split graphs with high p.
        1. Use K = {0, ..., k-1} as the base clique.
        2. Find v* in I with the maximum number of neighbors in K.
        3. Form clique C1 = K U {v*}. This covers all edges in K and all edges (v*, u) for u in K.
        4. For all other v in I, form cliques {u, v} for each neighbor u in K.
        """
        if k < 2:
            # Degenerate case, though n=30, k=10 so k>=2
            return [[u, v] for u, v in edges]

        # Identify cross edges and build adjacency for I vertices
        cross_edges = []
        adj_in_k = {v: [] for v in range(k, n)}
        
        for u, v in edges:
            if u < k and v >= k:
                cross_edges.append((u, v))
                adj_in_k[v].append(u)

        # Find v* with max degree in K
        max_deg = -1
        v_star = -1
        for v in range(k, n):
            deg = len(adj_in_k[v])
            if deg > max_deg:
                max_deg = deg
                v_star = v

        cliques = []
        
        # 1. The main clique K U {v*}
        if v_star != -1 and max_deg > 0:
            cliques.append(list(range(k)) + [v_star])
        elif k >= 2:
            # If no cross edges, just K
            cliques.append(list(range(k)))
        else:
            # k < 2 handled above, but if k==1 and no edges?
            pass

        # 2. Cover remaining cross edges
        # We need to skip edges incident to v_star
        for u, v in cross_edges:
            if v == v_star:
                continue
            cliques.append([u, v])
            
        return cliques

    results = []
    for spec in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        n, k, p, seed = spec
        edges = instance(n, k, p, seed)
        cliques = clique_partition(edges, n, k)
        # Ensure types are correct (int, list)
        results.append(cliques)
        
    return results
# EVOLVE-BLOCK-END
