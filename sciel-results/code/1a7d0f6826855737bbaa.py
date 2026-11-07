"""Rewritten program using maximum clique extraction via Bron-Kerbosch on complement."""
# EVOLVE-BLOCK-START
def construct():
    from random import Random

    def instance(n, k, p, seed):
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def max_cliques_cover(n, adj):
        # Greedy: repeatedly find a maximum clique in the remaining subgraph, add it, remove its edges
        # We track which edges are still uncovered.
        # For efficiency, maintain a set of active vertices (those with at least one uncovered edge).
        # But simpler: maintain the set of uncovered edges, and on each iteration find max clique
        # in the subgraph induced by vertices incident to uncovered edges.
        
        # Precompute all edges as a frozenset for fast lookup
        all_edges_set = set()
        for u, v in adj[0]:
            pass
        # Rebuild edge set from adj
        edge_set = set()
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edge_set.add((u, v))
        
        remaining_edges = set(edge_set)
        cliques = []
        
        # For Bron-Kerbosch, we need adjacency. We'll work on the subgraph of vertices that
        # are endpoints of remaining edges.
        
        # Helper: Bron-Kerbosch with pivot to find max clique in a graph defined by adj and a vertex set
        def bron_kerbosch_max(R, P, X, adj, n, best):
            # best is a mutable dict with 'size' and 'clique'
            if len(P) + len(X) <= best['size']:
                return
            if not P:
                if len(R) > best['size']:
                    best['size'] = len(R)
                    best['clique'] = list(R)
                return
            # Choose pivot u from P ∪ X that maximizes |N(u) ∩ P|
            P_list = list(P)
            X_list = list(X)
            candidates = P_list + X_list
            best_u = None
            best_count = -1
            for u in candidates:
                cnt = sum(1 for v in P if v in adj[u])
                if cnt > best_count:
                    best_count = cnt
                    best_u = u
            # If P is subset of N(u), we can prune
            if best_u is not None and all(v in adj[best_u] for v in P):
                return
            # Iterate over P \ N(u)
            P_without_N = set(v for v in P if v not in adj[best_u])
            for v in P_without_N:
                P_new = P - {v}
                X_new = X - {v}
                R_new = R | {v}
                # Only recurse if v can extend
                P_intersect_N = {w for w in P_new if w in adj[v]}
                X_intersect_N = {w for w in X_new if w in adj[v]}
                bron_kerbosch_max(R_new, P_intersect_N, X_intersect_N, adj, n, best)
                P.remove(v)
                X.add(v)
        
        while remaining_edges:
            # Find vertices involved in remaining edges
            active_verts = set()
            for u, v in remaining_edges:
                active_verts.add(u)
                active_verts.add(v)
            
            if not active_verts:
                break
            
            # Find max clique in the subgraph induced by active_verts
            # Build subgraph adjacency
            # We'll use the original adj but restrict to active_verts
            # For Bron-Kerbosch, P = active_verts, X = empty
            best = {'size': 0, 'clique': []}
            # Sort P for determinism
            P_init = frozenset(active_verts)
            bron_kerbosch_max(frozenset(), P_init, frozenset(), adj, n, best)
            
            if best['size'] < 2:
                # Should not happen if remaining_edges is non-empty, but guard
                break
            
            cliques.append(sorted(best['clique']))
            
            # Remove all edges within the clique from remaining_edges
            clique_set = set(best['clique'])
            new_remaining = set()
            for u, v in remaining_edges:
                if u in clique_set and v in clique_set:
                    continue
                new_remaining.add((u, v))
            remaining_edges = new_remaining
        
        return cliques

    results = []
    for (n, k, p, seed) in ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303)):
        edges = instance(n, k, p, seed)
        adj = build_adj(n, edges)
        cliques = max_cliques_cover(n, adj)
        # Ensure each clique is a list of ints with at least 2 vertices
        cleaned = []
        for c in cliques:
            if len(c) >= 2:
                cleaned.append([int(x) for x in c])
        results.append(cleaned)
    
    return results
# EVOLVE-BLOCK-END
