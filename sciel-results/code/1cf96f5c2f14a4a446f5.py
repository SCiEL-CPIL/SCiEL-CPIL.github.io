"""Rewritten program with a structural change: clique partition via maximum cliques + set cover."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = 150.0 - 5.0  # safety margin

    def instance(n, k, p, seed):
        from random import Random
        rng = Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    def build_graph(n, edges):
        """Build adjacency sets from edge list."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_max_cliques(adj, n):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bron_kerbosch(R, P, X):
            if time.time() > deadline:
                return
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(tuple(sorted(R)))
                return
            
            # Choose pivot u from P ∪ X to minimize branching
            if P:
                pivot_candidates = P
            else:
                pivot_candidates = X
            
            if pivot_candidates:
                # Pick the vertex with maximum degree in P
                pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P))
                candidates = [v for v in P if v not in adj[pivot]]
            else:
                candidates = list(P)
            
            for v in candidates:
                new_R = R | {v}
                new_P = P & adj[v]
                new_X = X & adj[v]
                bron_kerbosch(new_R, new_P, new_X)
                P.remove(v)
                X.add(v)
        
        bron_kerbosch(set(), set(range(n)), set())
        return cliques

    def greedy_set_cover(cliques, edges):
        """Greedy set cover: repeatedly pick clique covering most uncovered edges."""
        # Map each edge to which cliques contain it
        edge_set = set(edges)
        # For each clique, compute which edges it covers
        clique_edges = []
        for clique in cliques:
            if len(clique) < 2:
                clique_edges.append(frozenset())
            else:
                es = frozenset((min(a,b), max(a,b)) for a,b in itertools.combinations(clique, 2))
                clique_edges.append(es & edge_set)
        
        uncovered = set(edges)
        selected = []
        remaining_cliques = list(range(len(cliques)))
        
        while uncovered and remaining_cliques:
            # Find clique covering most uncovered edges
            best_idx = -1
            best_count = 0
            for i in remaining_cliques:
                count = len(clique_edges[i] & uncovered)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                break
            
            selected.append(cliques[best_idx])
            uncovered -= clique_edges[best_idx]
            remaining_cliques.remove(best_idx)
        
        return selected

    def try_partition(n, k, p, seed):
        """Attempt to find a valid clique partition for one instance."""
        edges = instance(n, k, p, seed)
        edge_set = set((min(u,v), max(u,v)) for u,v in edges)
        adj = build_graph(n, edges)
        
        # Find maximal cliques
        cliques = find_max_cliques(adj, n)
        
        if not cliques:
            # Fallback: use edges as cliques (trivial valid partition)
            return [[u, v] for u, v in edges]
        
        # Greedy set cover
        partition = greedy_set_cover(cliques, edge_set)
        
        # Verify: every edge must be in exactly one clique
        # Check validity
        edge_cover_count = {}
        for clique in partition:
            if len(clique) < 2:
                continue
            for a, b in itertools.combinations(clique, 2):
                e = (min(a,b), max(a,b))
                if e in edge_set:
                    edge_cover_count[e] = edge_cover_count.get(e, 0) + 1
        
        # Every edge must be covered exactly once
        for e in edge_set:
            if e not in edge_cover_count or edge_cover_count[e] != 1:
                # Fallback to trivial partition
                return [[u, v] for u, v in edges]
        
        # Also verify no clique contains a non-edge
        for clique in partition:
            for a, b in itertools.combinations(clique, 2):
                e = (min(a,b), max(a,b))
                if e not in edge_set:
                    return [[u, v] for u, v in edges]
        
        return partition

    results = []
    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    
    for spec in specs:
        if time.time() > deadline:
            # Not enough time; use trivial partition for remaining
            n, k, p, seed = spec
            edges = instance(n, k, p, seed)
            results.append([[u, v] for u, v in edges])
            continue
        partition = try_partition(*spec)
        results.append(partition)
    
    # Ensure all are Python int lists
    final = []
    for part in results:
        final.append([list(c) for c in part])
    
    return final
# EVOLVE-BLOCK-END
