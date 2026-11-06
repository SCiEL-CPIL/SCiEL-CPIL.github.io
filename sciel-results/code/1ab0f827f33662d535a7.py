import time
import random
from itertools import combinations
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def find_maximal_cliques(n, adj):
        """Find all maximal cliques using Bron-Kerbosch with pivot."""
        cliques = []
        
        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    cliques.append(frozenset(R))
                return
            
            # Choose pivot
            if P:
                pivot = max(P, key=lambda v: len(adj[v] & (P | X)))
                candidates = P - adj[pivot]
            else:
                candidates = X
            
            for v in list(candidates):
                bronk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bronk(set(), set(range(n)), set())
        return cliques

    def solve(edges, time_budget):
        if not edges:
            return []
        
        n = max(max(e) for e in edges) + 1
        edge_set = set()
        for u, v in edges:
            edge_set.add((u, v) if u < v else (v, u))
            
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all maximal cliques
        maximal_cliques = find_maximal_cliques(n, adj)
        
        if not maximal_cliques:
            # Fallback: each edge is its own clique
            return [list(e) for e in edge_set]
        
        # For each maximal clique, compute its edge set
        clique_edges = []
        for mc in maximal_cliques:
            es = set()
            for u, v in combinations(sorted(mc), 2):
                if (u, v) in edge_set or (v, u) in edge_set:
                    es.add((u, v) if u < v else (v, u))
            if es:
                clique_edges.append(es)
        
        # Greedy set cover: repeatedly pick the clique that covers the most uncovered edges
        remaining = set(edge_set)
        cliques_chosen = []
        
        while remaining:
            if time.time() > start_time + time_budget:
                # Fallback: add remaining edges as individual cliques
                for e in remaining:
                    cliques_chosen.append(list(e))
                break
            
            best_idx = -1
            best_count = 0
            for i, ce in enumerate(clique_edges):
                count = len(ce & remaining)
                if count > best_count:
                    best_count = count
                    best_idx = i
            
            if best_idx == -1 or best_count == 0:
                # No clique covers any remaining edge, add edges individually
                for e in remaining:
                    cliques_chosen.append(list(e))
                break
            
            cliques_chosen.append(list(maximal_cliques[best_idx]))
            remaining -= clique_edges[best_idx]
        
        return cliques_chosen

    # Split time budget between the two graphs
    c1 = solve(random_graph(32, 0.6, 601), 70)
    c2 = solve(random_graph(64, 0.55, 602), 70)
    
    return [c1, c2]
