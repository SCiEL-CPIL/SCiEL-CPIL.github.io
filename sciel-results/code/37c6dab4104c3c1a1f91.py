import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        # Use a set of edges for O(1) lookup and removal
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def get_adjacency(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_maximal_clique(adj, start_vertex, n):
        # Bron-Kerbosch with pivot
        # We want to find a maximal clique containing start_vertex
        # To bias towards larger cliques, we order candidates by degree
        
        def bk(R, P, X):
            if not P and not X:
                return list(R)
            
            # Choose pivot u from P union X with max degree in P
            P_union_X = P | X
            if not P_union_X:
                return None
            
            # Sort by degree in P to pick a good pivot
            # Actually, standard BK picks u to minimize |P \ N(u)|
            # We want to find a large clique, so maybe we should pick u such that P \ N(u) is small?
            # Or just pick the first one?
            # Let's pick u from P_union_X with max len(P & N(u)) to reduce branching?
            # Standard BK: pick u in P∪X. Recurse on P\N(u).
            # If we pick u with many neighbors in P, P\N(u) is small.
            
            best_u = None
            max_common = -1
            for u in P_union_X:
                common = len(P & adj[u])
                if common > max_common:
                    max_common = common
                    best_u = u
            
            if best_u is None:
                return None
                
            P_not_Nu = P - adj[best_u]
            
            # Order P_not_Nu by degree descending to find large cliques first
            candidates = sorted(list(P_not_Nu), key=lambda x: len(adj[x] & P), reverse=True)
            
            for v in candidates:
                if v in X:
                    continue
                # Recurse
                result = bk(R | {v}, P & adj[v], X & adj[v])
                if result is not None:
                    return result
                P.remove(v)
                X.add(v)
            
            return None

        # Initialize
        # R = {start_vertex}
        # P = N(start_vertex)
        # X = empty
        # We want a clique containing start_vertex.
        # So R starts with start_vertex.
        # P should be neighbors of start_vertex.
        # X should be empty.
        
        R = {start_vertex}
        P = set(adj[start_vertex])
        X = set()
        
        # We need to modify bk to handle the initial state correctly or just call it.
        # The standard bk finds cliques in P.
        # If we set R={start}, P=N(start), X={}, it finds cliques containing start.
        
        return bk(R, P, X)

    def greedy_clique_partition(n, edges):
        if not edges:
            return []
        
        adj = get_adjacency(n, edges)
        remaining_edges = set(edges)
        cliques = []
        
        # To speed up, we can track degrees
        degrees = [len(adj[i]) for i in range(n)]
        
        while remaining_edges:
            if time.time() > deadline:
                break
            
            # Find vertex with max degree
            # This is O(n) if we just scan
            max_deg_vertex = -1
            max_deg = -1
            for i in range(n):
                if degrees[i] > max_deg:
                    max_deg = degrees[i]
                    max_deg_vertex = i
            
            if max_deg_vertex == -1:
                break
            
            # Find a maximal clique containing max_deg_vertex
            clique = find_maximal_clique(adj, max_deg_vertex, n)
            
            if not clique:
                # Fallback: take a single edge
                edge = next(iter(remaining_edges))
                clique = list(edge)
            
            # Remove edges of this clique
            clique_set = set(clique)
            edges_to_remove = set()
            for u in clique:
                for v in clique:
                    if u < v:
                        e = (u, v)
                        if e in remaining_edges:
                            edges_to_remove.add(e)
            
            if not edges_to_remove:
                # Should not happen if clique is valid, but safety
                break
            
            remaining_edges -= edges_to_remove
            
            # Update adj and degrees
            for u, v in edges_to_remove:
                adj[u].remove(v)
                adj[v].remove(u)
                degrees[u] -= 1
                degrees[v] -= 1
            
            cliques.append(clique)
        
        return cliques

    def solve_instance(n, p, seed):
        edges = generate_graph(n, p, seed)
        return greedy_clique_partition(n, edges)

    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(solve_instance(n, p, seed))
        
    return results
