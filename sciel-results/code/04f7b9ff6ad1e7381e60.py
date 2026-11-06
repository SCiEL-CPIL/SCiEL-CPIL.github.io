import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    def generate_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def greedy_partition(edges, n):
        remaining_edges = set(edges)
        cliques = []
        
        # Precompute all maximal cliques or large cliques?
        # For dense graphs, we can try to find large cliques greedily.
        
        # Strategy:
        # 1. Compute adjacency matrix for O(1) lookups.
        # 2. While edges remain:
        #    a. Find a vertex with high degree in remaining graph.
        #    b. Greedily expand a clique from that vertex, prioritizing vertices that connect to many others in the current clique AND have high remaining degree.
        #    c. Remove all edges within the clique.
        
        # Adjacency matrix for original graph (for fast clique verification)
        adj_matrix = [[False] * n for _ in range(n)]
        for u, v in edges:
            adj_matrix[u][v] = True
            adj_matrix[v][u] = True
            
        # Remaining adjacency (dynamic)
        rem_adj = [set() for _ in range(n)]
        for u, v in edges:
            rem_adj[u].add(v)
            rem_adj[v].add(u)
            
        while remaining_edges:
            if time.time() > deadline:
                # Fallback: return current best plus edges as size-2 cliques
                for e in remaining_edges:
                    cliques.append(list(e))
                return cliques
                
            # Find vertex with max degree in remaining graph
            max_deg = -1
            v_start = -1
            for u in range(n):
                if len(rem_adj[u]) > max_deg:
                    max_deg = len(rem_adj[u])
                    v_start = u
                    
            if v_start == -1:
                break
                
            # Greedy clique expansion
            C = {v_start}
            # Candidates: neighbors of v_start in remaining graph
            candidates = set(rem_adj[v_start])
            
            # We want to pick vertices that form a clique.
            # At each step, pick a candidate u such that u is connected to all in C.
            # To maximize future potential, pick u with highest "connectivity score".
            # Score = number of vertices in candidates that are also connected to u.
            
            while candidates:
                # Filter candidates to those connected to ALL vertices in C
                valid_candidates = []
                for u in candidates:
                    is_valid = True
                    for v in C:
                        if v == u:
                            continue
                        if u not in rem_adj[v]:
                            is_valid = False
                            break
                    if is_valid:
                        valid_candidates.append(u)
                        
                if not valid_candidates:
                    break
                    
                # Score each valid candidate
                # Heuristic: pick the one with the most connections to other valid candidates
                # This tends to lead to larger cliques.
                best_candidate = None
                best_score = -1
                
                for u in valid_candidates:
                    # Count how many other valid candidates are connected to u
                    score = 0
                    for v in valid_candidates:
                        if u == v:
                            continue
                        if v in rem_adj[u]:
                            score += 1
                    # Tie-breaker: degree in remaining graph
                    if score > best_score or (score == best_score and len(rem_adj[u]) > len(rem_adj[best_candidate] if best_candidate is not None else -1)):
                        best_score = score
                        best_candidate = u
                        
                C.add(best_candidate)
                
                # Update candidates: remove best_candidate, and remove any vertex not connected to best_candidate
                new_candidates = set()
                for u in candidates:
                    if u == best_candidate:
                        continue
                    if best_candidate in rem_adj[u]:
                        new_candidates.add(u)
                candidates = new_candidates
                
            if len(C) >= 2:
                clique_list = sorted(C)
                cliques.append(clique_list)
                # Remove all edges within C from remaining_edges and rem_adj
                for i in range(len(clique_list)):
                    for j in range(i + 1, len(clique_list)):
                        u, v = clique_list[i], clique_list[j]
                        remaining_edges.discard((u, v))
                        rem_adj[u].discard(v)
                        rem_adj[v].discard(u)
                        
        return cliques

    # Instance 1
    edges1 = generate_graph(32, 0.6, 601)
    partition1 = greedy_partition(edges1, 32)
    
    # Instance 2
    edges2 = generate_graph(64, 0.55, 602)
    partition2 = greedy_partition(edges2, 64)
    
    return [partition1, partition2]
