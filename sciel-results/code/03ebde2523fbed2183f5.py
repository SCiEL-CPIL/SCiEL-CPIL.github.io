import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145  # safety margin before 150s kill

    def random_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return edges

    def build_adj(edges, n):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def greedy_clique_partition(edges, adj, n, time_limit):
        """
        Greedy clique partition: repeatedly find the largest clique that can be
        formed from remaining edges, add it to the partition, and remove its edges.
        
        Strategy: 
        1. Build a "clique potential" score for each edge based on how many other
           edges could join it in a clique.
        2. Pick the edge with highest potential, grow it into a maximal clique
           using greedy expansion.
        3. Remove those edges and repeat.
        """
        remaining_edges = set(edges)
        remaining_adj = [set() for _ in range(n)]
        for u, v in remaining_edges:
            remaining_adj[u].add(v)
            remaining_adj[v].add(u)
        
        cliques = []
        
        def edge_clique_potential(u, v):
            """Score an edge by how many other edges share common neighbors with it."""
            common = remaining_adj[u] & remaining_adj[v]
            # For each common neighbor w, count how many other edges in the potential clique
            score = 0
            for w in common:
                # w can join if all edges from w to u and v are in remaining_edges
                # (they are, since w is in remaining_adj[u] and remaining_adj[v])
                # Additionally, we want to know if w can connect to other candidates
                # Simple heuristic: degree of w in the common neighborhood
                score += len(remaining_adj[w] & common)
            return score
        
        while remaining_edges and time.time() < time_limit:
            # Find the best edge to start a clique
            best_edge = None
            best_score = -1
            
            # Sample a subset of edges if there are too many
            edge_list = list(remaining_edges)
            if len(edge_list) > 500:
                sample = random.sample(edge_list, 500)
            else:
                sample = edge_list
            
            for e in sample:
                u, v = e
                s = edge_clique_potential(u, v)
                if s > best_score:
                    best_score = s
                    best_edge = e
            
            if best_edge is None:
                best_edge = next(iter(remaining_edges))
            
            u, v = best_edge
            clique = {u, v}
            
            # Greedy expansion: add vertices that connect to all in clique
            while True:
                # Find candidates that connect to all vertices in clique
                candidates = set()
                for w in range(n):
                    if w in clique:
                        continue
                    ok = True
                    for c in clique:
                        if (min(c, w), max(c, w)) not in remaining_edges:
                            ok = False
                            break
                    if ok:
                        candidates.add(w)
                
                if not candidates:
                    break
                
                # Pick the candidate that maximizes future potential
                best_w = None
                best_w_score = -1
                for w in candidates:
                    # Score: degree in the current clique neighborhood
                    s = 0
                    for c in clique:
                        if (min(c, w), max(c, w)) in remaining_edges:
                            s += 1
                    # Also consider how many other candidates w connects to
                    for x in candidates:
                        if x != w and (min(w, x), max(w, x)) in remaining_edges:
                            s += 1
                    if s > best_w_score:
                        best_w_score = s
                        best_w = w
                
                if best_w is None:
                    break
                
                clique.add(best_w)
            
            # Convert to list and add to result
            clique_list = sorted(clique)
            cliques.append(clique_list)
            
            # Remove all edges in this clique from remaining
            for i in range(len(clique_list)):
                for j in range(i + 1, len(clique_list)):
                    a, b = clique_list[i], clique_list[j]
                    e = (min(a, b), max(a, b))
                    if e in remaining_edges:
                        remaining_edges.discard(e)
                        remaining_adj[a].discard(b)
                        remaining_adj[b].discard(a)
        
        return cliques

    def solve_instance(n, p, seed, time_budget):
        edges = random_graph(n, p, seed)
        adj = build_adj(edges, n)
        
        if not edges:
            return []
        
        time_limit = time.time() + time_budget
        cliques = greedy_clique_partition(edges, adj, n, time_limit)
        
        return cliques

    # Solve both instances
    t1 = time.time()
    budget1 = min(65, DEADLINE - (t1 - start_time) - 15)
    cliques1 = solve_instance(32, 0.6, 601, budget1)

    t2 = time.time()
    budget2 = min(65, DEADLINE - (t2 - start_time))
    cliques2 = solve_instance(64, 0.55, 602, budget2)

    return [cliques1, cliques2]
