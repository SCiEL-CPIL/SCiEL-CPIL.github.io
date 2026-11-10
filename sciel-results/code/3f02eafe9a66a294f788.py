import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = set()
        adj = [set() for _ in range(n)]
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
                    adj[u].add(v)
                    adj[v].add(u)
        return edges, adj

    def find_maximal_cliques(n, adj):
        maximal_cliques = []
        
        def bronk(R, P, X):
            if not P and not X:
                if len(R) >= 2:
                    maximal_cliques.append(list(R))
                return
            
            # Choose pivot u from P union X
            if P:
                pivot = next(iter(P))
            elif X:
                pivot = next(iter(X))
            else:
                return
            
            # v in (P - N(u))
            # We iterate over a copy of P to allow modification
            # But we need to ensure we don't visit nodes that are not in P - N(u)
            candidates = [v for v in P if v not in adj[pivot]]
            
            for v in candidates:
                bronk(R | {v}, P & adj[v], X & adj[v])
                P.remove(v)
                X.add(v)
        
        bronk(set(), set(range(n)), set())
        return maximal_cliques

    def solve_instance(n, p, seed):
        edges, adj = generate_graph(n, p, seed)
        if not edges:
            return []
        
        max_cliques = find_maximal_cliques(n, adj)
        
        # Precompute edge sets for each clique
        clique_edge_sets = []
        for clique in max_cliques:
            c_list = clique
            c_edges = set()
            for i in range(len(c_list)):
                for j in range(i+1, len(c_list)):
                    u, v = c_list[i], c_list[j]
                    if u > v:
                        u, v = v, u
                    c_edges.add((u, v))
            clique_edge_sets.append(c_edges)
        
        uncovered = set(edges)
        result = []
        
        # If no maximal cliques found (shouldn't happen if edges exist), fallback
        if not max_cliques:
             for e in edges:
                 result.append(list(e))
             return result

        while uncovered:
            if time.time() > deadline:
                # Fallback: add remaining edges as singletons
                for e in uncovered:
                    result.append(list(e))
                break
            
            best_clique_idx = -1
            best_count = 0
            best_edge_set = None
            
            # Find clique covering most uncovered edges
            for idx, c_edges in enumerate(clique_edge_sets):
                if not c_edges:
                    continue
                # Optimization: if c_edges is subset of uncovered, count is len(c_edges)
                # Otherwise intersection
                count = len(c_edges & uncovered)
                if count > best_count:
                    best_count = count
                    best_clique_idx = idx
                    best_edge_set = c_edges
            
            if best_clique_idx == -1:
                # No clique covers any uncovered edge? 
                # This implies some edges are not in any maximal clique? 
                # Impossible for simple graphs: every edge is in a maximal clique.
                # But maybe our enumeration missed it? Or time out?
                # If best_count is 0, we must pick something.
                e = next(iter(uncovered))
                result.append(list(e))
                uncovered.remove(e)
            else:
                result.append(max_cliques[best_clique_idx])
                uncovered -= best_edge_set
                
        return result

    res1 = solve_instance(32, 0.6, 601)
    res2 = solve_instance(64, 0.55, 602)
    
    return [res1, res2]
