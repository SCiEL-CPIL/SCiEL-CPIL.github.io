import time

def construct():
    def random_graph(n, p, seed):
        import random
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, p, seed):
        edges = random_graph(n, p, seed)
        remaining_edges = set(edges)
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        while remaining_edges:
            # Heuristic: pick edge with max sum of degrees in remaining graph
            deg = [0] * n
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
                
            seed_edge = None
            max_deg_sum = -1
            for u, v in remaining_edges:
                s = deg[u] + deg[v]
                if s > max_deg_sum:
                    max_deg_sum = s
                    seed_edge = (u, v)
            
            u, v = seed_edge
            
            # Greedy clique expansion
            K = {u, v}
            candidates = adj[u].intersection(adj[v])
            
            while candidates:
                best_w = None
                max_common = -1
                for w in candidates:
                    common = len(adj[w].intersection(K))
                    if common > max_common:
                        max_common = common
                        best_w = w
                
                if best_w is None:
                    break
                    
                K.add(best_w)
                candidates = candidates.intersection(adj[best_w])
                
            cliques.append(list(K))
            
            # Remove edges in K
            K_list = list(K)
            to_remove = set()
            for i in range(len(K_list)):
                for j in range(i + 1, len(K_list)):
                    e1 = (K_list[i], K_list[j])
                    e2 = (K_list[j], K_list[i])
                    if e1 in remaining_edges:
                        to_remove.add(e1)
                    if e2 in remaining_edges:
                        to_remove.add(e2)
            remaining_edges.difference_update(to_remove)
            
        return cliques

    c1 = solve_instance(32, 0.6, 601)
    c2 = solve_instance(64, 0.55, 602)
    
    return [c1, c2]
