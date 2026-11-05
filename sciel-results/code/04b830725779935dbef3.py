import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def generate_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def solve_instance(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        cliques = []
        
        while True:
            if not any(adj):
                break
            
            if time.time() > deadline - 1.0:
                # Fallback: cover remaining edges as pairs
                # We need to iterate through all remaining edges
                # and add them as cliques of size 2.
                # We must ensure we don't add duplicates, but since we remove
                # edges from adj, the remaining adj sets only contain uncovered edges.
                # However, iterating and removing while iterating is tricky.
                # We can collect all remaining edges first.
                remaining_edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            remaining_edges.append((u, v))
                
                for u, v in remaining_edges:
                    cliques.append([u, v])
                
                break
            
            # Find vertex with max degree
            max_deg = -1
            v = -1
            for i in range(n):
                deg = len(adj[i])
                if deg > max_deg:
                    max_deg = deg
                    v = i
            
            if v == -1 or max_deg == 0:
                break
            
            # Find large clique containing v
            C = []
            V = set(adj[v])
            
            while V:
                best_u = -1
                best_deg = -1
                # Optimize degree calculation
                for u in V:
                    # Count neighbors in V
                    # adj[u] is a set. V is a set.
                    deg = len(adj[u] & V)
                    if deg > best_deg:
                        best_deg = deg
                        best_u = u
                
                if best_u == -1:
                    break
                
                C.append(best_u)
                V = V & adj[best_u]
                V.discard(best_u)
            
            clique = [v] + C
            
            # Remove edges
            for i in range(len(clique)):
                a = clique[i]
                for j in range(i + 1, len(clique)):
                    b = clique[j]
                    if b in adj[a]:
                        adj[a].remove(b)
                    if a in adj[b]:
                        adj[b].remove(a)
            
            cliques.append(clique)

        return cliques

    edges1 = generate_graph(32, 0.6, 601)
    edges2 = generate_graph(64, 0.55, 602)
    
    cliques1 = solve_instance(32, edges1)
    cliques2 = solve_instance(64, edges2)
    
    return [cliques1, cliques2]
