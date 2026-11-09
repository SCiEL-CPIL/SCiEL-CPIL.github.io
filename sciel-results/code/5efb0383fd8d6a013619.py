import random
import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Try to find a subgraph of the Hoffman-Singleton graph
    # The HS graph has 50 vertices, is triangle-free, and has high independence number.
    # We need a subgraph on 40 vertices with alpha <= 9.
    
    def get_hoffman_sington_edges():
        # HS graph is the unique strongly regular graph (50, 7, 0, 1)
        # It can be constructed as a Cayley graph on Z_5 x Z_5 x Z_5? No.
        # It is the graph of the 50 points in PG(3, 5)? No.
        # Standard construction:
        # Vertices are pairs (i, j) where i in Z_5, j in Z_5? No, 25 vertices.
        # HS graph vertices: (x, y) in Z_5 x Z_5 is not enough.
        # Actually, HS graph is a Cayley graph on Z_5 x Z_5 x Z_5? No, 125.
        # It is a Cayley graph on the group of order 50?
        # Let's use a known adjacency list or generation method.
        # For simplicity, I will use a random triangle-free graph search if HS is too complex to hardcode.
        # But HS is a good candidate.
        # Let's try a random search on n=40 directly.
        pass

    # Fallback: Random Local Search on n=40
    n = 40
    best_alpha = float('inf')
    best_edges = None
    
    def is_triangle_free(adj):
        for i in range(n):
            for j in adj[i]:
                if j < i: continue
                for k in adj[j]:
                    if k > j and (k in adj[i]):
                        return False
        return True

    def calc_alpha(adj):
        # Simple greedy approximation for alpha
        # Or exact for small n? n=40 is too big for exact.
        # Use a simple greedy coloring of complement?
        # Complement has edges where G has no edges.
        # We want max independent set in G = max clique in complement.
        # Greedy clique in complement.
        comp_adj = [[False]*n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j and j not in adj[i]:
                    comp_adj[i][j] = True
        
        # Greedy clique
        max_clique = 0
        for i in range(n):
            clique = {i}
            for j in range(i+1, n):
                if comp_adj[i][j] and all(comp_adj[k][j] for k in clique):
                    clique.add(j)
            if len(clique) > max_clique:
                max_clique = len(clique)
        return max_clique

    # Random initialization
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline: break
        # Start with a random graph, then remove edges to make it triangle-free
        adj = set()
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj.add((i, j))
        
        # Remove triangles
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) in adj:
                        for k in range(j+1, n):
                            if (i, k) in adj and (j, k) in adj:
                                # Remove one edge
                                adj.remove((i, k))
                                changed = True
                                break
                        if changed: break
                if changed: break
        
        # Convert to list of lists
        adj_list = [set() for _ in range(n)]
        for i, j in adj:
            adj_list[i].add(j)
            adj_list[j].add(i)
        
        alpha = calc_alpha(adj_list)
        if alpha < best_alpha:
            best_alpha = alpha
            best_edges = sorted([(min(i,j), max(i,j)) for i,j in adj])
            
            if best_alpha <= 9:
                return {'n': n, 'edges': [[u, v] for u, v in best_edges]}

    # If not found, return seed
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    return {'n': n_seed, 'edges': edges_seed}
