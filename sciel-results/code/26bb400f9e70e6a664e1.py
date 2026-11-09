import random
import itertools
import numpy as np

def construct():
    """
    Find a C4-free graph on n vertices with minimum degree >= n-39.
    Maximize n.
    
    Approach: Use randomized local search starting from complete graph,
    remove edges to break C4s while maintaining minimum degree.
    """
    
    def has_c4(adj_matrix, n):
        """Check if graph has a C4 using matrix multiplication trick."""
        # A C4 exists if there are vertices u,v with common neighbors w,x
        # such that w-x is not an edge (to avoid triangles) - actually we just need any C4
        # Simpler: for each pair (u,v), check if they share >= 2 common neighbors
        # That would form a C4: u-w-v-x-u if w-x is edge... wait no.
        # C4: u-a-b-v-u. So u and v share common neighbors a and b, and a-b must be an edge?
        # No: u-a, a-b, b-v, v-u. So u and v are NOT directly connected necessarily.
        # Actually in a C4, opposite vertices have exactly 2 common neighbors.
        
        # Efficient check: count common neighbors for each pair
        for u in range(n):
            for v in range(u+1, n):
                # Count common neighbors
                common = 0
                for w in range(n):
                    if w != u and w != v and adj_matrix[u][w] and adj_matrix[v][w]:
                        common += 1
                        if common >= 2:
                            return True
        return False
    
    def find_c4_edges(adj_matrix, n):
        """Find all edges that are part of a C4."""
        c4_edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if not adj_matrix[u][v]:
                    continue
                # Check if edge (u,v) is in a C4
                # C4: u-w-x-v-u, so w and x are common neighbors of u and v, and w-x is an edge
                common_neighbors = [w for w in range(n) if w != u and w != v and adj_matrix[u][w] and adj_matrix[v][w]]
                for i in range(len(common_neighbors)):
                    for j in range(i+1, len(common_neighbors)):
                        w = common_neighbors[i]
                        x = common_neighbors[j]
                        if adj_matrix[w][x]:
                            # u-w-x-v-u is a C4, so edges (u,v), (u,w), (w,x), (x,v) are in C4
                            c4_edges.add((min(u,v), max(u,v)))
                            c4_edges.add((min(u,w), max(u,w)))
                            c4_edges.add((min(w,x), max(w,x)))
                            c4_edges.add((min(x,v), max(x,v)))
        return c4_edges
    
    def is_c4_free(adj_matrix, n):
        """Check if graph is C4-free."""
        for u in range(n):
            for v in range(u+1, n):
                common = 0
                for w in range(n):
                    if w != u and w != v and adj_matrix[u][w] and adj_matrix[v][w]:
                        common += 1
                        if common >= 2:
                            return False
        return True
    
    def min_degree(adj_matrix, n):
        return min(sum(adj_matrix[i][j] for j in range(n)) for i in range(n))
    
    def try_random_greedy(n, target_deg, num_trials=50):
        """
        Random greedy: build graph by adding edges in random order,
        skip edge if it creates C4 or if both vertices would exceed some limit.
        Then try to repair minimum degree.
        """
        for trial in range(num_trials):
            # Start with empty graph
            adj = [[False]*n for _ in range(n)]
            edges = []
            
            # Random edge order
            all_edges = list(itertools.combinations(range(n), 2))
            random.shuffle(all_edges)
            
            # Greedy add: add edge if it doesn't create C4
            for u, v in all_edges:
                # Check if adding (u,v) creates C4
                creates_c4 = False
                # C4 formed if there exist a,b such that u-a, a-b, b-v are edges
                # i.e., common neighbors of u and v that are connected
                nu = [w for w in range(n) if w != v and adj[u][w]]
                nv = [w for w in range(n) if w != u and adj[v][w]]
                for a in nu:
                    for b in nv:
                        if a != b and adj[a][b]:
                            creates_c4 = True
                            break
                    if creates_c4:
                        break
                
                if not creates_c4:
                    adj[u][v] = True
                    adj[v][u] = True
                    edges.append([u, v])
            
            # Check min degree
            md = min_degree(adj, n)
            if md >= target_deg:
                return {'n': n, 'edges': edges}
            
            # Try to repair: add edges to low-degree vertices
            # But we can't add edges that create C4s
            for _ in range(100):  # iterations
                md = min_degree(adj, n)
                if md >= target_deg:
                    break
                # Find vertex with min degree
                degs = [sum(adj[i][j] for j in range(n)) for i in range(n)]
                min_v = min(range(n), key=lambda i: degs[i])
                if degs[min_v] >= target_deg:
                    break
                # Try to add edges to min_v
                candidates = [v for v in range(n) if v != min_v and not adj[min_v][v]]
                random.shuffle(candidates)
                added = False
                for v in candidates:
                    # Check if adding (min_v, v) creates C4
                    creates_c4 = False
                    nu = [w for w in range(n) if w != v and adj[min_v][w]]
                    nv = [w for w in range(n) if w != min_v and adj[v][w]]
                    for a in nu:
                        for b in nv:
                            if a != b and adj[a][b]:
                                creates_c4 = True
                                break
                        if creates_c4:
                            break
                    if not creates_c4:
                        adj[min_v][v] = True
                        adj[v][min_v] = True
                        edges.append([min_v, v])
                        added = True
                        break
                if not added:
                    break
            
            md = min_degree(adj, n)
            if md >= target_deg:
                return {'n': n, 'edges': edges}
        
        return None
    
    # Try n from 47 down to 39
    for n in range(47, 38, -1):
        target_deg = n - 39
        res = try_random_greedy(n, target_deg, num_trials=30)
        if res:
            return res
    
    # Fallback: return n=39 with empty graph (min_deg 0 >= 0)
    return {'n': 39, 'edges': []}
