import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    def elapsed():
        return time.time() - start_time
    
    def time_left():
        return deadline - time.time()
    
    def check_c4_creation(adj, u, v, n):
        """Check if adding edge (u,v) would create a C4."""
        # A C4 is created if there exist a,b such that (u,a), (a,b), (b,v) are edges
        # and a != b, a not in {u,v}, b not in {u,v}
        # Equivalently: there's a path u-a-b-v of length 3
        neighbors_u = set(adj[u])
        neighbors_v = set(adj[v])
        # For each neighbor a of u, check if any neighbor b of a (b != u) is a neighbor of v (b != v)
        for a in neighbors_u:
            if a == v:
                continue
            for b in adj[a]:
                if b == u or b == v:
                    continue
                if b in neighbors_v:
                    return True
        return False
    
    def min_degree(adj, n):
        return min(len(adj[i]) for i in range(n))
    
    def is_c4_free(adj, n):
        """Verify the graph has no C4."""
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    # Check if there's a path of length 3 from u to v
                    # (which would form a C4 with edge (u,v))
                    # Actually we need to check all 4-cycles
                    pass
        # Better: check all quadruples
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if not adj[u][w]:
                        continue
                    for x in range(w+1, n):
                        if x == u or x == v:
                            continue
                        if adj[v][x] and adj[w][x]:
                            # u-v-x-w-u is a C4
                            return False
        return True
    
    def try_construct(n, target_min_deg, seed):
        """Try to construct a C4-free graph on n vertices with min degree >= target_min_deg."""
        rng = random.Random(seed)
        adj = [[False]*n for _ in range(n)]
        
        # Start with empty graph, greedily add edges
        # Priority: add edges incident to lowest-degree vertices
        for iteration in range(n * n * 10):
            if time_left() < 2:
                break
            # Find vertex with minimum degree
            min_deg = min(len([j for j in range(n) if adj[i][j]]) for i in range(n))
            if min_deg >= target_min_deg:
                break
            
            # Find a vertex with minimum degree
            min_deg_verts = [i for i in range(n) if len([j for j in range(n) if adj[i][j]]) == min_deg]
            u = rng.choice(min_deg_verts)
            
            # Try to add edges from u to non-neighbors
            non_neighbors = [v for v in range(n) if v != u and not adj[u][v]]
            rng.shuffle(non_neighbors)
            
            added = False
            for v in non_neighbors:
                if time_left() < 1:
                    break
                if not check_c4_creation(adj, u, v, n):
                    adj[u][v] = True
                    adj[v][u] = True
                    added = True
                    break
            
            if not added:
                # Try all pairs
                any_added = False
                for i in range(n):
                    for j in range(i+1, n):
                        if not adj[i][j]:
                            if not check_c4_creation(adj, i, j, n):
                                adj[i][j] = True
                                adj[j][i] = True
                                any_added = True
                                break
                    if any_added:
                        break
                if not any_added:
                    break
        
        # Check result
        md = min(len([j for j in range(n) if adj[i][j]]) for i in range(n))
        if md >= target_min_deg:
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
        return None
    
    # Try n=47 first (needs min degree 8)
    for seed in range(100):
        if time_left() < 3:
            break
        result = try_construct(47, 8, seed)
        if result:
            # Verify
            n = result['n']
            adj = [[False]*n for _ in range(n)]
            for e in result['edges']:
                adj[e[0]][e[1]] = True
                adj[e[1]][e[0]] = True
            if is_c4_free(adj, n):
                md = min(sum(adj[i]) for i in range(n))
                if md >= n - 39:
                    return result
    
    # Try n=46 (needs min degree 7)
    for seed in range(200):
        if time_left() < 3:
            break
        result = try_construct(46, 7, seed)
        if result:
            n = result['n']
            adj = [[False]*n for _ in range(n)]
            for e in result['edges']:
                adj[e[0]][e[1]] = True
                adj[e[1]][e[0]] = True
            if is_c4_free(adj, n):
                md = min(sum(adj[i]) for i in range(n))
                if md >= n - 39:
                    return result
    
    # Fallback: polarity graph of PG(2,6) - 43 vertices, C4-free, min degree 6
    # For q=6: vertices are (x,y) in GF(6)^2 plus (1:0:0) style... 
    # Actually let me just return the known small graph
    # Empty graph on 39 vertices
    return {'n': 39, 'edges': []}
