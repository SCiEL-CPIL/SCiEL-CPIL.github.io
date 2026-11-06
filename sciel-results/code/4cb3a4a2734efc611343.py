import time
from collections import defaultdict

def construct():
    def instance(n, d, seed):
        import random
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        # The seed program generates edges by random sampling.
        # We must replicate this exactly to get the same graph.
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        # Build adjacency list for G
        adj_g = defaultdict(set)
        for u, v in edges:
            adj_g[u].add(v)
            adj_g[v].add(u)
            
        # Map edges to indices
        edge_list = edges
        m = len(edge_list)
        edge_index = {e: i for i, e in enumerate(edge_list)}
        
        # Build H = L(G)^2
        # H[i] is the set of indices of edges that conflict with edge_list[i]
        h_adj = [set() for _ in range(m)]
        
        # Two edges e1=(u,v) and e2=(x,y) conflict if:
        # 1. They share a vertex: {u,v} & {x,y} != empty
        # 2. They are connected by an edge in G: 
        #    (u,x) in G or (u,y) in G or (v,x) in G or (v,y) in G
        
        for i, e1 in enumerate(edge_list):
            u1, v1 = e1
            for j, e2 in enumerate(edge_list):
                if i == j: continue
                u2, v2 = e2
                # Check share vertex
                if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
                    h_adj[i].add(j)
                    continue
                # Check connected by edge
                # Note: edges are (min, max)
                if (min(u1, u2), max(u1, u2)) in edge_index:
                    h_adj[i].add(j)
                elif (min(u1, v2), max(u1, v2)) in edge_index:
                    h_adj[i].add(j)
                elif (min(v1, u2), max(v1, u2)) in edge_index:
                    h_adj[i].add(j)
                elif (min(v1, v2), max(v1, v2)) in edge_index:
                    h_adj[i].add(j)
        
        # DSATUR Algorithm
        # color[i] = -1 if uncolored
        color = [-1] * m
        # colors_used[j] = set of colours used in neighbourhood of i? 
        # Better: for each vertex i, keep a set of colours present in N(i)
        # And a counter for saturation
        
        # To speed up, we can precompute degrees in H
        deg_h = [len(h_adj[i]) for i in range(m)]
        
        # We will maintain a list of uncolored vertices.
        # At each step, select vertex with max saturation, then max degree.
        
        # Optimization: Use a priority queue or just scan. m <= 240, scanning is fine.
        
        def dsatur():
            color = [-1] * m
            # sat[i] = set of colours in N(i)
            sat = [set() for _ in range(m)]
            uncolored = set(range(m))
            
            # Initial choice: vertex with max degree
            # But DSATUR starts with 0 saturation for all.
            # Standard DSATUR: pick max degree first.
            
            while uncolored:
                # Find best vertex
                best_v = -1
                best_sat = -1
                best_deg = -1
                
                for v in uncolored:
                    s = len(sat[v])
                    d = deg_h[v]
                    if s > best_sat or (s == best_sat and d > best_deg):
                        best_v = v
                        best_sat = s
                        best_deg = d
                        
                v = best_v
                uncolored.remove(v)
                
                # Find smallest available colour
                # Colours are 0, 1, 2...
                # We can just check 0, 1, 2... until we find one not in sat[v]
                c = 0
                while c in sat[v]:
                    c += 1
                color[v] = c
                
                # Update neighbours
                for u in h_adj[v]:
                    if u in uncolored:
                        sat[u].add(c)
                        
            return color

        start_time = time.time()
        # DSATUR is deterministic. 
        # If it's too slow, we might need a timeout, but for m=240 it should be fast.
        # Let's add a simple timeout just in case.
        # However, DSATUR is exponential in worst case. 
        # For safety, if it takes > 10s, we might need to fall back.
        # But typically these graphs are sparse enough for DSATUR to work well.
        
        try:
            # Run DSATUR
            col = dsatur()
            # Check validity
            # (DSATUR guarantees validity)
            # The number of colours is max(col) + 1
            results.append([col[edge_index[e]] for e in edge_list])
        except Exception as ex:
            # Fallback to greedy if DSATUR fails or times out (though it shouldn't)
            # Greedy
            col = [-1] * m
            sat = [set() for _ in range(m)]
            for v in range(m):
                c = 0
                while c in sat[v]:
                    c += 1
                col[v] = c
                for u in h_adj[v]:
                    if col[u] == -1:
                        sat[u].add(c)
            results.append([col[edge_index[e]] for e in edge_list])

    return results
