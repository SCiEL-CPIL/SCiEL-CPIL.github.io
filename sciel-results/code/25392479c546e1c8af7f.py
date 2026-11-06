def construct():
    import time
    start = time.time()
    DEADLINE = 145
    
    n = 20
    adj = [[-1]*n for _ in range(n)]
    edges = []
    
    # Round-robin ordering for better distribution
    all_edges = []
    for i in range(n):
        for j in range(i+1, n):
            all_edges.append((i,j))
    # Shuffle deterministically for variety? No, let's keep it simple.
    
    def check_edge(u, v, c):
        for w in range(n):
            if w == u or w == v: continue
            if adj[u][w] < 0: continue
            cw = adj[u][w]
            if cw == c: continue
            for x in range(n):
                if x == u or x == v or x == w: continue
                if adj[v][x] < 0: continue
                cx = adj[v][x]
                if cx == c or cx == cw: continue
                if adj[w][x] < 0: continue
                cwx = adj[w][x]
                if cwx == c or cwx == cw or cwx == cx:
                    return False
        return True
    
    for (u,v) in all_edges:
        if time.time() - start > DEADLINE:
            break
        for c in range(20):
            if check_edge(u, v, c):
                adj[u][v] = c
                adj[v][u] = c
                edges.append([u, v, c])
                break
    
    return edges
