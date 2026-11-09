import itertools
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_edges = 0
    best_graph = None
    
    # --- Helper Functions ---
    
    def contains_q3(G, S):
        S_list = list(S)
        for A in itertools.combinations(S_list, 4):
            A_set = set(A)
            B = tuple(v for v in S_list if v not in A_set)
            
            adj = [[0]*4 for _ in range(4)]
            for i, u in enumerate(A):
                for j, v in enumerate(B):
                    if G[u][v]:
                        adj[i][j] = 1
            
            # Quick check: min degree >= 3
            for i in range(4):
                if sum(adj[i]) < 3:
                    break
                if sum(adj[k][i] for k in range(4)) < 3:
                    break
            else:
                # Min degree >= 3, check flow
                cap = [[0]*10 for _ in range(10)]
                for i in range(4):
                    cap[0][1+i] = 3
                    cap[5+i][9] = 3
                    for j in range(4):
                        if adj[i][j]:
                            cap[1+i][5+j] = 1
                
                flow = 0
                while True:
                    visited = [False]*10
                    parent = [-1]*10
                    queue = [0]
                    visited[0] = True
                    while queue:
                        u = queue.pop(0)
                        for v in range(10):
                            if not visited[v] and cap[u][v] > 0:
                                visited[v] = True
                                parent[v] = u
                                queue.append(v)
                    if not visited[9]:
                        break
                    path_flow = float('inf')
                    v = 9
                    while v != 0:
                        u = parent[v]
                        path_flow = min(path_flow, cap[u][v])
                        v = u
                    flow += path_flow
                    v = 9
                    while v != 0:
                        u = parent[v]
                        cap[u][v] -= path_flow
                        cap[v][u] += path_flow
                        v = u
                
                if flow == 12:
                    return True
        return False

    def find_q3(G):
        vertices = list(range(16))
        random.shuffle(vertices)
        for S in itertools.combinations(vertices, 8):
            if contains_q3(G, S):
                return S
        return None

    def remove_q3_edge(G, S):
        edges_in_S = [(u, v) for u in S for v in S if u < v and G[u][v]]
        if edges_in_S:
            e = random.choice(edges_in_S)
            G[e[0]][e[1]] = 0
            G[e[1]][e[0]] = 0
            return e
        return None

    def count_edges(G):
        return sum(G[i][j] for i in range(16) for j in range(i+1, 16))

    # --- Strategy 1: Deletion ---
    def strategy_deletion(time_limit):
        local_best = 0
        local_best_G = None
        while time.time() < time_limit:
            G = [[0]*16 for _ in range(16)]
            for i in range(16):
                for j in range(16):
                    if i != j:
                        G[i][j] = 1
            
            while time.time() < time_limit:
                S = find_q3(G)
                if S is None:
                    break
                remove_q3_edge(G, S)
            
            c = count_edges(G)
            if c > local_best:
                local_best = c
                local_best_G = [row[:] for row in G]
        
        return local_best, local_best_G

    # --- Strategy 2: Greedy Add ---
    def creates_q3(G, u, v):
        Nu = [x for x in range(16) if x != v and G[u][x]]
        Nv = [x for x in range(16) if x != u and G[v][x]]
        
        for a in Nu:
            for b in Nu:
                if a == b: continue
                for c in Nv:
                    for d in Nv:
                        if c == d: continue
                        if len({u,v,a,b,c,d}) < 6: continue
                        
                        # Check cross edges
                        if (G[a][c] and G[b][d]) or (G[a][d] and G[b][c]):
                            # Find x, y
                            # x is common neighbor of a, b
                            # y is common neighbor of c, d
                            # x, y not in {u,v,a,b,c,d}
                            # (x,y) in G
                            excluded = {u,v,a,b,c,d}
                            CandX = [w for w in range(16) if w not in excluded and G[a][w] and G[b][w]]
                            CandY = [w for w in range(16) if w not in excluded and G[c][w] and G[d][w]]
                            for x in CandX:
                                for y in CandY:
                                    if x == y: continue
                                    if G[x][y]:
                                        return True
        return False

    def strategy_greedy_add(time_limit):
        local_best = 0
        local_best_G = None
        while time.time() < time_limit:
            G = [[0]*16 for _ in range(16)]
            pairs = [(i, j) for i in range(16) for j in range(i+1, 16)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                if not G[u][v]:
                    if not creates_q3(G, u, v):
                        G[u][v] = 1
                        G[v][u] = 1
            
            c = count_edges(G)
            if c > local_best:
                local_best = c
                local_best_G = [row[:] for row in G]
        
        return local_best, local_best_G

    # Run strategies
    t1 = time.time()
    # Give each strategy roughly half the time
    half_time = start_time + 70
    
    b1, g1 = strategy_deletion(half_time)
    if g1:
        if b1 > best_edges:
            best_edges = b1
            best_graph = g1
            
    t2 = time.time()
    b2, g2 = strategy_greedy_add(deadline)
    if g2:
        if b2 > best_edges:
            best_edges = b2
            best_graph = g2

    edges = []
    if best_graph:
        for i in range(16):
            for j in range(i+1, 16):
                if best_graph[i][j]:
                    edges.append([i, j])
    else:
        edges = [[a, b] for a in range(3) for b in range(3, 16)]
        
    return edges
