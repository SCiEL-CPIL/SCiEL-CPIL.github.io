import random
import time
import math

def construct():
    n = 20
    k = 20
    start_time = time.time()
    deadline = start_time + 115
    
    def time_ok():
        return time.time() < deadline

    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    def run_instance(seed):
        random.seed(seed)
        adj = [[False]*n for _ in range(n)]
        col = [[-1]*n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        current_edges = []
        edge_count = 0
        
        def add_edge(u, v, c):
            nonlocal edge_count
            adj[u][v] = True
            adj[v][u] = True
            col[u][v] = c
            col[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
            current_edges.append((u, v))
            edge_count += 1
            
        def remove_edge(u, v):
            nonlocal edge_count
            adj[u][v] = False
            adj[v][u] = False
            col[u][v] = -1
            col[v][u] = -1
            neighbors[u].remove(v)
            neighbors[v].remove(u)
            for i, (eu, ev) in enumerate(current_edges):
                if (eu == u and ev == v) or (eu == v and ev == u):
                    current_edges.pop(i)
                    break
            edge_count -= 1
            
        def is_valid(u, v, c):
            adj_l = adj
            col_l = col
            neigh_v = neighbors[v]
            neigh_u = neighbors[u]
            for w in neigh_v:
                if w == u:
                    continue
                cw = col_l[v][w]
                for x in neigh_u:
                    if x == v or x == w:
                        continue
                    if not adj_l[w][x]:
                        continue
                    cwx = col_l[w][x]
                    cxu = col_l[x][u]
                    if c == cw or c == cwx or c == cxu:
                        return False
                    if cw == cwx or cw == cxu:
                        return False
                    if cwx == cxu:
                        return False
            return True

        def greedy_init():
            nonlocal edge_count
            random.shuffle(all_pairs)
            for u, v in all_pairs:
                colors = list(range(k))
                random.shuffle(colors)
                for c in colors:
                    if is_valid(u, v, c):
                        add_edge(u, v, c)
                        break
                        
        def get_best_state():
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i][j]:
                        edges.append([i, j, col[i][j]])
            return edges

        greedy_init()
        
        best_edges = get_best_state()
        best_count = edge_count
        
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        no_improve = 0
        max_no_improve = 2000
        
        while time_ok():
            if T < T_min:
                T = T_min
                
            r = random.random()
            if r < 0.3:
                if edge_count < n*(n-1)//2:
                    u, v = random.choice(all_pairs)
                    if not adj[u][v]:
                        c = random.randint(0, k-1)
                        if is_valid(u, v, c):
                            add_edge(u, v, c)
                            if edge_count > best_count:
                                best_count = edge_count
                                best_edges = get_best_state()
                                no_improve = 0
                            else:
                                no_improve += 1
            elif r < 0.6:
                if current_edges:
                    idx = random.randint(0, len(current_edges)-1)
                    u, v = current_edges[idx]
                    old_c = col[u][v]
                    remove_edge(u, v)
                    if random.random() < math.exp(-1.0 / T):
                        no_improve += 1
                    else:
                        add_edge(u, v, old_c)
            else:
                if current_edges:
                    idx = random.randint(0, len(current_edges)-1)
                    u, v = current_edges[idx]
                    old_c = col[u][v]
                    new_c = random.randint(0, k-1)
                    if new_c != old_c:
                        if is_valid(u, v, new_c):
                            col[u][v] = new_c
                            col[v][u] = new_c
                            no_improve += 1
                            
            if no_improve > max_no_improve:
                break
                
            T *= cooling
            
        return best_edges

    best_overall = []
    best_count_overall = 0
    seed = 1
    while time_ok():
        edges = run_instance(seed)
        if len(edges) > best_count_overall:
            best_count_overall = len(edges)
            best_overall = edges
        seed += 1
        
    return best_overall
