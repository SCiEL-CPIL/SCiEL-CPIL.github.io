import time
import random
import math

N = 20
C = 20
DEADLINE = 145.0
START = time.time()

def elapsed():
    return time.time() - START

def make_adj(col_edges):
    adj = [[-1]*N for _ in range(N)]
    for u, v, c in col_edges:
        adj[u][v] = c
        adj[v][u] = c
    return adj

def count_edges(col_edges):
    return len(col_edges)

def _try_add(adj, u, v, c):
    if u == v or adj[u][v] >= 0:
        return False
    
    for x in range(N):
        if x == u or x == v:
            continue
        if adj[u][x] < 0:
            continue
        
        c_ux = adj[u][x]
        if c_ux == c:
            return False
            
        for y in range(N):
            if y == u or y == v or y == x:
                continue
            if adj[v][y] < 0:
                continue
            
            c_vy = adj[v][y]
            if c_vy == c or c_vy == c_ux:
                continue
                
            if adj[x][y] >= 0:
                c_xy = adj[x][y]
                if c_xy == c or c_xy == c_ux or c_xy == c_vy:
                    return False
    return True

def _try_recolor(adj, u, v, new_c):
    if adj[u][v] < 0:
        return False
    if adj[u][v] == new_c:
        return True
    
    for x in range(N):
        if x == u or x == v:
            continue
        if adj[u][x] < 0:
            continue
        
        c_ux = adj[u][x]
        if c_ux == new_c:
            return False
            
        for y in range(N):
            if y == u or y == v or y == x:
                continue
            if adj[v][y] < 0:
                continue
            
            c_vy = adj[v][y]
            if c_vy == new_c or c_vy == c_ux:
                continue
                
            if adj[x][y] >= 0:
                c_xy = adj[x][y]
                if c_xy == new_c or c_xy == c_ux or c_xy == c_vy:
                    return False
    return True

def _add_edge(col_edges, adj, u, v, c):
    col_edges.append([u, v, c])
    adj[u][v] = c
    adj[v][u] = c

def _remove_edge(col_edges, adj, u, v):
    for i in range(len(col_edges)):
        if (col_edges[i][0] == u and col_edges[i][1] == v) or \
           (col_edges[i][0] == v and col_edges[i][1] == u):
            col_edges.pop(i)
            adj[u][v] = -1
            adj[v][u] = -1
            break

def _set_color(col_edges, adj, u, v, new_c):
    for i in range(len(col_edges)):
        if (col_edges[i][0] == u and col_edges[i][1] == v) or \
           (col_edges[i][0] == v and col_edges[i][1] == u):
            col_edges[i][2] = new_c
            adj[u][v] = new_c
            adj[v][u] = new_c
            break

def is_rainbow(col_edges):
    adj = make_adj(col_edges)
    for a in range(N):
        for b in range(a+1, N):
            for c_ in range(b+1, N):
                for d in range(c_+1, N):
                    for (x,y,z,w) in [(a,b,c_,d), (a,b,d,c_), (a,c_,b,d)]:
                        if adj[x][y] >= 0 and adj[y][z] >= 0 and adj[z][w] >= 0 and adj[w][x] >= 0:
                            cols = {adj[x][y], adj[y][z], adj[z][w], adj[w][x]}
                            if len(cols) != 4:
                                return False
    return True

def construct():
    best_edges = []
    best_count = 0
    
    # Greedy Initialization
    adj = [[-1]*N for _ in range(N)]
    col_edges = []
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if elapsed() > DEADLINE - 30:
            break
        # Try to add with a random valid color
        colors = list(range(C))
        random.shuffle(colors)
        for c in colors:
            if _try_add(adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
                break
    
    if len(col_edges) > best_count:
        best_count = len(col_edges)
        best_edges = [e[:] for e in col_edges]
        
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    alpha = 0.9995
    
    current_edges = [e[:] for e in col_edges]
    current_adj = [row[:] for row in adj]
    current_count = len(current_edges)
    
    steps = 0
    last_improve_step = 0
    
    while elapsed() < DEADLINE - 5:
        steps += 1
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Add edge
            u, v = random.choice(all_pairs)
            if current_adj[u][v] >= 0:
                continue
            c = random.randint(0, C-1)
            if _try_add(current_adj, u, v, c):
                _add_edge(current_edges, current_adj, u, v, c)
                current_count += 1
                if current_count > best_count:
                    best_count = current_count
                    best_edges = [e[:] for e in current_edges]
                    last_improve_step = steps
            else:
                # Try a few other colors
                for _ in range(3):
                    c_alt = random.randint(0, C-1)
                    if c_alt != c and _try_add(current_adj, u, v, c_alt):
                        _add_edge(current_edges, current_adj, u, v, c_alt)
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                            best_edges = [e[:] for e in current_edges]
                            last_improve_step = steps
                        break

        elif move_type < 0.6:
            # Remove edge
            if current_count == 0:
                continue
            idx = random.randint(0, len(current_edges)-1)
            u, v, c = current_edges[idx]
            _remove_edge(current_edges, current_adj, u, v)
            current_count -= 1
            
            # Accept with prob exp(-1/T)
            if random.random() < math.exp(-1.0 / T):
                pass # Keep removed
            else:
                # Reject: add back
                _add_edge(current_edges, current_adj, u, v, c)
                current_count += 1

        elif move_type < 0.9:
            # Recolor edge
            if current_count == 0:
                continue
            idx = random.randint(0, len(current_edges)-1)
            u, v, c_old = current_edges[idx]
            c_new = random.randint(0, C-1)
            if c_new == c_old:
                continue
            if _try_recolor(current_adj, u, v, c_new):
                _set_color(current_edges, current_adj, u, v, c_new)
                # Neutral move, always accept to explore
                pass

        else:
            # Swap colors (simplified: skip for now)
            pass

        T *= alpha
        if T < T_min:
            T = T_min
            
        # Reheat if stuck for a long time
        if steps - last_improve_step > 5000 and T < 0.5:
            T = 0.8

    if best_edges and is_rainbow(best_edges):
        return best_edges
    
    star = [[0, v, v] for v in range(1, N)]
    return star
