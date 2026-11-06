import time
import random
from itertools import combinations

N = 20
C = 20
DEADLINE = 140.0
START = time.time()

def elapsed():
    return time.time() - START

def make_adj(col_edges):
    adj = [[-1]*N for _ in range(N)]
    for u, v, c in col_edges:
        adj[u][v] = c
        adj[v][u] = c
    return adj

def all_four_cycles(adj):
    n = N
    for a in range(n):
        for b in range(a+1, n):
            for c in range(b+1, n):
                for d in range(c+1, n):
                    for (x,y,z,w) in [(a,b,c,d), (a,b,d,c), (a,c,b,d)]:
                        if adj[x][y] >= 0 and adj[y][z] >= 0 and adj[z][w] >= 0 and adj[w][x] >= 0:
                            yield (x,y,adj[x][y]), (y,z,adj[y][z]), (z,w,adj[z][w]), (w,x,adj[w][x])

def is_rainbow(col_edges):
    adj = make_adj(col_edges)
    for cyc in all_four_cycles(adj):
        cols = set(e[2] for e in cyc)
        if len(cols) != 4:
            return False
    return True

def count_edges(col_edges):
    return len(col_edges)

def _try_add(adj, adj_list, u, v, c):
    if u == v:
        return False
    if adj[u][v] >= 0:
        return False
    for x in adj_list[u]:
        if x == v:
            continue
        ux_col = adj[u][x]
        for y in adj_list[x]:
            if y == u or y == x:
                continue
            if adj[y][v] < 0:
                continue
            xy_col = adj[x][y]
            yv_col = adj[y][v]
            if c == ux_col or c == xy_col or c == yv_col or ux_col == xy_col or ux_col == yv_col or xy_col == yv_col:
                return False
    return True

def _add_edge(col_edges, adj, adj_list, u, v, c):
    col_edges.append([u, v, c])
    adj[u][v] = c
    adj[v][u] = c
    adj_list[u].append(v)
    adj_list[v].append(u)

def greedy_construct(seed_val):
    random.seed(seed_val)
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if elapsed() > DEADLINE - 5:
            break
        if adj[u][v] >= 0:
            continue
        
        colours = list(range(C))
        random.shuffle(colours)
        
        for c in colours:
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
                break
    
    return col_edges

def local_improve(col_edges):
    adj = make_adj(col_edges)
    adj_list = [[] for _ in range(N)]
    for u, v, c in col_edges:
        adj_list[u].append(v)
        adj_list[v].append(u)
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    for attempt in range(10):
        random.shuffle(all_pairs)
        improved = False
        for u, v in all_pairs:
            if elapsed() > DEADLINE - 5:
                break
            if adj[u][v] >= 0:
                continue
            colours = list(range(C))
            random.shuffle(colours)
            for c in colours:
                if _try_add(adj, adj_list, u, v, c):
                    _add_edge(col_edges, adj, adj_list, u, v, c)
                    improved = True
                    break
        if not improved:
            break
    
    return col_edges

def try_structured_coloring():
    """Try the (u+v) mod C coloring on complete graph."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u + v) % C
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
    
    return col_edges

def try_structured_coloring2():
    """Try (u-v) mod C coloring."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u - v) % C
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
    
    return col_edges

def try_structured_coloring3():
    """Try (u*v) mod C coloring."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u * v) % C
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
    
    return col_edges

def try_structured_coloring4():
    """Try (u*v + u + v) mod C coloring."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u * v + u + v) % C
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
    
    return col_edges

def try_structured_coloring5():
    """Try (u^2 + v^2) mod C coloring."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    adj_list = [[] for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u*u + v*v) % C
            if _try_add(adj, adj_list, u, v, c):
                _add_edge(col_edges, adj, adj_list, u, v, c)
    
    return col_edges

def construct():
    best = []
    best_count = 0
    
    # Try structured colorings first
    for fn in [try_structured_coloring, try_structured_coloring2, try_structured_coloring3,
               try_structured_coloring4, try_structured_coloring5]:
        if elapsed() > DEADLINE - 30:
            break
        result = fn()
        count = count_edges(result)
        if count > best_count:
            best = result
            best_count = count
    
    # Run multiple greedy attempts
    max_attempts = 50
    for i in range(max_attempts):
        if elapsed() > DEADLINE - 15:
            break
        
        result = greedy_construct(i)
        count = count_edges(result)
        
        if count > best_count:
            best = result
            best_count = count
    
    # Final polish: try to add more edges to the best solution
    if best and elapsed() < DEADLINE - 10:
        improved = local_improve([e[:] for e in best])
        improved_count = count_edges(improved)
        if improved_count > best_count:
            best = improved
            best_count = improved_count
    
    # Final verification
    if best and is_rainbow(best):
        return best
    
    # Fallback
    star = [[0, v, v] for v in range(1, N)]
    return star
