import time
import random
import numpy as np

N = 20
C = 20
DEADLINE = 135.0
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
    """Check if adding edge (u,v) with color c creates any non-rainbow 4-cycle."""
    if u == v:
        return False
    if adj[u][v] >= 0:
        return False
    
    # For a 4-cycle containing (u,v), we need two other vertices x,y
    # such that u-x-y-v-u forms a cycle.
    # The cycle edges are (u,x), (x,y), (y,v), (v,u).
    # Colors must be distinct: adj[u][x], adj[x][y], adj[y][v], c must all be different.
    
    # We can iterate over all pairs (x, y) where x is neighbor of u, y is neighbor of v,
    # and x is neighbor of y.
    # To be efficient, we iterate x over neighbors of u, y over neighbors of v.
    
    neighbors_u = [x for x in range(N) if adj[u][x] >= 0]
    neighbors_v = [y for y in range(N) if adj[v][y] >= 0]
    
    for x in neighbors_u:
        if x == v:
            continue
        for y in neighbors_v:
            if y == u or y == x:
                continue
            if adj[x][y] >= 0:
                # Check if (u,x,y,v,u) is a 4-cycle
                # Edges: (u,x), (x,y), (y,v), (v,u)
                # Colors: adj[u][x], adj[x][y], adj[y][v], c
                # They must be all distinct.
                if adj[u][x] == adj[x][y] or adj[u][x] == adj[y][v] or adj[u][x] == c \
                   or adj[x][y] == adj[y][v] or adj[x][y] == c \
                   or adj[y][v] == c:
                    return False
    return True

def _add_edge(col_edges, adj, u, v, c):
    col_edges.append([u, v, c])
    adj[u][v] = c
    adj[v][u] = c

def greedy_construct(seed_val):
    random.seed(seed_val)
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    
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
            if _try_add(adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
                break
    
    return col_edges

def local_improve(col_edges):
    adj = make_adj(col_edges)
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
                if _try_add(adj, u, v, c):
                    _add_edge(col_edges, adj, u, v, c)
                    improved = True
                    break
        if not improved:
            break
    
    return col_edges

def try_structured_coloring():
    """Try the (u+v) mod C coloring on complete graph."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u + v) % C
            if _try_add(adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
    
    return col_edges

def try_structured_coloring2():
    """Try (u-v) mod C coloring."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u - v) % C
            if _try_add(adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
    
    return col_edges

def try_structured_coloring3():
    """Try (u*v) mod C coloring."""
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    
    for u in range(N):
        for v in range(u+1, N):
            c = (u * v) % C
            if _try_add(adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
    
    return col_edges

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
    best = []
    best_count = 0
    
    # Try structured colorings first
    for fn in [try_structured_coloring, try_structured_coloring2, try_structured_coloring3]:
        if elapsed() > DEADLINE - 30:
            break
        result = fn()
        count = count_edges(result)
        if count > best_count:
            best = result
            best_count = count
    
    # Run multiple greedy attempts
    max_attempts = 30
    for i in range(max_attempts):
        if elapsed() > DEADLINE - 15:
            break
        
        result = greedy_construct(i)
        count = count_edges(result)
        
        if count > best_count:
            best = result
            best_count = count
            
            if elapsed() < DEADLINE - 20:
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
