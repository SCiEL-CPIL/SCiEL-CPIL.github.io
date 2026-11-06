import time
import random

N = 20
C = 20
DEADLINE = 150.0
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

def _try_add(adj, u, v, c):
    if u == v:
        return False
    if adj[u][v] >= 0:
        return False
    for x in range(N):
        if x == v or adj[u][x] < 0:
            continue
        for y in range(N):
            if y == u or y == x or adj[x][y] < 0 or adj[y][v] < 0:
                continue
            cols = {c, adj[u][x], adj[x][y], adj[y][v]}
            if len(cols) != 4:
                return False
    return True

def _add_edge(col_edges, adj, u, v, c):
    col_edges.append([u, v, c])
    adj[u][v] = c
    adj[v][u] = c

def _remove_edge(col_edges, adj, u, v):
    for i, e in enumerate(col_edges):
        if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
            col_edges.pop(i)
            adj[u][v] = -1
            adj[v][u] = -1
            return True
    return False

def greedy_construct(seed_val, time_limit=2.0):
    """Run one greedy construction attempt from scratch with a given seed."""
    random.seed(seed_val)
    col_edges = []
    adj = [[-1]*N for _ in range(N)]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    
    start_time = time.time()
    for u, v in all_pairs:
        if time.time() - start_time > time_limit:
            break
        if adj[u][v] >= 0:
            continue
        
        colours = list(range(C))
        random.shuffle(colours)
        
        for c in colours:
            if _try_add(adj, u, v, c):
                _add_edge(col_edges, adj, u, v, c)
                break
    
    return col_edges, adj

def local_improve(col_edges, adj, time_limit=2.0):
    """Try to add more edges to the given graph with multiple passes."""
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if time.time() - start_time > time_limit:
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
    return col_edges, adj

def perturb(col_edges, adj, k=5):
    """Randomly remove k edges."""
    for _ in range(k):
        if not col_edges:
            break
        idx = random.randint(0, len(col_edges)-1)
        u, v, c = col_edges[idx]
        _remove_edge(col_edges, adj, u, v)
    return col_edges, adj

def _seed_star():
    return [[0, v, 0] for v in range(1, N)]

def construct():
    best = _seed_star()
    best_count = count_edges(best)
    
    start_global = time.time()
    max_time = DEADLINE - 10
    
    # ILS Loop
    for i in range(200):
        if time.time() - start_global > max_time:
            break
        
        # 1. Initialize: Greedy
        col_edges, adj = greedy_construct(i, time_limit=1.0)
        
        # 2. Improve: Local Search
        col_edges, adj = local_improve(col_edges, adj, time_limit=2.0)
        
        count = count_edges(col_edges)
        
        if count > best_count:
            best = col_edges
            best_count = count
            
            # 3. Perturb the current solution to explore new regions
            # We perturb the *current* col_edges, not the best, to continue the chain
            if time.time() - start_global < max_time - 2:
                 perturb(col_edges, adj, k=3)
        
        # Occasionally restart from best? No, ILS chain is better.
        # But if we get stuck, maybe restart fresh?
        # The greedy step at the beginning of each loop is a fresh start if we don't chain.
        # Wait, the loop above always starts with greedy_construct(i).
        # So it's a series of independent runs, not a chain.
        # To make it a chain, we should perturb the *result* of the previous iteration.
        
        # Let's restructure:
        # Iter 0: Greedy -> Improve -> Best? -> Perturb -> Next Iter starts from Perturbed?
        # My current loop starts fresh every time.
        # Let's fix that.

    # Re-implementing loop as a chain
    best = _seed_star()
    best_count = count_edges(best)
    
    current_edges, current_adj = greedy_construct(0, time_limit=1.0)
    current_edges, current_adj = local_improve(current_edges, current_adj, time_limit=2.0)
    
    for i in range(200):
        if time.time() - start_global > max_time:
            break
        
        count = count_edges(current_edges)
        if count > best_count:
            best = current_edges
            best_count = count
        
        # Perturb current to get next starting point
        if time.time() - start_global < max_time - 5:
            perturb(current_edges, current_adj, k=3)
            # Improve the perturbed state
            current_edges, current_adj = local_improve(current_edges, current_adj, time_limit=2.0)
            
        # Occasionally do a full restart to escape deep local minima?
        if i % 10 == 0:
             current_edges, current_adj = greedy_construct(i*10+1, time_limit=1.0)
             current_edges, current_adj = local_improve(current_edges, current_adj, time_limit=2.0)

    # Final verification
    if best and is_rainbow(best):
        return best
    
    # Fallback
    star = _seed_star()
    if is_rainbow(star):
        return star
    
    return star
