import random
import time

N = 20
C = 20

def _edge_ok(adj_mat, adj_list, colour_of, u, v, c):
    """Check whether adding edge (u,v) with colour c keeps every 4-cycle rainbow."""
    adj_v = adj_list[v]
    adj_u = adj_list[u]
    
    for x in adj_v:
        if x == u:
            continue
        cvx = colour_of[v][x]
        
        if cvx == c:
            # If cvx == c, any 4-cycle involving x will have repeated colors c.
            # So we must ensure no such 4-cycle exists.
            for y in adj_u:
                if y == v or y == x:
                    continue
                if adj_mat[x][y]:
                    return False
            continue

        for y in adj_u:
            if y == v or y == x:
                continue
            if adj_mat[x][y]:
                # Cycle u-v-x-y-u exists
                cxy = colour_of[x][y]
                cyu = colour_of[y][u]
                
                # Check distinctness: c, cvx, cxy, cyu
                if c == cvx or c == cxy or c == cyu:
                    return False
                if cvx == cxy or cvx == cyu:
                    return False
                if cxy == cyu:
                    return False
    return True

def _greedy_construct(seed):
    """Run a single greedy construction pass."""
    adj_list = [[] for _ in range(N)]
    adj_mat = [[False] * N for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    placed = []
    
    # Generate all potential edges
    potential_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng = random.Random(seed)
    rng.shuffle(potential_edges)
    
    for (u, v) in potential_edges:
        cols = list(range(C))
        rng.shuffle(cols)
        
        for c in cols:
            if _edge_ok(adj_mat, adj_list, colour_of, u, v, c):
                adj_list[u].append(v)
                adj_list[v].append(u)
                adj_mat[u][v] = True
                adj_mat[v][u] = True
                colour_of[u][v] = c
                colour_of[v][u] = c
                placed.append([u, v, c])
                break
    return placed

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin under 120s limit
    
    best_edges = []
    max_edges = 0
    
    # Try multiple random restarts to find a denser graph
    # Each restart is very fast, so we can do many
    seed = 42
    while time.time() < deadline:
        edges = _greedy_construct(seed)
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges
        seed += 1
        
    return best_edges
