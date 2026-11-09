import random
import time
import sys

N = 20
C = 20

def _check_4cycle_rainbow(colour_of, adj, u, v, c):
    """
    Check if adding edge (u,v) with colour c would create a non-rainbow 4-cycle.
    A 4-cycle containing (u,v) is u-v-x-y-u where x~v, y~u, x~y.
    We need c, col(v,x), col(x,y), col(y,u) all distinct.
    """
    # For each x in adj[v] (x != u), for each y in adj[u] (y != v, y != x):
    #   if x~y, then we have a 4-cycle u-v-x-y-u
    #   colors: c, col(v,x), col(x,y), col(y,u) must be distinct
    
    adj_v = adj[v]
    adj_u = adj[u]
    
    for x in adj_v:
        if x == u:
            continue
        cvx = colour_of[v][x]
        if cvx == c:
            # If cvx == c, we need to make sure no y exists such that x~y
            # Actually, we just need to check: is there any y in adj_u (y!=v, y!=x) with x~y?
            # If yes, then the cycle u-v-x-y-u has colors c, cvx(=c), ... which is not rainbow.
            for y in adj_u:
                if y == v or y == x:
                    continue
                if x in adj[y]:
                    return False
            continue
        
        for y in adj_u:
            if y == v or y == x:
                continue
            if x in adj[y]:
                cxy = colour_of[x][y]
                cyu = colour_of[y][u]
                # Check all four distinct
                if c == cvx or c == cxy or c == cyu:
                    return False
                if cvx == cxy or cvx == cyu:
                    return False
                if cxy == cyu:
                    return False
    return True

def _greedy_build(seed, deadline):
    """
    Greedy construction: try to add all edges, choosing colors to maintain rainbow 4-cycles.
    Uses a smarter order: try edges in an order that's likely to be feasible.
    """
    adj = [set() for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    placed = []
    
    # Precompute all edges
    all_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng = random.Random(seed)
    rng.shuffle(all_edges)
    
    for (u, v) in all_edges:
        if time.time() > deadline:
            break
        
        # Try each colour
        cols = list(range(C))
        rng.shuffle(cols)
        
        for c in cols:
            if _check_4cycle_rainbow(colour_of, adj, u, v, c):
                adj[u].add(v)
                adj[v].add(u)
                colour_of[u][v] = c
                colour_of[v][u] = c
                placed.append([u, v, c])
                break
    
    return placed

def _local_search(edges, adj, colour_of, deadline):
    """
    Try to add more edges by trying different colour assignments or removing/re-adding edges.
    For now, a simple approach: try to add any missing edge with any colour.
    """
    # Build set of existing edges
    edge_set = set()
    for e in edges:
        u, v, c = e
        edge_set.add((min(u,v), max(u,v)))
    
    # Try to add missing edges
    for u in range(N):
        for v in range(u+1, N):
            if (u, v) in edge_set:
                continue
            if time.time() > deadline:
                return edges, adj, colour_of
            
            for c in range(C):
                if _check_4cycle_rainbow(colour_of, adj, u, v, c):
                    adj[u].add(v)
                    adj[v].add(u)
                    colour_of[u][v] = c
                    colour_of[v][u] = c
                    edges.append([u, v, c])
                    edge_set.add((u, v))
                    break
    return edges, adj, colour_of

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin under 150s limit
    
    best_edges = []
    max_edges = 0
    
    # Strategy: multiple random restarts with greedy construction
    # Then local search to add more edges
    
    seed = 0
    while time.time() < deadline - 5.0:
        edges = _greedy_build(seed, deadline)
        
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges[:]
        
        # Try local search on the best so far
        if time.time() < deadline - 2.0:
            # Reconstruct adj and colour_of from best_edges
            adj = [set() for _ in range(N)]
            colour_of = [[-1] * N for _ in range(N)]
            for e in best_edges:
                u, v, c = e
                adj[u].add(v)
                adj[v].add(u)
                colour_of[u][v] = c
                colour_of[v][u] = c
            
            improved_edges, _, _ = _local_search(best_edges[:], adj, colour_of, deadline)
            if len(improved_edges) > max_edges:
                max_edges = len(improved_edges)
                best_edges = improved_edges[:]
        
        seed += 1
    
    return best_edges
