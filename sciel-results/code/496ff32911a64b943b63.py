import random
import time

N = 20
C = 20
DEADLINE = time.time() + 110.0  # Safety margin: start + 110s, well under 150s


def _edge_ok(adj, colour_of, u, v, c):
    """Check whether adding edge (u,v) with colour c keeps every 4-cycle rainbow."""
    # A 4-cycle containing (u,v) is u-v-x-y-u
    # x is neighbor of v, y is neighbor of u
    # (x,y) must be an edge
    # Colors: c_uv=c, c_vx, c_xy, c_yu must be distinct
    
    adj_v = adj[v]
    adj_u = adj[u]
    
    for x in adj_v:
        if x == u:
            continue
        cvx = colour_of[v][x]
        
        if cvx == c:
            # If cvx == c, then any y that forms a cycle with x will fail because c==cvx
            # Check if such y exists
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


def _init_bipartite():
    """Initialize with the valid K(10,10) construction from Inspiration 3."""
    adj = [set() for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    placed = []
    
    U = list(range(0, 10))
    V = list(range(10, 20))
    
    for u in U:
        for v in V:
            v_prime = v - 10
            c = (u + 10 * v_prime) % 20
            adj[u].add(v)
            adj[v].add(u)
            colour_of[u][v] = c
            colour_of[v][u] = c
            placed.append([u, v, c])
            
    return adj, colour_of, placed

def _try_extend(seed, adj_init, colour_of_init, placed_init, remaining_edges, rng):
    """Try to add remaining edges to the initial bipartite structure."""
    adj = [set(s) for s in adj_init]
    colour_of = [[c for c in row] for row in colour_of_init]
    placed = list(placed_init)
    
    order = list(remaining_edges)
    rng.shuffle(order)
    
    for (u, v) in order:
        if time.time() > DEADLINE - 5.0:
            break
            
        if u in adj[v]:
            continue
            
        # Try colors in a random order
        cols = list(range(C))
        rng.shuffle(cols)
        
        added = False
        for c in cols:
            if _edge_ok(adj, colour_of, u, v, c):
                adj[u].add(v)
                adj[v].add(u)
                colour_of[u][v] = c
                colour_of[v][u] = c
                placed.append([u, v, c])
                added = True
                break
    return placed

def construct():
    # 1. Initialize with the known valid 100-edge bipartite graph
    adj_init, colour_of_init, placed_init = _init_bipartite()
    
    # 2. Identify remaining edges (intra-U and intra-V)
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    existing_edges = set((u, v) for u, v, c in placed_init)
    remaining_edges = [e for e in all_edges if e not in existing_edges]
    
    best = placed_init
    best_len = len(placed_init)
    
    rng = random.Random(12345)
    
    # 3. Run multiple restarts of the greedy extension
    # Each restart shuffles the order of remaining edges
    # The base structure is always the same, but the order of trying to add
    # intra-partition edges affects which colors are available and which cycles form.
    
    while time.time() < DEADLINE - 10.0:
        seed = rng.randint(0, 10**9)
        cand = _try_extend(seed, adj_init, colour_of_init, placed_init, remaining_edges, rng)
        
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            
            # If we found a perfect graph (190 edges), we can stop
            if best_len == 190:
                break

    return best
