import random
import time

N = 20
C = 20
# Safety margin: start + 110s, well under 150s limit
# Note: time.time() is called at module load, but construct() might be called later.
# It is safer to calculate deadline inside construct or use a global offset.
# However, the current program uses a global DEADLINE. I will use a local deadline 
# in construct to be precise relative to the call time.

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
        
        # Early exit: if cvx == c, we need to ensure no such y exists?
        # No, if cvx == c, then for ANY y forming a cycle, the colors c and cvx are equal.
        # So if there is ANY y in adj_u such that y != v, y != x, and (x,y) is an edge, it fails.
        # We can optimize this: if cvx == c, check if such y exists. If yes, return False.
        # If no such y exists, continue to next x.
        if cvx == c:
            # Check if there is any y in adj_u (y != v, y != x) such that (x,y) is an edge
            found_y = False
            for y in adj_u:
                if y == v or y == x:
                    continue
                if x in adj[y]:
                    found_y = True
                    break
            if found_y:
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


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin under 120s limit
    
    # 1. Start with the structured K(10,10) construction from Inspiration 2
    # Vertices 0..9 in U, vertices 10..19 in V
    U = list(range(0, 10))
    V = list(range(10, 20))
    
    adj = [set() for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    placed = []
    
    # Add all edges between U and V
    for u in U:
        for v in V:
            v_prime = v - 10
            c = (u + 10 * v_prime) % 20
            adj[u].add(v)
            adj[v].add(u)
            colour_of[u][v] = c
            colour_of[v][u] = c
            placed.append([u, v, c])
            
    # 2. Try to add intra-partition edges (U-U and V-V)
    # Generate list of potential edges: (u, v) where u, v in U or u, v in V
    potential_edges = []
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            potential_edges.append((U[i], U[j]))
    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            potential_edges.append((V[i], V[j]))
            
    rng = random.Random(42)
    rng.shuffle(potential_edges)
    
    # Greedy addition of remaining edges
    for (u, v) in potential_edges:
        if time.time() > deadline - 5.0:
            break
            
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
        # If no color worked, we skip this edge
        
    return placed

