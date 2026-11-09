import random
import time

N = 20
C = 20
DEADLINE = time.time() + 110.0


def _edge_ok(adj, colour_of, u, v, c):
    """Check whether adding edge (u,v) with colour c keeps every 4-cycle rainbow."""
    adj_v = adj[v]
    adj_u = adj[u]
    
    for x in adj_v:
        if x == u:
            continue
        cvx = colour_of[v][x]
        for y in adj_u:
            if y == v or y == x:
                continue
            if x in adj[y]:
                cxy = colour_of[x][y]
                cyu = colour_of[y][u]
                if c == cvx or c == cxy or c == cyu:
                    return False
                if cvx == cxy or cvx == cyu:
                    return False
                if cxy == cyu:
                    return False
    return True


def _try_build(edge_order, rng):
    adj = [set() for _ in range(N)]
    colour_of = [[-1] * N for _ in range(N)]
    placed = []
    skipped = []
    
    order = list(edge_order)
    rng.shuffle(order)
    
    for (u, v) in order:
        if time.time() > DEADLINE - 10.0:
            break
        
        if u in adj[v]:
            continue
            
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
        
        if not added:
            skipped.append((u, v))
    
    return adj, colour_of, placed, skipped


def _local_improve(adj, colour_of, placed, skipped, rng):
    """Try to add skipped edges and swap colors to free up space."""
    improved = True
    while improved and time.time() < DEADLINE - 5.0:
        improved = False
        
        # Try to add skipped edges
        still_skipped = []
        for (u, v) in skipped:
            if u in adj[v]:
                continue
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
                    improved = True
                    break
            if not added:
                still_skipped.append((u, v))
        skipped = still_skipped
        
        # Try color swaps: for each edge, try swapping its color with another
        # This can free up a color for a skipped edge
        for i in range(len(placed)):
            if time.time() > DEADLINE - 5.0:
                break
            u, v, c = placed[i]
            # Try swapping color c with another color c2 on this edge
            for c2 in range(C):
                if c2 == c:
                    continue
                # Check if changing (u,v) to c2 is valid
                old_c = c
                colour_of[u][v] = c2
                colour_of[v][u] = c2
                
                valid = True
                # Check all 4-cycles containing (u,v)
                adj_v = adj[v]
                adj_u = adj[u]
                for x in adj_v:
                    if x == u:
                        continue
                    cvx = colour_of[v][x]
                    for y in adj_u:
                        if y == v or y == x:
                            continue
                        if x in adj[y]:
                            cxy = colour_of[x][y]
                            cyu = colour_of[y][u]
                            if c2 == cvx or c2 == cxy or c2 == cyu:
                                valid = False
                                break
                            if cvx == cxy or cvx == cyu:
                                valid = False
                                break
                            if cxy == cyu:
                                valid = False
                                break
                    if not valid:
                        break
                
                if valid:
                    # Check if this helps add a skipped edge
                    for (su, sv) in skipped:
                        if su in adj[sv]:
                            continue
                        if _edge_ok(adj, colour_of, su, sv, old_c):
                            # Apply the swap and add the skipped edge
                            placed[i] = [u, v, c2]
                            adj[su].add(sv)
                            adj[sv].add(su)
                            colour_of[su][sv] = old_c
                            colour_of[sv][su] = old_c
                            placed.append([su, sv, old_c])
                            skipped.remove((su, sv))
                            improved = True
                            break
                    else:
                        # No benefit, revert
                        colour_of[u][v] = old_c
                        colour_of[v][u] = old_c
                else:
                    # Revert
                    colour_of[u][v] = old_c
                    colour_of[v][u] = old_c
    
    return adj, colour_of, placed, skipped


def construct():
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    best = []
    rng = random.Random(12345)
    
    while time.time() < DEADLINE - 15.0:
        adj, colour_of, placed, skipped = _try_build(all_edges, rng)
        adj, colour_of, placed, skipped = _local_improve(adj, colour_of, placed, skipped, rng)
        
        if len(placed) > len(best):
            best = placed
    
    return best
