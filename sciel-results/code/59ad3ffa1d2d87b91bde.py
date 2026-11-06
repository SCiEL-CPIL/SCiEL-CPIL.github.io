"""K(2,2,2)-free 3-graph on 16 vertices: maximize triples."""
import time
import random
from itertools import combinations

V = 16
ALL_TRIPLES = list(combinations(range(V), 3))

def _creates_k222(adj, x, y, z):
    """Check if adding {x,y,z} to G (represented by adj) creates a K(2,2,2)."""
    used = {x, y, z}
    for xp in range(V):
        if xp in used:
            continue
        # Need {xp, y, z} in G
        if z not in adj[xp][y]:
            continue
        for yp in range(V):
            if yp in used or yp == xp:
                continue
            # Need {x, yp, z} in G and {xp, yp, z} in G
            if z not in adj[x][yp]:
                continue
            if z not in adj[xp][yp]:
                continue
            # Need z' such that {x,y,z'}, {xp,y,z'}, {x,yp,z'}, {xp,yp,z'} all in G
            # and z' not in used ∪ {xp, yp}
            candidates = adj[x][y] & adj[xp][y] & adj[x][yp] & adj[xp][yp]
            candidates -= used
            candidates -= {xp, yp}
            if candidates:
                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin
    
    best = set()
    best_size = 0
    
    # Phase 1: Many greedy restarts
    for i in range(5000):
        if time.time() > deadline - 30:
            break
        
        trips_set = set()
        adj = [[set() for _ in range(V)] for _ in range(V)]
        remaining = list(ALL_TRIPLES)
        random.shuffle(remaining)
        
        for t in remaining:
            if time.time() > deadline - 30:
                break
            x, y, z = t
            if not _creates_k222(adj, x, y, z):
                trips_set.add(t)
                adj[x][y].add(z); adj[y][x].add(z)
                adj[x][z].add(y); adj[z][x].add(y)
                adj[y][z].add(x); adj[z][y].add(x)
        
        if len(trips_set) > best_size:
            best = trips_set
            best_size = len(trips_set)
    
    # Phase 2: Local search on the best solution
    if time.time() < deadline - 5 and best_size > 0:
        adj = [[set() for _ in range(V)] for _ in range(V)]
        for t in best:
            x, y, z = t
            adj[x][y].add(z); adj[y][x].add(z)
            adj[x][z].add(y); adj[z][x].add(y)
            adj[y][z].add(x); adj[z][y].add(x)
        
        trips_list = list(best)
        current_size = len(trips_list)
        
        for _ in range(200000):
            if time.time() > deadline - 5:
                break
            
            t = random.choice(ALL_TRIPLES)
            if t in best:
                continue
            x, y, z = t
            if not _creates_k222(adj, x, y, z):
                best.add(t)
                adj[x][y].add(z); adj[y][x].add(z)
                adj[x][z].add(y); adj[z][x].add(y)
                adj[y][z].add(x); adj[z][y].add(x)
                current_size += 1
                trips_list = list(best)
                continue
            
            # Try a swap: remove one random triple, then try to add
            if current_size > 0:
                to_remove = random.choice(trips_list)
                best.remove(to_remove)
                ax, ay, az = to_remove
                adj[ax][ay].discard(az); adj[ay][ax].discard(az)
                adj[ax][az].discard(ay); adj[az][ax].discard(ay)
                adj[ay][az].discard(ax); adj[az][ay].discard(ax)
                current_size -= 1
                
                added = False
                for _ in range(10):
                    t2 = random.choice(ALL_TRIPLES)
                    if t2 in best:
                        continue
                    x2, y2, z2 = t2
                    if not _creates_k222(adj, x2, y2, z2):
                        best.add(t2)
                        adj[x2][y2].add(z2); adj[y2][x2].add(z2)
                        adj[x2][z2].add(y2); adj[z2][x2].add(y2)
                        adj[y2][z2].add(x2); adj[z2][y2].add(x2)
                        current_size += 1
                        added = True
                        break
                
                trips_list = list(best)
    
    return [list(t) for t in sorted(best)]
