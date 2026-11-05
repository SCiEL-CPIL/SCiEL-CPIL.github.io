import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    n = 20
    num_colors = 20
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {(u, v): i for i, (u, v) in enumerate(all_edges)}
    m = len(all_edges)
    
    # Precompute for each pair (u,v) the list of common neighbors (a,b) where a~u, b~v, a!=b, and a~b
    # This is expensive to precompute fully, so we'll compute on-the-fly but cache
    # Actually, let's precompute common_neighborhoods for speed
    # common_neighbors[u][v] = set of w such that w~u and w~v (in the full K_n)
    # But we need dynamic info. Let's just optimize the inner loop.
    
    best_count = 0
    best_edges = []
    
    def greedy_build(seed, init_edges=None):
        rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        edges = []
        
        if init_edges:
            for u, v, c in init_edges:
                adj[u][v] = c
                adj[v][u] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
                edges.append([u, v, c])
            existing = set((min(u,v), max(u,v)) for u, v, c in edges)
            remaining = [e for e in all_edges if e not in existing]
        else:
            remaining = all_edges[:]
        
        rng.shuffle(remaining)
        
        # Use bitmask for forbidden colors for speed
        for (u, v) in remaining:
            forbidden_mask = 0
            feasible = True
            
            nu = neighbors[u]
            nv = neighbors[v]
            
            for a in nu:
                if a == v:
                    continue
                ca = adj[u][a]
                for b in nv:
                    if b == u or b == a:
                        continue
                    if adj[a][b] != -1:
                        cb = adj[a][b]
                        cv = adj[b][v]
                        if ca == cb or cb == cv or ca == cv:
                            feasible = False
                            break
                        forbidden_mask |= (1 << ca) | (1 << cb) | (1 << cv)
                if not feasible:
                    break
            
            if feasible:
                # Find available colors using bitmask
                all_mask = (1 << num_colors) - 1
                available = all_mask & ~forbidden_mask
                if available:
                    # Pick a random available color
                    # Get list of available color indices
                    avail_list = [c for c in range(num_colors) if (available >> c) & 1]
                    c = rng.choice(avail_list)
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
                elif not forbidden_mask:
                    c = rng.randrange(num_colors)
                    adj[u][v] = c
                    adj[v][u] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    edges.append([u, v, c])
        
        return edges
    
    def check_validity(edges):
        """Verify all 4-cycles are rainbow."""
        adj = [[-1] * n for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
        
        for u in range(n):
            for v in range(n):
                if u == v or adj[u][v] == -1:
                    continue
                c1 = adj[u][v]
                for w in range(n):
                    if w == u or w == v or adj[v][w] == -1:
                        continue
                    c2 = adj[v][w]
                    if c2 == c1:
                        # Can't have same color on adjacent edges in a 4-cycle? 
                        # Actually in a 4-cycle u-v-w-x-u, edges are (u,v),(v,w),(w,x),(x,u)
                        # c1 and c2 are adjacent, they must be different for rainbow
                        continue
                    for x in range(n):
                        if x == u or x == v or x == w:
                            continue
                        if adj[w][x] == -1 or adj[x][u] == -1:
                            continue
                        c3 = adj[w][x]
                        c4 = adj[x][u]
                        if c3 == c4 or c3 == c1 or c3 == c2 or c4 == c1 or c4 == c2:
                            return False
        return True
    
    def recolor_search(edges, seed, time_limit):
        """Try to improve by re-coloring existing edges to unlock new additions."""
        rng = random.Random(seed)
        adj = [[-1] * n for _ in range(n)]
        neighbors = [set() for _ in range(n)]
        for u, v, c in edges:
            adj[u][v] = c
            adj[v][u] = c
            neighbors[u].add(v)
            neighbors[v].add(u)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try re-coloring each edge
            for idx, (u, v, old_c) in enumerate(edges):
                if time.time() > time_limit:
                    break
                # Try each new color for this edge
                for new_c in range(num_colors):
                    if new_c == old_c:
                        continue
                    # Check if re-coloring is valid: all 4-cycles containing (u,v) must remain rainbow
                    valid = True
                    for a in neighbors[u]:
                        if a == v:
                            continue
                        ca = adj[u][a]
                        for b in neighbors[v]:
                            if b == u or b == a:
                                continue
                            if adj[a][b] != -1:
                                cb = adj[a][b]
                                cv = adj[b][v]
                                # The 4-cycle is (a,u,v,b) with colors ca, new_c, cv, cb
                                if ca == new_c or new_c == cv or cv == cb or ca == cb:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        continue
                    
                    # Re-color is valid; check if it allows adding a new edge
                    # Try to add one missing edge
                    added = False
                    for (wu, wv) in all_edges:
                        if adj[wu][wv] != -1:
                            continue
                        # Check if we can add edge (wu, wv) with some color
                        forbidden_add = set()
                        feasible_add = True
                        for a in neighbors[wu]:
                            if a == wv:
                                continue
                            ca = adj[wu][a]
                            for b in neighbors[wv]:
                                if b == wu or b == a:
                                    continue
                                if adj[a][b] != -1:
                                    cb = adj[a][b]
                                    cv = adj[b][wv]
                                    if ca == cb or cb == cv or ca == cv:
                                        feasible_add = False
                                        break
                                    forbidden_add.add(ca)
                                    forbidden_add.add(cb)
                                    forbidden_add.add(cv)
                            if not feasible_add:
                                break
                        if not feasible_add:
                            continue
                        # Pick a color for the new edge
                        candidates = [c for c in range(num_colors) if c not in forbidden_add]
                        if candidates:
                            nc = rng.choice(candidates)
                            # Apply re-color and add
                            adj[u][v] = new_c
                            adj[v][u] = new_c
                            adj[wu][wv] = nc
                            adj[wv][wu] = nc
                            neighbors[wu].add(wv)
                            neighbors[wv].add(wu)
                            edges[idx] = [u, v, new_c]
                            edges.append([wu, wv, nc])
                            improved = True
                            added = True
                            break
                        else:
                            feasible_add = False
                        if not feasible_add:
                            continue
                    if added:
                        break
            # After re-coloring, try to add more edges greedily
            if improved:
                existing = set((min(u,v), max(u,v)) for u, v, c in edges)
                remaining = [e for e in all_edges if e not in existing]
                rng.shuffle(remaining)
                for (u, v) in remaining:
                    if time.time() > time_limit:
                        break
                    forbidden = set()
                    feasible = True
                    for a in neighbors[u]:
                        if a == v:
                            continue
                        ca = adj[u][a]
                        for b in neighbors[v]:
                            if b == u or b == a:
                                continue
                            if adj[a][b] != -1:
                                cb = adj[a][b]
                                cv = adj[b][v]
                                if ca == cb or cb == cv or ca == cv:
                                    feasible = False
                                    break
                                forbidden.add(ca)
                                forbidden.add(cb)
                                forbidden.add(cv)
                            if not feasible:
                                break
                    if not feasible:
                        continue
                    candidates = [c for c in range(num_colors) if c not in forbidden]
                    if candidates:
                        c = rng.choice(candidates)
                        adj[u][v] = c
                        adj[v][u] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        edges.append([u, v, c])
        
        return edges
    
    # Phase 1: Greedy random restarts
    num_restarts = 300
    for i in range(num_restarts):
        if time.time() > deadline:
            break
        edges = greedy_build(i)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
    
    # Phase 2: Local search — remove k edges, refill greedily
    current_best = best_edges[:]
    
    while time.time() < deadline:
        k = random.randint(1, 8)
        if len(current_best) <= k:
            break
        
        to_remove = random.sample(current_best, k)
        rem_set = set((min(u,v), max(u,v)) for u, v, c in to_remove)
        base = [e for e in current_best if (min(e[0], e[1]), max(e[0], e[1])) not in rem_set]
        
        seed = random.randint(0, 10**9)
        new_edges = greedy_build(seed, base)
        
        if len(new_edges) > best_count:
            best_count = len(new_edges)
            best_edges = new_edges[:]
            current_best = new_edges[:]
    
    # Phase 3: Re-coloring search to unlock new edges
    if time.time() < deadline - 5:
        recolor_limit = min(deadline - 2, time.time() + 30)
        recolored = recolor_search(current_best[:], 42, recolor_limit)
        if len(recolored) > best_count:
            best_count = len(recolored)
            best_edges = recolored[:]
            current_best = recolored[:]
    
    # Final verification - ensure the result is valid
    if best_edges and not check_validity(best_edges):
        # Find a valid fallback
        best_edges = []
        for i in range(50):
            candidate = greedy_build(1000 + i)
            if check_validity(candidate):
                best_edges = candidate
                break
        if not best_edges:
            best_edges = [[0, v, v % 20] for v in range(1, 20)]
    
    if not best_edges:
        best_edges = [[0, v, v % 20] for v in range(1, 20)]
    
    return best_edges
