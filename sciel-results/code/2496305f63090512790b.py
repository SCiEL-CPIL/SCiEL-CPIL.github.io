import time
import random

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # Safety margin
    
    n = 16
    m = 65
    
    # Bitmask representation
    adj = [0] * n
    deg = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        deg[u] += 1
        deg[v] += 1
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        deg[u] -= 1
        deg[v] -= 1
        
    def has_edge(u, v):
        return (adj[u] >> v) & 1
        
    def _score():
        best = 0
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if (au >> v) & 1:
                    common = au & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        s = deg[u] + deg[v] + deg[w]
                        if s > best:
                            best = s
                        c ^= lsb
        return best

    # Initialize: K_{8,8} + one edge in A
    for u in range(8):
        for v in range(8, 16):
            add_edge(u, v)
    add_edge(0, 1)
    
    cur_score = _score()
    best_score = cur_score
    best_adj = adj[:]
    best_deg = deg[:]
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    all_edges_set = set()
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                all_edges_set.add((u, v))
    
    def pick_random_edge():
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and has_edge(u, v):
                return u, v
                
    def pick_random_non_edge():
        while True:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and not has_edge(u, v):
                return u, v

    # Get list of all triangles and their degree sums
    def get_triangles():
        tris = []
        for u in range(n):
            au = adj[u]
            for v in range(u + 1, n):
                if (au >> v) & 1:
                    common = au & adj[v]
                    c = common
                    while c:
                        lsb = c & -c
                        w = lsb.bit_length() - 1
                        if w > v:  # ensure u < v < w
                            s = deg[u] + deg[v] + deg[w]
                            tris.append((u, v, w, s))
                        c ^= lsb
        return tris

    # Try all single swaps systematically
    def try_all_swaps():
        nonlocal cur_score, best_score, best_adj, best_deg, all_edges_set
        improved = False
        edges_list = list(all_edges_set)
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in all_edges_set]
        
        for e1 in edges_list:
            for e2 in non_edges:
                # Check if swap is valid (doesn't create duplicate)
                u1, v1 = e1
                u2, v2 = e2
                if (u2, v2) in all_edges_set:
                    continue
                # Also check we're not removing an edge that's the same as adding
                if {u1, v1} == {u2, v2}:
                    continue
                
                # Perform swap
                remove_edge(u1, v1)
                add_edge(u2, v2)
                all_edges_set.discard(e1)
                all_edges_set.add(e2)
                
                new_score = _score()
                
                if new_score < cur_score:
                    cur_score = new_score
                    improved = True
                    if cur_score < best_score:
                        best_score = cur_score
                        best_adj = adj[:]
                        best_deg = deg[:]
                else:
                    # Revert
                    remove_edge(u2, v2)
                    add_edge(u1, v1)
                    all_edges_set.discard(e2)
                    all_edges_set.add(e1)
                    
                if time.time() - t0 > DEADLINE - 5:
                    return improved
        
        return improved

    # Simulated annealing with incremental improvements
    no_improve = 0
    max_no_improve = 500
    T = 3.0
    cool_rate = 0.998
    
    while time.time() - t0 < DEADLINE:
        # First try systematic swap search
        if no_improve > 100:
            if try_all_swaps():
                no_improve = 0
                T = 3.0
            else:
                # Perturb: break the worst triangle
                tris = get_triangles()
                if tris:
                    max_tris = [t for t in tris if t[3] == max(t2[3] for t2 in tris)]
                    if max_tris:
                        u, v, w, _ = max_tris[random.randint(0, len(max_tris) - 1)]
                        # Remove one edge from the worst triangle
                        edge_to_remove = random.choice([(u, v), (u, w), (v, w)])
                        if edge_to_remove in all_edges_set:
                            remove_edge(edge_to_remove[0], edge_to_remove[1])
                            all_edges_set.discard(edge_to_remove)
                            # Add a random non-edge
                            while True:
                                u2 = random.randint(0, n - 1)
                                v2 = random.randint(0, n - 1)
                                if u2 != v2 and (min(u2, v2), max(u2, v2)) not in all_edges_set:
                                    add_edge(u2, v2)
                                    all_edges_set.add((min(u2, v2), max(u2, v2)))
                                    break
                            cur_score = _score()
                            no_improve = 0
                            T = 3.0
                continue
        
        # Random swap with SA acceptance
        u1, v1 = pick_random_edge()
        u2, v2 = pick_random_non_edge()
        
        e1 = (min(u1, v1), max(u1, v1))
        e2 = (min(u2, v2), max(u2, v2))
        
        # Perform swap
        remove_edge(u1, v1)
        add_edge(u2, v2)
        all_edges_set.discard(e1)
        all_edges_set.add(e2)
        
        new_score = _score()
        
        if new_score < cur_score:
            cur_score = new_score
            no_improve = 0
            if cur_score < best_score:
                best_score = cur_score
                best_adj = adj[:]
                best_deg = deg[:]
        else:
            # Simulated Annealing acceptance
            delta = new_score - cur_score
            if delta <= 1 and random.random() < 0.15:
                cur_score = new_score
                no_improve += 1
            else:
                # Revert
                remove_edge(u2, v2)
                add_edge(u1, v1)
                all_edges_set.discard(e2)
                all_edges_set.add(e1)
                no_improve += 1
        
        T *= cool_rate
        if T < 0.3:
            T = 0.3
            
        if no_improve > max_no_improve:
            # Restart with perturbation
            for _ in range(3):
                eu1, ev1 = pick_random_edge()
                eu2, ev2 = pick_random_non_edge()
                ee1 = (min(eu1, ev1), max(eu1, ev1))
                ee2 = (min(eu2, ev2), max(eu2, ev2))
                remove_edge(eu1, ev1)
                add_edge(eu2, ev2)
                all_edges_set.discard(ee1)
                all_edges_set.add(ee2)
            cur_score = _score()
            no_improve = 0
            T = 3.0
            max_no_improve = 500
            
    # Reconstruct edges from best_adj
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return {'n': n, 'edges': edges}

