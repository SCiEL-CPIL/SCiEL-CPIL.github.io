"""Triangle-free diameter-four augmentation via hub-based construction."""
# EVOLVE-BLOCK-START

import time

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}


def _validate_fast(added_edges):
    """Fast validation: triangle-free (bipartite), connected, diameter <= 4."""
    n = 24
    base_edge_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    
    added_set = set()
    for u, v in added_edges:
        if not (0 <= u <= 23 and 0 <= v <= 23) or u == v:
            return False
        key = (min(u, v), max(u, v))
        if key in base_edge_set or key in added_set:
            return False
        added_set.add(key)
        # Check bipartite: one even, one odd
        if (u % 2) == (v % 2):
            return False
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    for u, v in added_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # BFS from each vertex to check connectivity and diameter
    max_d = 0
    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for d in dist:
            if d == -1:
                return False
            if d > max_d:
                max_d = d
                if max_d > 4:
                    return False
    return True


def _generate_hub_config(hub_even, hub_odd, budget):
    """Generate a hub-based configuration with given hubs and edge budget.
    
    hub_even: even vertex used as hub (connects to odd vertices)
    hub_odd: odd vertex used as hub (connects to even vertices)
    budget: max number of edges to use
    
    Strategy: each hub connects to vertices it can't reach within 2 steps via path.
    """
    n = 24
    base_edge_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
    
    # Path distances
    def path_dist(a, b):
        return abs(a - b)
    
    edges = []
    
    # Check if hubs should be connected
    if path_dist(hub_even, hub_odd) > 2:
        edges.append([hub_even, hub_odd])
    
    # For hub_even (even): connect to odd vertices where path_dist > 2
    # and where going through hub_odd doesn't help
    odd_verts = [i for i in range(1, 24, 2)]
    for o in odd_verts:
        if o == hub_odd:
            continue
        if (min(hub_even, o), max(hub_even, o)) in base_edge_set:
            continue
        # Can we reach o within 2 steps without this edge?
        # Via path: path_dist(hub_even, o)
        # Via hub_odd: 1 (hub_even->hub_odd) + path_dist(hub_odd, o)
        # But hub_odd is odd, so hub_odd->o requires o to be even... no, hub_odd connects to even.
        # So from hub_even to o (odd): direct (1), or via path, or via hub_odd?
        # hub_even -> hub_odd (1 step) -> ... -> o: hub_odd is odd, o is odd, can't go direct.
        # Need hub_odd -> some_even -> o. That's 2 more steps.
        # So via hub_odd: 1 + 1 + path_dist(some_even, o) where some_even is adjacent to hub_odd
        # This is complex. Let's just check if path_dist(hub_even, o) <= 2.
        if path_dist(hub_even, o) <= 2:
            continue
        # Check if reachable via hub_odd in 3 steps: hub_even->hub_odd->e->o where e adj to hub_odd
        # hub_odd's path neighbors: hub_odd-1, hub_odd+1 (even)
        reachable_via_hub = False
        for e in [hub_odd - 1, hub_odd + 1]:
            if 0 <= e <= 23 and path_dist(e, o) <= 1:
                reachable_via_hub = True
                break
        if reachable_via_hub and len(edges) < budget - 1:
            # Can reach in 3 steps via hub_odd, so maybe don't need direct edge
            # But we still might need it for diameter 4 from other vertices
            pass  # Skip for now, add later if needed
        edges.append([hub_even, o])
    
    # For hub_odd (odd): connect to even vertices where path_dist > 2
    even_verts = [i for i in range(0, 24, 2)]
    for e in even_verts:
        if e == hub_even:
            continue
        if (min(hub_odd, e), max(hub_odd, e)) in base_edge_set:
            continue
        if path_dist(hub_odd, e) <= 2:
            continue
        reachable_via_hub = False
        for o in [hub_even - 1, hub_even + 1]:
            if 0 <= o <= 23 and path_dist(o, e) <= 1:
                reachable_via_hub = True
                break
        if reachable_via_hub and len(edges) < budget - 1:
            pass
        edges.append([hub_odd, e])
    
    return edges


def construct():
    """
    Find minimum edges for triangle-free diameter-4 augmentation of P24.
    
    Strategy:
    1. Enumerate hub-pair configurations (even hub, odd hub)
    2. For each, generate a candidate edge set
    3. Validate and keep best
    4. Local search: try removing/swapping edges
    """
    deadline = time.time() + 85  # 5s safety margin
    
    n = 24
    even_hubs = list(range(0, 24, 2))  # 0,2,4,...,22
    odd_hubs = list(range(1, 24, 2))   # 1,3,5,...,23
    
    best_edges = None
    best_count = 10  # Start with known 10-edge solution
    
    # Known good solution
    fallback = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
    if _validate_fast(fallback):
        best_edges = fallback
        best_count = 10
    
    # Try single-hub configurations first (one hub connects to many)
    # Single even hub h: connects to odd vertices
    # Single odd hub h: connects to even vertices
    
    # For a single even hub h, which odd vertices does it need to connect to?
    # An odd vertex o is "covered" if dist(h, o) <= 2 in the augmented graph.
    # Without added edges, dist(h, o) = |h-o| on the path.
    # With hub h connected to some odd vertices, other even vertices can reach
    # those odd vertices in 2 steps.
    
    # Key insight: for diameter 4, every pair must be within 4.
    # Farthest pairs in path are (0,23), (0,22), etc.
    
    # Let's try all pairs of hubs (one even, one odd) and see what edges are needed.
    
    for he in even_hubs:
        if time.time() > deadline:
            break
        for ho in odd_hubs:
            if time.time() > deadline:
                break
            
            # Generate candidate: connect hubs, then add edges as needed
            # Start with just the hub-hub edge (if not adjacent)
            edges = []
            if abs(he - ho) > 1:
                edges.append([he, ho])
            
            # Determine which additional edges are needed
            # For each pair (u,v) with path_dist > 4, check if they can be
            # connected within 4 steps using the hub edges.
            
            # Build tentative adjacency
            adj = [set() for _ in range(n)]
            for u, v in P['edges']:
                adj[u].add(v)
                adj[v].add(u)
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
            
            # Check which pairs are still > 4
            # For efficiency, only check pairs with path_dist > 4
            needs_edge = set()
            for u in range(n):
                for v in range(u + 1, n):
                    if abs(u - v) <= 4:
                        continue
                    # BFS from u to v, limited to depth 4
                    dist = [-1] * n
                    dist[u] = 0
                    queue = [u]
                    head = 0
                    while head < len(queue) and dist[queue[head]] < 4:
                        node = queue[head]
                        head += 1
                        for nb in adj[node]:
                            if dist[nb] == -1:
                                dist[nb] = dist[node] + 1
                                if dist[nb] <= 4:
                                    queue.append(nb)
                    if dist[v] == -1 or dist[v] > 4:
                        needs_edge.add((u, v))
            
            if not needs_edge:
                if len(edges) < best_count:
                    if _validate_fast(edges):
                        best_edges = [list(e) for e in edges]
                        best_count = len(edges)
                continue
            
            # For each needed pair, find which hub edge would cover it
            # Greedy: add edges that cover the most uncovered pairs
            remaining = set(needs_edge)
            current_edges = list(edges)
            current_adj = [set(s) for s in adj]
            
            while remaining and time.time() < deadline:
                if len(current_edges) >= best_count:
                    break
                
                # Find the edge that covers the most remaining pairs
                best_e = None
                best_cover = 0
                
                # Candidate edges: hub_even to odd, hub_odd to even
                candidates = []
                for o in range(1, 24, 2):
                    if o != ho and abs(he - o) > 1:
                        key = (min(he, o), max(he, o))
                        if key not in set((min(e[0],e[1]),max(e[0],e[1])) for e in current_edges):
                            candidates.append([he, o])
                for e in range(0, 24, 2):
                    if e != he and abs(ho - e) > 1:
                        key = (min(ho, e), max(ho, e))
                        if key not in set((min(e2[0],e2[1]),max(e2[0],e2[1])) for e2 in current_edges):
                            candidates.append([ho, e])
                
                for cand in candidates:
                    if time.time() > deadline:
                        break
                    u, v = cand
                    # Temporarily add
                    current_adj[u].add(v)
                    current_adj[v].add(u)
                    
                    # Count covered pairs
                    cover = 0
                    for (a, b) in remaining:
                        if abs(a - b) <= 4:
                            cover += 1
                            continue
                        dist = [-1] * n
                        dist[a] = 0
                        queue = [a]
                        head = 0
                        while head < len(queue) and dist[queue[head]] < 4:
                            node = queue[head]
                            head += 1
                            for nb in current_adj[node]:
                                if dist[nb] == -1:
                                    dist[nb] = dist[node] + 1
                                    if dist[nb] <= 4:
                                        queue.append(nb)
                        if dist[b] != -1 and dist[b] <= 4:
                            cover += 1
                    
                    # Remove
                    current_adj[u].discard(v)
                    current_adj[v].discard(u)
                    
                    if cover > best_cover:
                        best_cover = cover
                        best_e = cand
                
                if best_e is None or best_cover == 0:
                    break
                
                current_edges.append(best_e)
                current_adj[best_e[0]].add(best_e[1])
                current_adj[best_e[1]].add(best_e[0])
                
                # Update remaining
                new_remaining = set()
                for (a, b) in remaining:
                    if abs(a - b) <= 4:
                        continue
                    dist = [-1] * n
                    dist[a] = 0
                    queue = [a]
                    head = 0
                    while head < len(queue) and dist[queue[head]] < 4:
                        node = queue[head]
                        head += 1
                        for nb in current_adj[node]:
                            if dist[nb] == -1:
                                dist[nb] = dist[node] + 1
                                if dist[nb] <= 4:
                                    queue.append(nb)
                    if dist[b] == -1 or dist[b] > 4:
                        new_remaining.add((a, b))
                remaining = new_remaining
            
            if not remaining and len(current_edges) < best_count:
                if _validate_fast(current_edges):
                    best_edges = [list(e) for e in current_edges]
                    best_count = len(current_edges)
    
    # Local search on best: try removing edges, then swapping
    if best_edges is not None:
        # Try removing edges
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(best_edges)):
                if time.time() > deadline:
                    break
                trial = best_edges[:i] + best_edges[i+1:]
                if _validate_fast(trial):
                    best_edges = trial
                    best_count = len(trial)
                    improved = True
                    break
        
        # Try 1-swap: remove one, add another
        if time.time() < deadline:
            base_edge_set = set((min(u, v), max(u, v)) for u, v in P['edges'])
            current_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in best_edges)
            
            # All valid candidate edges (even-odd, not in base, not already added)
            all_candidates = []
            for e in range(0, 24, 2):
                for o in range(1, 24, 2):
                    key = (min(e, o), max(e, o))
                    if key not in base_edge_set and key not in current_set:
                        all_candidates.append([e, o])
            
            for i in range(len(best_edges)):
                if time.time() > deadline:
                    break
                removed = best_edges[i]
                trial_base = best_edges[:i] + best_edges[i+1:]
                trial_set = set((min(e[0], e[1]), max(e[0], e[1])) for e in trial_base)
                
                for cand in all_candidates:
                    if time.time() > deadline:
                        break
                    key = (min(cand[0], cand[1]), max(cand[0], cand[1]))
                    if key in trial_set:
                        continue
                    trial = trial_base + [cand]
                    if _validate_fast(trial):
                        best_edges = trial
                        best_count = len(best_edges)
                        break
                if time.time() > deadline:
                    break
    
    # Convert to proper format
    if best_edges is None:
        best_edges = fallback
    
    result = []
    for edge in best_edges:
        result.append([int(edge[0]), int(edge[1])])
    
    return result

# EVOLVE-BLOCK-END
