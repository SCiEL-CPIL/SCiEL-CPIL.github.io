"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    """Construct a k-regular graph with girth >= g."""
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    # Moore bound lower bound
    if k == 2:
        n_min = g
    else:
        half_g = g // 2
        n_min = 2 * (k ** half_g - 1) // (k - 1)

    # --- Step 1: Guaranteed baseline from LUW construction ---
    baseline_n, baseline_edges = _luw_construct(k, g)
    best_n = baseline_n
    best_edges = baseline_edges

    if k != 3:
        return best_n, best_edges

    # For k=3, g=16: try to find a smaller graph via random + local search
    # Moore bound for k=3, g=16: 2*(2^8 - 1) = 510. So n >= 510 (even => 512)
    # Known best is 936. We try to beat that.
    
    if time.time() < deadline - 15.0:
        # Try randomized construction with local improvement
        best_n, best_edges = _random_search_3reg(
            g, n_min, best_n, deadline, start_time
        )

    return best_n, best_edges


def _random_search_3reg(g, n_min, current_best, deadline, start_time):
    """Randomized search for a 3-regular graph with girth >= g, smaller than current_best."""
    # We'll try several random 3-regular graphs of size n and try to "fix" them
    # by removing short cycles via edge swaps.
    
    # Try sizes from n_min up to current_best - 2 (even numbers)
    # But to save time, sample a few sizes
    candidate_sizes = []
    for n in range(n_min, current_best, 2):
        candidate_sizes.append(n)
    
    # If too many candidates, sample
    if len(candidate_sizes) > 10:
        random.shuffle(candidate_sizes)
        candidate_sizes = sorted(candidate_sizes[:10])
    
    best_n = current_best
    best_edges = None
    
    for n in candidate_sizes:
        if time.time() > deadline - 5.0:
            break
        if n % 2 != 0:
            continue
        # n * 3 must be even => n must be even, which it is
        
        # Try a few random graphs of this size
        for attempt in range(3):
            if time.time() > deadline - 5.0:
                break
            
            edges = _generate_random_3regular(n)
            if edges is None:
                continue
            
            # Try to improve girth via local search
            improved_edges = _local_search_3reg(edges, n, g, deadline)
            if improved_edges is not None:
                # Verify girth
                girth = _compute_girth_fast(improved_edges, n, g)
                if girth >= g:
                    if n < best_n:
                        best_n = n
                        best_edges = improved_edges
                    break  # Found valid for this n, move to next n
    
    if best_edges is not None:
        return best_n, best_edges
    return current_best, None


def _generate_random_3regular(n):
    """Generate a random 3-regular graph on n vertices using configuration model."""
    if n % 2 != 0:
        return None
    if n < 4:
        return None
    
    # Configuration model
    stubs = []
    for i in range(n):
        stubs.extend([i, i, i])
    
    for _ in range(100):  # Try up to 100 times to get a simple graph
        random.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                valid = False
                break
            key = (min(u,v), max(u,v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        if valid:
            return list(edges)
    
    return None


def _local_search_3reg(edges, n, g, deadline):
    """Try to increase girth via edge swaps. Returns edges if girth >= g, else None."""
    edge_set = set((min(u,v), max(u,v)) for u, v in edges)
    
    # Build adjacency
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # Check current girth
    girth = _compute_girth_fast(edges, n, g)
    if girth >= g:
        return edges
    
    # Local search: find short cycles and try to break them
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline - 5.0:
            break
        
        # Find a short cycle
        cycle = _find_short_cycle(adj, n, g)
        if cycle is None:
            # No short cycle found (girth >= g)
            return list(edge_set)
        
        # Try to break the cycle by swapping an edge
        # Pick a random edge on the cycle and try to rewire
        cycle_edges = []
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            cycle_edges.append((min(u,v), max(u,v)))
        
        # Try a random edge on the cycle
        u, v = random.choice(cycle_edges)
        
        # Find a neighbor of u not in cycle and a neighbor of v not in cycle
        # to do a swap
        neighbors_u = [x for x in adj[u] if x != v]
        neighbors_v = [x for x in adj[v] if x != u]
        
        if not neighbors_u or not neighbors_v:
            continue
        
        # Try swapping: remove (u,v), add (u, x) and (v, y) where x in neighbors_u, y in neighbors_v
        # But this changes degree. Instead, do a 2-opt swap:
        # Remove (u,v) and (x,y), add (u,y) and (v,x)
        # This preserves degree.
        
        # Pick x from neighbors_u, y from neighbors_v such that (x,y) is an edge
        # This is complex. Let's try a simpler approach:
        # Just remove one edge from the cycle and try to reconnect.
        
        # Actually, for 3-regular, a simple edge swap is tricky.
        # Let's try: pick two edges (u,v) and (x,y) and swap to (u,y) and (v,x)
        
        # Find a random edge (x,y) in the graph
        all_edges = list(edge_set)
        if len(all_edges) < 2:
            continue
        x, y = random.choice(all_edges)
        if (x, y) == (u, v) or (x, y) == (v, u):
            continue
        
        # Check if swap is valid: (u,y) and (v,x) not already edges, no loops
        if u == y or v == x:
            continue
        if (min(u,y), max(u,y)) in edge_set:
            continue
        if (min(v,x), max(v,x)) in edge_set:
            continue
        
        # Perform swap
        edge_set.remove((u, v))
        edge_set.remove((x, y))
        edge_set.add((min(u,y), max(u,y)))
        edge_set.add((min(v,x), max(v,x)))
        
        # Update adjacency
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        adj[u].append(y)
        adj[y].append(u)
        adj[v].append(x)
        adj[x].append(v)
        
        # Check if girth improved
        new_girth = _compute_girth_fast(list(edge_set), n, g)
        if new_girth < girth:
            # Revert
            edge_set.remove((min(u,y), max(u,y)))
            edge_set.remove((min(v,x), max(v,x)))
            edge_set.add((u, v))
            edge_set.add((x, y))
            adj[u].remove(y)
            adj[y].remove(u)
            adj[v].remove(x)
            adj[x].remove(v)
            adj[u].append(v)
            adj[v].append(u)
            adj[x].append(y)
            adj[y].append(x)
            continue
        
        girth = new_girth
        if girth >= g:
            return list(edge_set)
    
    return None


def _find_short_cycle(adj, n, g):
    """Find a cycle of length < g. Returns list of vertices or None."""
    # BFS from a random start
    start = random.randint(0, n-1)
    dist = [-1] * n
    parent = [-1] * n
    dist[start] = 0
    parent[start] = -2
    queue = deque([start])
    
    while queue:
        u = queue.popleft()
        if dist[u] * 2 >= g:
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
            elif v != parent[u]:
                # Potential cycle
                # Reconstruct cycle
                if dist[u] + dist[v] + 1 < g:
                    # Found a short cycle
                    cycle = [u]
                    # Trace back from u
                    path_u = [u]
                    cur = u
                    while cur != start:
                        cur = parent[cur]
                        path_u.append(cur)
                    path_u.reverse()
                    
                    path_v = [v]
                    cur = v
                    while cur != start:
                        cur = parent[cur]
                        path_v.append(cur)
                    path_v.reverse()
                    
                    # Merge paths
                    cycle = path_u + path_v[-2::-1]
                    return cycle
    return None


def _compute_girth_fast(edges, n, target_g):
    """Compute girth. Returns girth if < target_g, else target_g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    max_depth = target_g // 2
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        parent[start] = -2
        queue = deque([start])
        found_cycle = None
        while queue:
            u = queue.popleft()
            if dist[u] >= max_depth:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if found_cycle is None or cycle_len < found_cycle:
                        found_cycle = cycle_len
                    if found_cycle < target_g:
                        return found_cycle
        if found_cycle is not None and found_cycle < target_g:
            return found_cycle
    return target_g


def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction."""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
