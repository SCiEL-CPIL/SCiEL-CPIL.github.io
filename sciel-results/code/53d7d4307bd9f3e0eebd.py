import time
import random
import heapq
from collections import deque

def _compute_girth(adjacency, max_girth):
    """Compute girth by BFS from each vertex, tracking shortest cycle through that vertex.
    Returns the girth (shortest cycle length) or infinity if none found up to max_girth.
    """
    n = len(adjacency)
    min_girth = max_girth + 1
    
    for start in range(n):
        # BFS from start, tracking distance and parent
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= max_girth - 1:
                continue
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle: u -> ... -> v and v is already visited
                    # Cycle length = dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
    
    return min_girth

def _has_short_cycle_after_switch(adjacency, u1, v1, u2, v2, g):
    """Check if removing (u1,v1) and (u2,v2) and adding (u1,u2) and (v1,v2) 
    creates any cycle shorter than g.
    We need to check:
    1. No short cycle in the modified graph
    2. Specifically, check paths between the four vertices
    """
    # Temporarily modify adjacency
    adjacency[u1].remove(v1)
    adjacency[v1].remove(u1)
    adjacency[u2].remove(v2)
    adjacency[v2].remove(u2)
    adjacency[u1].add(u2)
    adjacency[u2].add(u1)
    adjacency[v1].add(v2)
    adjacency[v2].add(v1)
    
    # Check girth
    girth = _compute_girth(adjacency, g - 1)
    
    # Restore
    adjacency[u1].remove(u2)
    adjacency[u2].remove(u1)
    adjacency[v1].remove(v2)
    adjacency[v2].remove(v1)
    adjacency[u1].add(v1)
    adjacency[v1].add(u1)
    adjacency[u2].add(v2)
    adjacency[v2].add(u2)
    
    return girth >= g

def _generate_random_regular(k, n):
    """Generate a random k-regular graph using configuration model + simplification.
    Returns adjacency list or None if failed.
    """
    if k * n % 2 != 0:
        return None
    
    # Configuration model: create 2k stubs, pair them randomly
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    # Pair stubs
    edges_set = set()
    adjacency = [set() for _ in range(n)]
    
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            # Self-loop, need to fix
            return None
        if u in adjacency[v]:
            # Multi-edge, need to fix
            return None
        edges_set.add((min(u, v), max(u, v)))
        adjacency[u].add(v)
        adjacency[v].add(u)
    
    # Check degrees
    for i in range(n):
        if len(adjacency[i]) != k:
            return None
    
    return adjacency

def _improve_girth_by_switching(adjacency, k, g, time_limit=60.0):
    """Try to improve girth using edge switching.
    Returns (adjacency, girth) or None if failed.
    """
    start_time = time.time()
    n = len(adjacency)
    
    # Compute initial girth
    current_girth = _compute_girth(adjacency, g - 1)
    if current_girth >= g:
        return adjacency, current_girth
    
    best_girth = current_girth
    best_adjacency = [set(neigh) for neigh in adjacency]
    
    # Collect all edges
    def get_edges(adj):
        edges = []
        for i in range(len(adj)):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return edges
    
    # Repeatedly try random switches
    for iteration in range(100000):
        if time.time() - start_time > time_limit:
            break
        
        # Pick two random edges
        edges = get_edges(adjacency)
        if len(edges) < 2:
            break
        
        (u1, v1), (u2, v2) = random.sample(edges, 2)
        
        # Check if switch is valid (no self-loops, no multi-edges)
        if u1 == u2 or u1 == v2 or v1 == u2 or v1 == v2:
            continue
        if u2 in adjacency[u1] and v2 in adjacency[v1]:
            # Would create multi-edge
            continue
        if v2 in adjacency[u1] and u2 in adjacency[v1]:
            continue
        
        # Check if switch improves or maintains girth
        if _has_short_cycle_after_switch(adjacency, u1, v1, u2, v2, g):
            # Perform the switch
            adjacency[u1].remove(v1)
            adjacency[v1].remove(u1)
            adjacency[u2].remove(v2)
            adjacency[v2].remove(u2)
            adjacency[u1].add(u2)
            adjacency[u2].add(u1)
            adjacency[v1].add(v2)
            adjacency[v2].add(v1)
            
            # Compute new girth
            new_girth = _compute_girth(adjacency, g - 1)
            if new_girth > best_girth:
                best_girth = new_girth
                best_adjacency = [set(neigh) for neigh in adjacency]
            
            if best_girth >= g:
                return best_adjacency, best_girth
    
    return best_adjacency, best_girth

def _luw_construct(k, g):
    # Copy of the seed logic
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

def construct(k, g):
    # For the specific case k=5, g=9, use edge-switching approach
    if k == 5 and g == 9:
        # Moore bound for k=5, g=9:
        # For odd girth g, Moore bound is k * (k-1)^(g-3)/2 + (k-1)^(g-2)/2
        # For k=5, g=9: 5 * 4^3/2 + 4^4/2 = 5*64/2 + 256/2 = 160 + 128 = 288
        # Actually let me recalculate:
        # Moore bound for odd girth g: 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        # = 1 + 5 * (1 + 4 + 16 + 64) = 1 + 5 * 85 = 1 + 425 = 426
        # So lower bound is around 426.
        
        # Try sizes from near the bound to larger
        target_ns = [430, 500, 600, 800, 1000, 1200]
        
        overall_start = time.time()
        time_budget = 100.0
        
        for n in target_ns:
            remaining = time_budget - (time.time() - overall_start)
            if remaining < 5.0:
                break
            
            # Try multiple random initializations
            for attempt in range(10):
                if time.time() - overall_start > time_budget:
                    break
                
                adjacency = _generate_random_regular(k, n)
                if adjacency is None:
                    continue
                
                # Try to improve girth
                improved_adj, girth = _improve_girth_by_switching(
                    adjacency, k, g, time_limit=min(8.0, remaining - 2.0)
                )
                
                if improved_adj is not None and girth >= g:
                    # Verify and return
                    edges = []
                    for i in range(n):
                        for j in improved_adj[i]:
                            if i < j:
                                edges.append((i, j))
                    
                    # Final verification
                    for i in range(n):
                        if len(improved_adj[i]) != k:
                            break
                    else:
                        return n, edges
        
    # Fallback to LUW
    return _luw_construct(k, g)
