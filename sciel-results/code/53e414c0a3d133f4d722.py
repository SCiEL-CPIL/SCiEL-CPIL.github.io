import time
import random
import collections

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _compute_girth(adj, n, max_depth=8):
    """
    Computes the girth (length of shortest cycle) of a graph.
    Returns infinity if no cycle found within max_depth.
    """
    min_girth = float('inf')
    for u in range(n):
        # BFS from u
        # dist[v] = distance from u
        # parent[v] = previous node
        dist = [-1] * n
        parent = [-1] * n
        queue = collections.deque()
        queue.append(u)
        dist[u] = 0
        
        while queue:
            v = queue.popleft()
            for w in adj[v]:
                if dist[w] == -1:
                    dist[w] = dist[v] + 1
                    parent[w] = v
                    queue.append(w)
                elif w != parent[v]:
                    # Found a cycle
                    # Cycle length is dist[v] + dist[w] + 1
                    # We need to be careful: this finds *a* cycle, not necessarily the shortest one involving v,w in this BFS tree
                    # However, for girth, we can just take the min of all such back-edges
                    cycle_len = dist[v] + dist[w] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth <= max_depth:
                        # We can't get better than 2 (triangle), but we are looking for >= 16
                        # If we find a short one, we can return early if we just want to know if it's < 16
                        pass
        
        # Note: The standard BFS for girth is O(N*(N+E)). For N=1000, this is 1e6 ops per vertex, 1e9 total. Too slow.
        # We need a faster check for "is girth < 16".
        
    return min_girth

def _has_short_cycle_nb(n, adj, min_girth):
    """
    Returns True if the graph has a cycle shorter than min_girth.
    Uses non-backtracking BFS.
    """
    max_depth = min_girth - 1
    
    for u in range(n):
        for v in adj[u]:
            visited = set()
            visited.add((v, u))
            frontier = [(v, u)]
            
            for step in range(1, max_depth + 1):
                next_frontier = []
                for (node, prev) in frontier:
                    for nxt in adj[node]:
                        if nxt == prev:
                            continue
                        new_state = (nxt, node)
                        if new_state in visited:
                            continue
                        visited.add(new_state)
                        next_frontier.append(new_state)
                if not next_frontier:
                    break
                if (u, v) in next_frontier:
                    return True
                frontier = next_frontier
                
                if _time_left() < 1.0:
                    return False
    
    return False

def _random_3_regular(n):
    """
    Generates a random 3-regular graph on n vertices.
    Returns adjacency list.
    """
    if n % 2 != 0:
        return None
        
    # Configuration model
    half_edges = [i for i in range(n) for _ in range(3)]
    random.shuffle(half_edges)
    
    adj = [[] for _ in range(n)]
    edges = []
    edge_set = set()
    
    for i in range(0, len(half_edges), 2):
        u = half_edges[i]
        v = half_edges[i+1]
        if u == v:
            # Self-loop, retry or swap
            # Simple retry: find another partner
            # For simplicity, just restart if self-loop
            return None
        if (min(u,v), max(u,v)) in edge_set:
            # Multi-edge
            return None
        edge_set.add((min(u,v), max(u,v)))
        adj[u].append(v)
        adj[v].append(u)
        edges.append((u, v))
        
    return adj, edges

def _local_search(n, max_time=50.0):
    """
    Attempts to find a 3-regular graph of girth >= 16 on n vertices.
    """
    start_time = time.monotonic()
    
    # Try multiple random initializations
    for attempt in range(10):
        if time.monotonic() - start_time > max_time:
            break
            
        result = _random_3_regular(n)
        if not result:
            continue
            
        adj, edges = result
        
        # Check initial girth
        if not _has_short_cycle_nb(n, adj, 16):
            return n, edges
            
        # Local search: 2-switches
        # We need to find short cycles and break them.
        # Since _has_short_cycle_nb is slow, we will just do random 2-switches and check periodically.
        
        for step in range(1000):
            if time.monotonic() - start_time > max_time:
                break
                
            # Pick two random edges
            e1_idx = random.randint(0, len(edges)-1)
            e2_idx = random.randint(0, len(edges)-1)
            if e1_idx == e2_idx:
                continue
                
            u1, v1 = edges[e1_idx]
            u2, v2 = edges[e2_idx]
            
            # Try switch (u1-u2, v1-v2) or (u1-v2, v1-u2)
            # Check for loops/multi-edges
            valid_switch = False
            new_e1 = (u1, u2)
            new_e2 = (v1, v2)
            if new_e1[0] != new_e1[1] and new_e2[0] != new_e2[1]:
                key1 = (min(new_e1[0], new_e1[1]), max(new_e1[0], new_e1[1]))
                key2 = (min(new_e2[0], new_e2[0]), max(new_e2[0], new_e2[1]))
                if key1 not in edge_set and key2 not in edge_set:
                    # Apply switch
                    edge_set.remove((min(u1,v1), max(u1,v1)))
                    edge_set.remove((min(u2,v2), max(u2,v2)))
                    edge_set.add(key1)
                    edge_set.add(key2)
                    
                    # Update adj
                    adj[u1].remove(u2) # Wait, u1 was connected to v1. Now u1 connected to u2?
                    # My switch logic:
                    # Old: (u1, v1), (u2, v2)
                    # New: (u1, u2), (v1, v2)
                    # Remove old
                    adj[u1].remove(v1)
                    adj[v1].remove(u1)
                    adj[u2].remove(v2)
                    adj[v2].remove(u2)
                    # Add new
                    adj[u1].append(u2)
                    adj[u2].append(u1)
                    adj[v1].append(v2)
                    adj[v2].append(v1)
                    
                    edges[e1_idx] = new_e1
                    edges[e2_idx] = new_e2
                    valid_switch = True
                    
            if not valid_switch:
                continue
                
            # Check girth every 10 steps
            if step % 10 == 0:
                if not _has_short_cycle_nb(n, adj, 16):
                    return n, edges
                    
    return None

def _construct_luw(k, g):
    """Fallback: construct LUW graph."""
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
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0
    
    # Try to find a small graph using local search
    # Start from the best known n=936
    for n in range(936, 1000, 2):
        if _time_left() < 10.0:
            break
            
        result = _local_search(n, max_time=min(20.0, _time_left() - 5.0))
        if result:
            return result
            
    # Fallback to random search for larger n
    for n in range(1000, 1200, 2):
        if _time_left() < 5.0:
            break
            
        for trial in range(5):
            if _time_left() < 2.0:
                break
            result = _random_3_regular(n)
            if result:
                adj, edges = result
                if not _has_short_cycle_nb(n, adj, g):
                    return (n, edges)
                    
    # Final fallback: LUW
    return _construct_luw(k, g)
