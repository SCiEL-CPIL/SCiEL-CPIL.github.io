import time
import random
import heapq
from collections import deque

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

def _generate_random_k_regular(n, k, max_attempts=1000):
    """Generate a random simple k-regular graph on n vertices using configuration model + 2-switches."""
    if n * k % 2 != 0:
        return None
    if k >= n:
        return None
    
    for attempt in range(max_attempts):
        # Configuration model: create 2n stubs, n*k/2 edges worth
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        # Pair up stubs to form edges
        adjacency = [set() for _ in range(n)]
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            if v in adjacency[u]:
                valid = False
                break
            adjacency[u].add(v)
            adjacency[v].add(u)
        
        if not valid:
            continue
        
        # Verify it's a simple k-regular graph
        for i in range(n):
            if len(adjacency[i]) != k:
                valid = False
                break
        if not valid:
            continue
        
        return adjacency
    
    return None

def _compute_girth(adjacency, n, max_girth_check=20):
    """Compute the girth of the graph using BFS from each vertex."""
    min_girth = float('inf')
    
    for start in range(n):
        # BFS from start
        visited = [-1] * n
        parent = [-1] * n
        visited[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if visited[v] == -1:
                    visited[v] = visited[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and visited[u] + 1 >= visited[v]:
                    # Found a cycle
                    cycle_len = visited[u] + visited[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
    
    return min_girth if min_girth != float('inf') else max_girth_check + 1

def _has_short_cycle_between(u, v, adjacency, max_dist):
    """Check if there's a path of length < max_dist between u and v (excluding direct edge)."""
    if u == v:
        return True
    if v in adjacency[u]:
        return True  # Direct edge means path of length 1
    
    visited = {u: 0}
    queue = deque([(u, 0)])
    
    while queue:
        node, dist = queue.popleft()
        if dist >= max_dist:
            continue
        for neighbor in adjacency[node]:
            if neighbor == v:
                return True
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))
    
    return False

def _try_switch(adjacency, n, g, max_dist_check):
    """Try a random 2-switch that doesn't create short cycles."""
    # Pick two random edges (u,v) and (w,x)
    edges_list = []
    for u in range(n):
        for v in adjacency[u]:
            if u < v:
                edges_list.append((u, v))
    
    if len(edges_list) < 2:
        return None
    
    # Try several random pairs
    for _ in range(50):
        e1, e2 = random.sample(edges_list, 2)
        u, v = e1
        w, x = e2
        
        # Check no shared vertices
        if len({u, v, w, x}) < 4:
            continue
        
        # Try switch: (u,v) and (w,x) -> (u,w) and (v,x) or (u,x) and (v,w)
        # Check if new edges would create short cycles
        for new_edges in [((u, w), (v, x)), ((u, x), (v, w))]:
            (a, b), (c, d) = new_edges
            
            # Check if (a,b) and (c,d) already exist
            if b in adjacency[a] or d in adjacency[c]:
                continue
            
            # Check girth constraint: no path of length < g-1 between a and b (excluding the new edge)
            # Remove existing edges temporarily for check
            # Actually, we need to check if adding (a,b) creates a cycle < g
            # This means: is there a path from a to b of length < g-1 in the graph without edge (a,b)?
            
            # Simplified: check if there's a path of length < g-1 between a and b
            if _has_short_cycle_between(a, b, adjacency, g - 2):
                continue
            if _has_short_cycle_between(c, d, adjacency, g - 2):
                continue
            
            # Perform the switch
            adjacency[u].remove(v)
            adjacency[v].remove(u)
            adjacency[w].remove(x)
            adjacency[x].remove(w)
            
            adjacency[a].add(b)
            adjacency[b].add(a)
            adjacency[c].add(d)
            adjacency[d].add(c)
            
            return True
    
    return False

def _switching_construct(k, g, n, time_limit=80.0):
    """Construct a k-regular graph of girth >= g using edge switching."""
    start_time = time.time()
    
    # Generate initial random k-regular graph
    adjacency = _generate_random_k_regular(n, k, max_attempts=2000)
    if adjacency is None:
        return None
    
    # Compute initial girth
    current_girth = _compute_girth(adjacency, n)
    
    # If already good, return
    if current_girth >= g:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # Try to improve girth via switching
    max_dist_check = g - 2  # We need paths of length >= g-1, so reject if path < g-1
    
    improvement_count = 0
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        # Try a switch
        if _try_switch(adjacency, n, g, max_dist_check):
            no_improve_count = 0
            # Recompute girth periodically
            if improvement_count % 10 == 0:
                new_girth = _compute_girth(adjacency, n)
                if new_girth >= g:
                    edges = []
                    for i in range(n):
                        for j in adjacency[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
                if new_girth > current_girth:
                    current_girth = new_girth
                    improvement_count += 1
        else:
            no_improve_count += 1
            if no_improve_count > 1000:
                # Restart with new random graph
                adjacency = _generate_random_k_regular(n, k, max_attempts=500)
                if adjacency is None:
                    return None
                current_girth = _compute_girth(adjacency, n)
                if current_girth >= g:
                    edges = []
                    for i in range(n):
                        for j in adjacency[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
                no_improve_count = 0
    
    # Final check
    final_girth = _compute_girth(adjacency, n)
    if final_girth >= g:
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    return None

def construct(k, g):
    # For the specific case k=5, g=9, try switching approach
    if k == 5 and g == 9:
        # Try several sizes near the known optimum
        target_ns = [1116, 1120, 1150, 1200, 1300, 1500]
        
        # Time budget: ~90s for switching, 10s for fallback
        time_per_attempt = 15.0
        
        for n in target_ns:
            if time.time() > 100:
                break
            res = _switching_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
        
    # Fallback to LUW
    return _luw_construct(k, g)
