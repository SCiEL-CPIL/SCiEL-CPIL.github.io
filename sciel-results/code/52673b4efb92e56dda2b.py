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


def _compute_girth(adjacency, n, max_check=12):
    """Compute girth using BFS from each vertex, looking for shortest cycles."""
    min_cycle = max_check + 1
    for start in range(n):
        # BFS from start, tracking parent to detect cycles
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_check:
                continue
            for v in adjacency[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    # Found a cycle: dist[u] + dist[v] + 1
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
    return min_cycle if min_cycle <= max_check else None


def _try_2switch(adjacency, n, g, max_iter=50000):
    """
    Try to increase girth using 2-switches.
    A 2-switch: remove edges (a,b) and (c,d), add edges (a,c) and (b,d)
    (or (a,d) and (b,c)).
    Accept if new girth >= old_girth and ideally improves it.
    """
    start_time = time.time()
    current_girth = _compute_girth(adjacency, n, max_check=g + 2)
    if current_girth is None:
        current_girth = g + 10  # Assume large enough
    
    if current_girth >= g:
        return True, current_girth
    
    # Build list of all edges
    edge_list = []
    for u in range(n):
        for v in adjacency[u]:
            if u < v:
                edge_list.append((u, v))
    
    random.shuffle(edge_list)
    
    accepted = 0
    for _ in range(max_iter):
        if time.time() - start_time > 30.0:
            break
        
        # Pick two random edges
        e1_idx = random.randint(0, len(edge_list) - 1)
        e2_idx = random.randint(0, len(edge_list) - 1)
        if e1_idx == e2_idx:
            continue
        
        a, b = edge_list[e1_idx]
        c, d = edge_list[e2_idx]
        
        # Check they are distinct and don't share vertices
        if len({a, b, c, d}) < 4:
            continue
        
        # Check new edges don't already exist
        if c in adjacency[a] or d in adjacency[b]:
            continue
        
        # Try switch: remove (a,b),(c,d), add (a,c),(b,d)
        adjacency[a].discard(b)
        adjacency[b].discard(a)
        adjacency[c].discard(d)
        adjacency[d].discard(c)
        adjacency[a].add(c)
        adjacency[c].add(a)
        adjacency[b].add(d)
        adjacency[d].add(b)
        
        # Check if this creates a short cycle
        # Quick check: verify girth locally
        new_girth = _compute_girth(adjacency, n, max_check=g - 1)
        
        if new_girth is None or new_girth >= g:
            # Accept the switch
            accepted += 1
            # Update current girth occasionally
            if accepted % 100 == 0:
                current_girth = _compute_girth(adjacency, n, max_check=g + 2)
                if current_girth is not None and current_girth >= g:
                    # Success!
                    return True, current_girth
        else:
            # Revert
            adjacency[a].discard(c)
            adjacency[c].discard(a)
            adjacency[b].discard(d)
            adjacency[d].discard(b)
            adjacency[a].add(b)
            adjacency[b].add(a)
            adjacency[c].add(d)
            adjacency[d].add(c)
    
    # Final check
    final_girth = _compute_girth(adjacency, n, max_check=g + 2)
    if final_girth is None or final_girth >= g:
        return True, final_girth
    return False, final_girth


def _random_regular_graph(k, n):
    """Generate a random k-regular graph on n vertices using configuration model + 2-switches."""
    # Configuration model: create 2*n stubs, n per vertex, k stubs each
    stubs = []
    for i in range(n):
        for _ in range(k):
            stubs.append(i)
    
    random.shuffle(stubs)
    
    # Pair up stubs to form edges
    adjacency = [set() for _ in range(n)]
    for i in range(0, len(stubs), 2):
        u, v = stubs[i], stubs[i + 1]
        if u == v:
            # Self-loop, need to fix
            continue
        if v in adjacency[u]:
            # Parallel edge, need to fix
            continue
        adjacency[u].add(v)
        adjacency[v].add(u)
    
    # Fix issues with 2-switches until we have a simple k-regular graph
    max_attempts = 100000
    for attempt in range(max_attempts):
        # Check if all degrees are correct
        valid = True
        for i in range(n):
            if len(adjacency[i]) != k:
                valid = False
                break
        if valid:
            # Check for parallel edges (shouldn't happen with sets)
            return adjacency
        
        # Find vertices with wrong degree
        over = [i for i in range(n) if len(adjacency[i]) > k]
        under = [i for i in range(n) if len(adjacency[i]) < k]
        
        if not over or not under:
            break
        
        # Try to fix by finding edges to remove/add
        # This is complex; simpler to restart
        if attempt % 1000 == 0:
            # Restart configuration model
            stubs = []
            for i in range(n):
                for _ in range(k):
                    stubs.append(i)
            random.shuffle(stubs)
            adjacency = [set() for _ in range(n)]
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    continue
                if v in adjacency[u]:
                    continue
                adjacency[u].add(v)
                adjacency[v].add(u)
    
    return None


def _greedy_construct(k, g, n, time_limit=10.0):
    # Try to construct a k-regular graph of girth >= g on n vertices
    start_time = time.time()
    adjacency = [set() for _ in range(n)]
    degrees = [0] * n
    
    # Randomize the order of vertices to break symmetry
    vertices = list(range(n))
    random.shuffle(vertices)
    
    def has_short_cycle(u, v, max_depth):
        if u == v: return True
        visited = {u: 0}
        queue = deque([(u, 0)])
        while queue:
            node, dist = queue.popleft()
            if dist >= max_depth:
                continue
            for neighbor in adjacency[node]:
                if neighbor == v:
                    return True
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))
        return False

    active = list(range(n))
    random.shuffle(active)
    
    while active:
        if time.time() - start_time > time_limit:
            return None
            
        u_idx = random.randint(0, len(active) - 1)
        u = active[u_idx]
        
        found = False
        active_copy = active[:]
        random.shuffle(active_copy)
        
        for v in active_copy:
            if v == u:
                continue
            if v in adjacency[u]:
                continue
            
            if has_short_cycle(u, v, g - 2):
                continue
            
            adjacency[u].add(v)
            adjacency[v].add(u)
            degrees[u] += 1
            degrees[v] += 1
            found = True
            break
            
        if not found:
            return None
            
        new_active = []
        for node in active:
            if degrees[node] < k:
                new_active.append(node)
        active = new_active
        
    for i in range(n):
        if len(adjacency[i]) != k:
            return None
            
    edges = []
    for i in range(n):
        for j in adjacency[i]:
            if i < j:
                edges.append((i, j))
                
    return n, edges


def _switching_construct(k, g, n, time_limit=80.0):
    """
    Probabilistic edge-switching approach:
    1. Generate random k-regular graph on n vertices
    2. Use 2-switches to increase girth
    """
    start_time = time.time()
    
    # Generate initial random k-regular graph
    adjacency = None
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            return None
        adjacency = _random_regular_graph(k, n)
        if adjacency is not None:
            break
    
    if adjacency is None:
        return None
    
    # Verify it's actually k-regular and simple
    for i in range(n):
        if len(adjacency[i]) != k:
            return None
    
    # Check initial girth
    initial_girth = _compute_girth(adjacency, n, max_check=g + 2)
    if initial_girth is not None and initial_girth >= g:
        # Already good!
        edges = []
        for i in range(n):
            for j in adjacency[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # Use 2-switches to improve girth
    success, final_girth = _try_2switch(adjacency, n, g, max_iter=200000)
    
    if success:
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
        # Try a few sizes, starting from best known
        target_ns = [1116, 1120, 1150, 1200, 1300, 1500]
        
        total_start = time.time()
        time_per_attempt = 25.0
        
        for n in target_ns:
            if time.time() - total_start > 100:
                break
            res = _switching_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
        
        # Fallback to greedy
        for n in target_ns:
            if time.time() - total_start > 110:
                break
            res = _greedy_construct(k, g, n, time_limit=5.0)
            if res is not None:
                return res
        
    # Fallback to LUW
    return _luw_construct(k, g)
