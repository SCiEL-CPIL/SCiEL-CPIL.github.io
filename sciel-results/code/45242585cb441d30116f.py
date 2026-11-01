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

def _compute_girth_bounded(adj, n, max_girth):
    """Compute the girth of the graph, bounded by max_girth. Returns min cycle length or max_girth if none found."""
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= max_girth - 1:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and dist[u] + 1 > dist[v]:
                    cycle_len = dist[u] + 1 + dist[v]
                    if cycle_len < max_girth:
                        return cycle_len
    return max_girth

def _random_k_regular_simple(k, n, max_attempts=50):
    """Generate a random simple k-regular graph using configuration model with retries."""
    for _ in range(max_attempts):
        stubs = []
        for i in range(n):
            stubs.extend([i] * k)
        random.shuffle(stubs)
        
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        
        if valid and all(len(adj[i]) == k for i in range(n)):
            return adj
    return None

def _edge_switch_inplace(adj, n):
    """Perform a random 2-switch in-place. Returns True if successful, False otherwise."""
    # Collect all edges
    edges = []
    for u in range(n):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    
    if len(edges) < 2:
        return False
    
    # Try a few times to find a valid switch
    for _ in range(20):
        (a, b), (c, d) = random.sample(edges, 2)
        if len({a, b, c, d}) < 4:
            continue
        
        # Remove old edges
        adj[a].remove(b)
        adj[b].remove(a)
        adj[c].remove(d)
        adj[d].remove(c)
        
        # Choose new pairing
        if random.random() < 0.5:
            new_pairs = [(a, c), (b, d)]
        else:
            new_pairs = [(a, d), (b, c)]
        
        # Check validity
        valid = True
        for u, v in new_pairs:
            if u == v or v in adj[u]:
                valid = False
                break
        
        if not valid:
            # Revert
            adj[a].add(b)
            adj[b].add(a)
            adj[c].add(d)
            adj[d].add(c)
            continue
        
        # Add new edges
        for u, v in new_pairs:
            adj[u].add(v)
            adj[v].add(u)
        
        return True
    
    return False

def _switching_construct(k, g, n, time_limit=20.0):
    """Construct a k-regular graph of girth >= g using edge switching."""
    start_time = time.time()
    
    # Generate initial random k-regular simple graph
    adj = _random_k_regular_simple(k, n)
    if adj is None:
        return None
    
    # Compute initial girth
    current_girth = _compute_girth_bounded(adj, n, g)
    
    if current_girth >= g:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # Perform switching to improve girth
    no_improve = 0
    max_no_improve = 500
    
    while time.time() - start_time < time_limit:
        if no_improve > max_no_improve:
            break
        
        # Make a copy for the switch
        new_adj = [set(s) for s in adj]
        
        if not _edge_switch_inplace(new_adj, n):
            no_improve += 1
            continue
        
        # Check new girth
        new_girth = _compute_girth_bounded(new_adj, n, g)
        
        if new_girth >= current_girth:
            adj = new_adj
            if new_girth > current_girth:
                current_girth = new_girth
                no_improve = 0
                if current_girth >= g:
                    edges = []
                    for i in range(n):
                        for j in adj[i]:
                            if i < j:
                                edges.append((i, j))
                    return n, edges
            else:
                no_improve += 1
        else:
            no_improve += 1
    
    return None

def construct(k, g):
    if k == 5 and g == 9:
        # Try various n values
        target_ns = [430, 450, 500, 550, 600, 700, 800, 1000, 1200]
        
        total_time_budget = 105.0
        time_per_attempt = total_time_budget / len(target_ns)
        
        for n in target_ns:
            if time.time() > 115:
                break
            res = _switching_construct(k, g, n, time_limit=time_per_attempt)
            if res is not None:
                return res
    
    # Fallback to LUW
    return _luw_construct(k, g)
