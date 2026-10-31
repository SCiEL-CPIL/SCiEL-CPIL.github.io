import time
import random
import sys

def _luw_construct(k, g):
    # Seed program's construction
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

def _generate_random_3_regular(n):
    # Generate a random 3-regular graph
    # Using the configuration model
    half_edges = []
    for i in range(n):
        for _ in range(3):
            half_edges.append(i)
    random.shuffle(half_edges)
    edges = set()
    for i in range(0, len(half_edges), 2):
        u, v = half_edges[i], half_edges[i+1]
        if u == v:
            # Retry or swap to avoid loop
            # Simple retry: find another partner
            for j in range(i+2, len(half_edges), 2):
                if half_edges[j] != u:
                    half_edges[i+1], half_edges[j] = half_edges[j], half_edges[i+1]
                    v = half_edges[i+1]
                    break
            else:
                # Fallback: just pick a random different vertex
                v = (u + 1) % n
                while v == u or (u, v) in edges or (v, u) in edges:
                    v = (v + 1) % n
        if (u, v) in edges or (v, u) in edges:
             # Multi-edge, retry
             for j in range(i+2, len(half_edges), 2):
                if half_edges[j] != u and (u, half_edges[j]) not in edges and (half_edges[j], u) not in edges:
                    half_edges[i+1], half_edges[j] = half_edges[j], half_edges[i+1]
                    v = half_edges[i+1]
                    break
        edges.add((u, v))
    
    # Convert to adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _get_shortest_cycle(adj, n):
    # BFS from each vertex to find shortest cycle
    # This is O(n * (n + m))
    min_cycle = n + 1
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        for u in queue:
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_cycle:
                        min_cycle = cycle_len
    return min_cycle

def _local_search(n, g, time_limit):
    adj = _generate_random_3_regular(n)
    start_time = time.time()
    
    # Initial girth check
    girth = _get_shortest_cycle(adj, n)
    
    while girth < g and time.time() - start_time < time_limit:
        # Find a short cycle to break
        # For efficiency, we can just pick a random edge and try to switch it
        # But targeting the shortest cycle is better
        # To avoid O(n^2) girth check every step, we do it every K steps
        # Here we assume the girth is still < g and we just pick a random edge
        # Actually, we should pick an edge that is part of a short cycle.
        # Approximation: Pick a random edge.
        u = random.randint(0, n-1)
        if not adj[u]: continue
        v = random.choice(adj[u])
        
        # Find a partner edge (x, y)
        # Ensure x != u, y != u, x != v, y != v
        # And (u, x) and (v, y) are not already edges
        x, y = -1, -1
        for _ in range(100): # Try 100 times
            x = random.randint(0, n-1)
            if x == u or x == v: continue
            if not adj[x]: continue
            y = random.choice(adj[x])
            if y == u or y == v: continue
            if y == x: continue
            # Check if (u, x) and (v, y) are valid
            if x in adj[u] or y in adj[v]: continue
            # Check if (u, y) and (v, x) are valid (alternative switch)
            # We'll do (u, x) and (v, y)
            # Ensure no multi-edges
            if x in adj[u] or y in adj[v]: continue
            break
        else:
            continue
            
        # Perform 2-switch: remove (u, v) and (x, y), add (u, x) and (v, y)
        adj[u].remove(v)
        adj[v].remove(u)
        adj[x].remove(y)
        adj[y].remove(x)
        
        adj[u].append(x)
        adj[x].append(u)
        adj[v].append(y)
        adj[y].append(v)
        
        # Check girth periodically
        if random.random() < 0.1: # 10% chance
            girth = _get_shortest_cycle(adj, n)
            if girth >= g:
                # Success
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                return n, edges
        else:
            # Local check: BFS from u, v, x, y up to depth 8
            # If we find a cycle < 16, we revert or continue
            # For simplicity, we just continue, but this might create short cycles
            # A better approach: Revert if local check fails
            pass
            
    # Fallback: return current graph if valid, else None
    girth = _get_shortest_cycle(adj, n)
    if girth >= g:
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    return None

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 100 # 100s limit
    
    # Try local search for various n
    # Known best is 936. L-U-W is 1458.
    # Try 950, 1000, 1100, 1200, 1300, 1400
    n_candidates = [950, 1000, 1100, 1200, 1300, 1400]
    
    for n in n_candidates:
        if time.time() > deadline:
            break
        # Allocate remaining time
        remaining = deadline - time.time()
        if remaining < 5:
            break
        result = _local_search(n, g, remaining - 5)
        if result:
            return result
            
    # Fallback to L-U-W
    return _luw_construct(k, g)
