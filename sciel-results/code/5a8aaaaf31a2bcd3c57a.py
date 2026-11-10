import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    """Construct a k-regular graph of girth at least g with as few vertices as possible."""
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Moore bound for k=3, even g
    # n >= 2 + k * sum_{i=0}^{g/2-2} (k-1)^i
    half_g = g // 2
    moore_bound = 2 + k * ((k-1)**(half_g - 1) - 1) // (k - 2) if k > 2 else 2
    moore_bound = max(moore_bound, 2)
    
    # Start with a reasonable n
    # For (3,16), best known is 936, lower bound ~512
    # Try starting near the lower bound and increase if needed
    base_n = max(moore_bound, 2 * (k - 1) ** (half_g - 1))
    
    def generate_random_3reg(n):
        """Generate a random 3-regular graph on n vertices using configuration model."""
        if n % 2 != 0 or n < 4:
            return None
        # Create half-edges: each vertex has 3 stubs
        stubs = []
        for v in range(n):
            stubs.extend([v] * 3)
        random.shuffle(stubs)
        # Pair up stubs
        edges = set()
        adj = [set() for _ in range(n)]
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            if u == v:
                continue  # Skip loops, try to fix later
            if v in adj[u]:
                continue  # Skip multi-edges
            adj[u].add(v)
            adj[v].add(u)
            edges.add((min(u,v), max(u,v)))
        # Check if all vertices have degree 3
        for v in range(n):
            if len(adj[v]) != 3:
                return None
        return edges
    
    def find_shortest_cycle(adj, n):
        """Find shortest cycle using BFS from each vertex (non-backtracking)."""
        min_cycle = n + 1
        for start in range(n):
            # BFS tracking parent and depth
            visited = [-1] * n  # parent
            depth = [-1] * n
            visited[start] = -2  # Mark start
            depth[start] = 0
            queue = deque([(start, -1)])  # (node, parent)
            while queue:
                node, parent = queue.popleft()
                if depth[node] >= min_cycle:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if depth[nb] == -1:
                        depth[nb] = depth[node] + 1
                        visited[nb] = node
                        queue.append((nb, node))
                    elif visited[nb] != node:  # Found a cycle
                        cycle_len = depth[node] + depth[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
        return min_cycle
    
    def try_edge_swap(adj, n, edges, cycle_nodes):
        """Try to break a cycle by swapping edges."""
        if not cycle_nodes or len(cycle_nodes) < 3:
            return False
        # Try all pairs of edges on the cycle to swap
        cycle_edges = []
        for i in range(len(cycle_nodes)):
            u = cycle_nodes[i]
            v = cycle_nodes[(i+1) % len(cycle_nodes)]
            cycle_edges.append((u, v))
        
        for i in range(len(cycle_edges)):
            u1, v1 = cycle_edges[i]
            for j in range(i+1, len(cycle_edges)):
                u2, v2 = cycle_edges[j]
                # Remove edges (u1,v1) and (u2,v2), add (u1,u2) and (v1,v2) or (u1,v2) and (v1,u2)
                for new_e1, new_e2 in [((u1,u2),(v1,v2)), ((u1,v2),(v1,u2))]:
                    a1, b1 = new_e1
                    a2, b2 = new_e2
                    if a1 == b1 or a2 == b2:
                        continue
                    if b1 in adj[a1] and b2 in adj[a2]:  # Already have these edges
                        continue
                    if a2 in adj[b1] or a1 in adj[b2]:  # Would create multi-edge
                        continue
                    # Check degrees won't be violated
                    if len(adj[a1]) >= 3 and len(adj[b1]) >= 3:
                        continue
                    # Remove old edges
                    adj[u1].discard(v1)
                    adj[v1].discard(u1)
                    adj[u2].discard(v2)
                    adj[v2].discard(u2)
                    edges.discard((min(u1,v1), max(u1,v1)))
                    edges.discard((min(u2,v2), max(u2,v2)))
                    # Add new edges
                    adj[a1].add(b1)
                    adj[b1].add(a1)
                    adj[a2].add(b2)
                    adj[b2].add(a2)
                    edges.add((min(a1,b1), max(a1,b1)))
                    edges.add((min(a2,b2), max(a2,b2)))
                    return True
        return False
    
    # Try increasing n until we find a valid graph
    for n in range(max(4, base_n), max(4, base_n) + 200):
        if time.time() > deadline:
            break
        if n % 2 == 0 and n >= 4:
            # Try multiple random graphs
            for attempt in range(50):
                if time.time() > deadline:
                    break
                edges = generate_random_3reg(n)
                if edges is None:
                    continue
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                
                # Verify 3-regular
                valid = all(len(adj[v]) == 3 for v in range(n))
                if not valid:
                    continue
                
                # Check girth
                girth = find_shortest_cycle(adj, n)
                if girth >= g:
                    return n, list(edges)
                
                # Try to improve girth via local search
                for step in range(100):
                    if time.time() > deadline:
                        break
                    girth = find_shortest_cycle(adj, n)
                    if girth >= g:
                        return n, list(edges)
                    if girth == n + 1:  # No cycle found (shouldn't happen)
                        break
                    # Find a short cycle and try to break it
                    # Simplified: just try random edge swaps
                    u = random.choice(range(n))
                    v = random.choice(list(adj[u]))
                    w = random.choice([x for x in adj[v] if x != u]) if len(adj[v]) > 1 else None
                    if w is None:
                        continue
                    x = random.choice([y for y in adj[w] if y != v]) if len(adj[w]) > 1 else None
                    if x is None or x == u:
                        continue
                    # Swap: remove (u,v) and (w,x), add (u,w) and (v,x)
                    if w in adj[u] or x in adj[v]:
                        continue
                    adj[u].discard(v)
                    adj[v].discard(u)
                    adj[w].discard(x)
                    adj[x].discard(w)
                    edges.discard((min(u,v), max(u,v)))
                    edges.discard((min(w,x), max(w,x)))
                    adj[u].add(w)
                    adj[w].add(u)
                    adj[v].add(x)
                    adj[x].add(v)
                    edges.add((min(u,w), max(u,w)))
                    edges.add((min(v,x), max(v,x)))
    
    # Fallback: use LNW construction
    return _lnw_construct(k, g)

def _lnw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
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
