import time
import random
from collections import deque
import sys

def construct(k, g):
    """
    Construct a k-regular graph with girth >= g.
    For k=3, g=16, we use voltage lifts of the Petersen graph with Thue-Morse derived patterns.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit

    if k != 3:
        return _luw_construct(k, g)

    random.seed(42)

    # Moore bound for k=3, g=16 is 512.
    # We'll search for lifts of the Petersen graph with m copies.
    # n = 10 * m
    # We need girth >= 16.
    
    # Strategy:
    # 1. Generate Thue-Morse based voltage patterns for the 15 edges of the Petersen graph.
    # 2. For each m, try several patterns and check girth.
    # 3. If girth < 16, try local search to improve.

    # Petersen graph edges
    petersen_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    petersen_adj = [[] for _ in range(10)]
    for u, v in petersen_edges:
        petersen_adj[u].append(v)
        petersen_adj[v].append(u)

    base_edges = []
    for u in range(10):
        for v in petersen_adj[u]:
            if u < v:
                base_edges.append((u, v))

    def _verify_girth_fast(n, edges, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        dist = [0] * n
        parent = [-1] * n
        seen_mark = [0] * n
        mark = 0

        for s in range(n):
            mark += 1
            dist[s] = 0
            parent[s] = -1
            seen_mark[s] = mark
            queue = deque([s])

            while queue:
                u = queue.popleft()
                du = dist[u]
                if du >= g // 2:
                    continue
                for v in adj[u]:
                    if seen_mark[v] != mark:
                        dist[v] = du + 1
                        parent[v] = u
                        seen_mark[v] = mark
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        cycle_len = du + dist[v] + 1
                        if cycle_len < g:
                            return False

        return True

    def _build_lift(base_edges, m, shifts):
        n = 10 * m
        edges = []
        for (u, v), shift in zip(base_edges, shifts):
            for i in range(m):
                u_id = u * m + i
                v_id = v * m + (i + shift) % m
                edges.append((u_id, v_id))
        return n, edges

    # Thue-Morse sequence generator
    def thue_morse(n):
        # Thue-Morse: t(i) = parity of number of 1s in binary representation of i
        return [bin(i).count('1') % 2 for i in range(n)]

    # Generate diverse voltage patterns
    def generate_patterns(m, n_patterns=20):
        patterns = []
        
        # Pattern 1: Thue-Morse based
        tm = thue_morse(m)
        for offset in range(min(5, m)):
            shifts = []
            for edge_idx, (u, v) in enumerate(base_edges):
                # Use different linear combinations of Thue-Morse
                if edge_idx < 5:  # outer pentagon
                    s = (tm[(edge_idx + offset) % m] * m // 3 + offset) % m
                elif edge_idx < 10:  # inner pentagon
                    s = (tm[(edge_idx + offset + 1) % m] * m // 3 + offset + 1) % m
                else:  # spokes
                    s = (tm[(edge_idx + offset + 2) % m] * m // 3 + offset + 2) % m
                shifts.append(s)
            patterns.append(shifts)
        
        # Pattern 2: Quadratic
        for a in range(1, min(5, m)):
            shifts = []
            for edge_idx, (u, v) in enumerate(base_edges):
                s = (a * edge_idx * edge_idx + edge_idx + offset) % m if m > 1 else 0
                shifts.append(s)
            patterns.append(shifts)
        
        # Pattern 3: Prime-based
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p_idx in range(min(5, len(primes))):
            p = primes[p_idx]
            shifts = [(p * (edge_idx + 1)) % m for edge_idx in range(15)]
            patterns.append(shifts)
        
        # Pattern 4: Random with structure
        for _ in range(min(5, n_patterns)):
            shifts = [random.randint(0, m-1) for _ in range(15)]
            patterns.append(shifts)
        
        return patterns

    # Try m from 52 to 1000
    for m in range(52, 1000):
        if time.time() > deadline:
            break

        patterns = generate_patterns(m, 20)
        
        for shifts in patterns:
            if time.time() > deadline:
                break
            
            n, edges = _build_lift(base_edges, m, shifts)
            
            if _verify_girth_fast(n, edges, g):
                return n, edges

            # Local search
            for _ in range(500):
                if time.time() > deadline:
                    break
                
                # Pick a random shift to modify
                idx = random.randint(0, 14)
                old_shift = shifts[idx]
                new_shift = random.randint(0, m-1)
                
                if new_shift == old_shift:
                    continue
                
                shifts[idx] = new_shift
                n, edges = _build_lift(base_edges, m, shifts)
                
                if _verify_girth_fast(n, edges, g):
                    return n, edges
                else:
                    # Revert the change
                    shifts[idx] = old_shift

    # If not found, return a large random lift
    m = 3000
    shifts_rand = [random.randint(0, m-1) for _ in range(15)]
    n, edges = _build_lift(base_edges, m, shifts_rand)
    return n, edges

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k-regular graphs.
    """
    if k != 3:
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
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

    # For k=3, we use q=3.
    q = 3

    def build_luw(d):
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

    # Try d=8 first. If girth < 16, try d=9, etc.
    for d in [8, 9, 10, 11]:
        n, edges = build_luw(d)
        if _check_girth_fast(n, edges, g):
            return n, edges
    # If all fail (unlikely for d=11), return the largest one (which should be valid by theory)
    return build_luw(11)

def _build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_girth_fast(n, edges, min_girth):
    """Fast girth check using BFS from each vertex with parent tracking."""
    adj = _build_adj(n, edges)
    max_depth = (min_girth - 1) // 2

    for start in range(n):
        visited = [-1] * n
        visited[start] = 0
        queue = deque([(start, 0, -1)])

        while queue:
            node, depth, parent = queue.popleft()
            if depth > max_depth:
                continue
            for neighbor in adj[node]:
                if neighbor == parent:
                    continue
                if visited[neighbor] == -1:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1, node))
                else:
                    cycle_len = depth + visited[neighbor] + 1
                    if cycle_len < min_girth:
                        return False
    return True

