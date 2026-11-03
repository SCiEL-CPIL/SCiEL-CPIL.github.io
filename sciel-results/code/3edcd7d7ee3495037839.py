import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    # Specific handling for k=3, g=16 using Voltage Lift on Petersen Graph
    if k == 3 and g == 16:
        result = _voltage_lift_petersen(94, 16, deadline)
        if result:
            return result
        # Fallback to LUW if lift fails
        result = _luw_fallback(k, g, deadline)
        if result:
            return result

    # Generic LUW for other cases
    return _luw_fallback(k, g, deadline)

def _voltage_lift_petersen(m, target_girth, deadline):
    # Petersen Graph
    # 10 vertices.
    # Edges:
    # 0-1, 1-2, 2-3, 3-4, 4-0 (Outer pentagon)
    # 5-6, 6-7, 7-8, 8-9, 9-5 (Inner star)
    # 0-5, 1-6, 2-7, 3-8, 4-9 (Spokes)
    base_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    adj = [[] for _ in range(10)]
    for i, (u, v) in enumerate(base_edges):
        adj[u].append((v, i))
        adj[v].append((u, i))

    # Initialize weights
    weights = [random.randint(0, m - 1) for _ in range(len(base_edges))]
    
    # BFS to check if valid
    def is_valid(weights):
        # State: (node, prev_node, sum_mod_m)
        # Start from all nodes
        # We want to find a non-backtracking closed walk with sum 0
        # Visited: set of (node, prev, sum)
        # To be fast, we can just BFS up to depth target_girth - 1
        # If we reach (start, start, 0) at depth > 0, we found a cycle.
        
        for start_node in range(10):
            # Queue: (node, prev, sum, depth)
            q = deque()
            q.append((start_node, -1, 0, 0))
            visited = set()
            visited.add((start_node, -1, 0))
            
            while q:
                node, prev, s, depth = q.popleft()
                if depth >= target_girth:
                    continue
                
                for neighbor, edge_idx in adj[node]:
                    if neighbor == prev:
                        continue
                    w = weights[edge_idx]
                    new_s = (s + w) % m
                    if neighbor == start_node and depth > 0:
                        # Check if this is a valid cycle
                        # We just reached the start node.
                        # Is it a simple cycle in the lift?
                        # Yes, if the walk in base is non-backtracking and simple?
                        # Actually, a closed walk in base maps to a cycle in lift
                        # if the voltages sum to 0.
                        # We need to ensure the lift cycle is simple.
                        # If the base walk is non-backtracking, the lift cycle
                        # might still self-intersect.
                        # However, for girth calculation, we just need the
                        # shortest closed walk with sum 0.
                        # If the shortest such walk has length L, the girth
                        # is at most L.
                        # We want NO such walk with length < target_girth.
                        if new_s == 0:
                            return False
                    # Only expand if not visited
                    # Note: We must visit (neighbor, node, new_s)
                    key = (neighbor, node, new_s)
                    if key not in visited:
                        visited.add(key)
                        q.append((neighbor, node, new_s, depth + 1))
        return True

    attempts = 0
    while time.time() < deadline - 5:
        attempts += 1
        if is_valid(weights):
            # Construct the graph
            n = 10 * m
            edges = []
            for i, (u, v) in enumerate(base_edges):
                w = weights[i]
                for j in range(m):
                    u_l = u * m + j
                    v_l = v * m + (j + w) % m
                    edges.append((u_l, v_l))
            return n, edges
        
        # Perturb weights
        # Pick a random edge and change its weight
        # Or, if we found a bad cycle, change weights on that cycle
        # For simplicity, random perturbation
        if attempts % 10 == 0:
            # Reshuffle completely
            weights = [random.randint(0, m - 1) for _ in range(len(base_edges))]
        else:
            idx = random.randint(0, len(weights) - 1)
            weights[idx] = (weights[idx] + random.randint(1, m - 1)) % m

    return None

def _luw_fallback(k, g, deadline):
    # Original LUW implementation
    def _luw_impl(k, d):
        q = k
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add = lambda x, y: x ^ y
            sub = lambda x, y: x ^ y
            mul = lambda x, y: table[x][y]
        else:
            add = lambda x, y: (x + y) % q
            sub = lambda x, y: (x - y) % q
            mul = lambda x, y: x * y % q
            
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    d_min = (g - 2) // 2 + 1
    for d in range(d_min, d_min + 3):
        if time.time() > deadline - 2:
            break
        try:
            n, edges = _luw_impl(k, d)
            return n, edges
        except Exception:
            pass
    return None
