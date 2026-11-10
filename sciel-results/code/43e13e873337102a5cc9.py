import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # 1. Fallback: LUW Graph
    def get_luw_graph():
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
            if time.time() > deadline:
                break
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    luw_n, luw_edges = get_luw_graph()
    best_n = luw_n
    best_edges = luw_edges

    # 2. GQ(4,2)-based construction with cycle extension
    # The incidence graph of GQ(4,2) has 25 points + 25 lines = 50 vertices,
    # each point on 3 lines, each line has 3 points -> 6-regular? No.
    # Actually GQ(4,2): parameters (s,t)=(4,2), so each line has s+1=5 points,
    # each point is on t+1=3 lines. Total points = (s+1)(st+1) = 5*9 = 45
    # Total lines = 45. Each point on 3 lines, each line has 5 points.
    # Incidence graph: 45+45=90 vertices, each vertex has degree 3 or 5.
    # Not regular. Let me use a different approach.
    
    # Actually, for k=5, g=9, let me try to construct a 5-regular graph
    # by starting with a known structure and applying edge switches.
    
    # Strategy: Start with LUW graph, find 8-cycles (if g=9), and try to
    # extend them to 9-cycles via edge switching.
    
    if k != 5 or g != 9:
        return best_n, best_edges
    
    # Build adjacency from LUW
    n = luw_n
    adj = [set() for _ in range(n)]
    for u, v in luw_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Verify it's 5-regular
    for i in range(n):
        if len(adj[i]) != k:
            return best_n, best_edges
    
    # Find all 8-cycles
    def find_cycles_of_length(length):
        cycles = []
        for start in range(n):
            # BFS/DFS to find cycles of exact length
            # Use non-backtracking walks
            visited = [None] * n
            stack = [(start, [start])]
            while stack:
                node, path = stack.pop()
                if len(path) == length:
                    if node == start:
                        # Found a cycle, normalize it
                        cycle = tuple(path)
                        # Check if it's a simple cycle (no repeated vertices except start=end)
                        if len(set(cycle[:-1])) == length - 1:
                            # Normalize: rotate so smallest vertex is first
                            min_idx = min(range(length), key=lambda i: cycle[i])
                            normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
                            # Also check reverse
                            rev = tuple(normalized[:1] + normalized[:0:-1])
                            if normalized < rev:
                                if normalized not in cycles:
                                    cycles.append(normalized)
                    continue
                if len(path) > length:
                    continue
                for nb in adj[node]:
                    if nb == (path[-2] if len(path) >= 2 else -1):
                        continue  # non-backtracking
                    if nb in path:
                        if nb == start and len(path) == length:
                            continue  # handled above
                        else:
                            continue  # would create shorter cycle
                    stack.append((nb, path + [nb]))
        return cycles
    
    # This is too slow for large n. Let me use a simpler heuristic.
    # Instead, let me try to find short cycles using BFS from each vertex.
    
    def find_shortest_cycle_from(start, max_len):
        # BFS to find shortest cycle through start
        dist = {start: 0}
        parent = {start: -1}
        queue = deque([(start, 0)])
        while queue:
            node, d = queue.popleft()
            if d >= max_len:
                continue
            for nb in adj[node]:
                if nb == start and d + 1 >= 3:
                    # Found a cycle
                    # Reconstruct
                    cycle = [nb]
                    curr = node
                    while curr != -1:
                        cycle.append(curr)
                        curr = parent[curr]
                    cycle.reverse()
                    return cycle
                if nb not in dist:
                    dist[nb] = d + 1
                    parent[nb] = node
                    queue.append((nb, d + 1))
        return None
    
    # Try to find and break 8-cycles
    def try_break_cycle(cycle):
        # Pick an edge in the cycle to remove
        # Find an alternative edge to add that maintains regularity
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            # Try to find x, y such that we can swap (u,v) and (x,y) with (u,x) and (v,y)
            for x in adj[u]:
                if x == v or x in cycle:
                    continue
                for y in adj[v]:
                    if y == u or y in cycle or y == x:
                        continue
                    # Check if (x,y) is an edge
                    if y in adj[x]:
                        # Check that (u,x) and (v,y) are not already edges
                        if x not in adj[u] and y not in adj[v]:
                            # Perform swap
                            adj[u].remove(v)
                            adj[v].remove(u)
                            adj[x].remove(y)
                            adj[y].remove(x)
                            adj[u].add(x)
                            adj[x].add(u)
                            adj[v].add(y)
                            adj[y].add(v)
                            return True
        return False
    
    # Iteratively find and break 8-cycles
    iterations = 0
    max_iterations = 5000
    while time.time() < deadline - 5 and iterations < max_iterations:
        # Find an 8-cycle
        found = False
        for start in range(n):
            if time.time() > deadline - 5:
                break
            cycle = find_shortest_cycle_from(start, 8)
            if cycle and len(cycle) <= 8:
                if try_break_cycle(cycle):
                    found = True
                    break
        if not found:
            break
        iterations += 1
    
    # Check if we improved
    # Verify girth >= 9
    def check_girth(min_girth):
        # Check that no cycle shorter than min_girth exists
        for start in range(n):
            dist = {start: 0}
            parent = {start: -1}
            queue = deque([(start, 0)])
            while queue:
                node, d = queue.popleft()
                if d >= min_girth - 1:
                    continue
                for nb in adj[node]:
                    if nb == start and d + 1 >= 3:
                        return False  # Found cycle shorter than min_girth
                    if nb not in dist:
                        dist[nb] = d + 1
                        parent[nb] = node
                        queue.append((nb, d + 1))
        return True
    
    if check_girth(g):
        # Extract edges
        new_edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    new_edges.append((u, v))
        # Verify regularity
        for i in range(n):
            if len(adj[i]) != k:
                break
        else:
            return n, new_edges
    
    return best_n, best_edges
