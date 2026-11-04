import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    # 1. Generate LUW graph as fallback
    n_luw, edges_luw = _luw_graph(k, g)

    # 2. Try to find a smaller graph using voltage assignment on K_4
    if k == 3 and g == 16:
        # Target n=936, Base K_4 (4 vertices), Group Z_234
        result = _search_k4_voltage(k, g, 234, deadline)
        if result:
            n, edges = result
            if n < n_luw:
                return n, edges

    return n_luw, edges_luw

def _luw_graph(k, g):
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

def _search_k4_voltage(k, g, Q, deadline):
    # K_4 edges: 0-1, 0-2, 0-3, 1-2, 1-3, 2-3
    # For a voltage assignment on K_4, the lift has 4Q vertices.
    # A cycle in the lift projects to a closed walk in K_4.
    # The voltage around the closed walk must be 0 for the lift to close.
    #
    # Short cycles (length < g) in the lift correspond to closed walks in K_4
    # of length < g whose total voltage is 0 mod Q.
    #
    # For girth >= 16, we need no closed walk of length 1-15 with voltage 0.
    # Since K_4 is simple, no closed walks of length 1 or 2.
    # Length 3: triangles in K_4. K_4 has 4 triangles.
    #   Triangle 0-1-2-0: voltage v01 + v12 + v20 = v01 + v12 - v02
    #   Triangle 0-1-3-0: voltage v01 + v13 - v03
    #   Triangle 0-2-3-0: voltage v02 + v23 - v03
    #   Triangle 1-2-3-0: wait, 1-2-3-1: voltage v12 + v23 - v13
    #   Actually K_4 has 4 triangles: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    #   (0,1,2): v01 + v12 - v02 ≡ 0
    #   (0,1,3): v01 + v13 - v03 ≡ 0
    #   (0,2,3): v02 + v23 - v03 ≡ 0
    #   (1,2,3): v12 + v23 - v13 ≡ 0
    #
    # Length 4: 4-cycles in K_4. K_4 has 3 four-cycles (each missing one vertex).
    #   (0,1,2,3,0): v01 + v12 + v23 + v30 = v01 + v12 + v23 - v03
    #   (0,1,3,2,0): v01 + v13 + v32 + v20 = v01 + v13 - v23 - v02
    #   (0,2,1,3,0): v02 + v21 + v13 + v30 = v02 - v12 + v13 - v03
    #
    # For girth >= 16, we need all these to be non-zero mod Q.
    # This is a system of constraints on the 6 voltages.
    #
    # Strategy: Random search with constraint checking.
    # Check if a voltage assignment satisfies all constraints for cycles of length 3-15.
    # For length 3-4, we have explicit constraints. For longer, we need BFS.
    #
    # But first, let's just check length 3 and 4 constraints, then do BFS for the rest.

    best_girth = 0
    best_voltages = None

    # Try many random assignments
    for attempt in range(50000):
        if time.time() > deadline:
            break

        # Random voltages
        current = [random.randint(0, Q - 1) for _ in range(6)]
        v01, v02, v03, v12, v13, v23 = current

        # Check simplicity: no two edges between same pair of vertices in lift
        # For K_4 lift, edges are:
        # (0g, 1(g+v01)) for all g
        # (0g, 2(g+v02)) for all g
        # (0g, 3(g+v03)) for all g
        # (1g, 2(g+v12)) for all g
        # (1g, 3(g+v13)) for all g
        # (2g, 3(g+v23)) for all g
        #
        # Multi-edge if two different g values give same (u,v) pair.
        # E.g., (0g, 1(g+v01)) = (0g', 1(g'+v01)) implies g=g', so no issue for same base edge.
        # But could (0g, 1(g+v01)) = (1g', 0(g'-v01))? No, base vertices differ.
        # Could (0g, 2(g+v02)) = (0g', 2(g'+v02))? Only if g=g'.
        # Could (0g, 1(g+v01)) = (2g', 0(g'-v02))? No, base vertices 0-1 vs 2-0.
        # Actually, edges are undirected. So (u,v) same as (v,u).
        # (0g, 1(g+v01)) and (1g', 0(g'-v01)) would be the same edge if 0g=1g' and 1(g+v01)=0(g'-v01), impossible.
        # So simplicity is automatic for voltage lifts of simple graphs? 
        # Wait, no. Consider (0g, 2(g+v02)) and (2g', 0(g'-v02)). These are the same undirected edge.
        # But they come from the same base edge 0-2, just traversed in opposite directions.
        # In the lift, for each g, we add edge (0g, 2(g+v02)). For g'=g+v02, we add (2(g+v02), 0(g+v02-v02))=(2(g+v02), 0g).
        # So the edge (0g, 2(g+v02)) appears twice: once from g in the 0-2 loop, once from g+v02 in the 2-0 direction.
        # But we're only adding edges in one direction (from lower base vertex to higher, or just once per base edge).
        # In _build_k4_lift, we add each base edge once per g. So no duplicates.
        # So simplicity is automatic.

        # Check girth constraints for length 3 and 4
        c3_012 = (v01 + v12 - v02) % Q
        c3_013 = (v01 + v13 - v03) % Q
        c3_023 = (v02 + v23 - v03) % Q
        c3_123 = (v12 + v23 - v13) % Q

        c4_0123 = (v01 + v12 + v23 - v03) % Q
        c4_0132 = (v01 + v13 - v23 - v02) % Q
        c4_0213 = (v02 - v12 + v13 - v03) % Q

        # For girth >= 16, all these must be non-zero
        if c3_012 == 0 or c3_013 == 0 or c3_023 == 0 or c3_123 == 0:
            continue
        if c4_0123 == 0 or c4_0132 == 0 or c4_0213 == 0:
            continue

        # Now check for cycles of length 5-15 using BFS
        girth = _check_girth_fast(current, Q, deadline)

        if girth >= g:
            return _build_k4_lift(current, Q)

        if girth > best_girth:
            best_girth = girth
            best_voltages = current.copy()

    # Return best found if valid
    if best_voltages and best_girth >= g:
        return _build_k4_lift(best_voltages, Q)

    return None

def _check_girth_fast(voltages, Q, deadline):
    # Build adjacency list
    N = 4 * Q
    adj = [[] for _ in range(N)]
    v01, v02, v03, v12, v13, v23 = voltages

    for g in range(Q):
        u = 0 * Q + g
        adj[u].append(1 * Q + (g + v01) % Q)
        adj[u].append(2 * Q + (g + v02) % Q)
        adj[u].append(3 * Q + (g + v03) % Q)

    for g in range(Q):
        u = 1 * Q + g
        adj[u].append(0 * Q + (g - v01) % Q)
        adj[u].append(2 * Q + (g + v12) % Q)
        adj[u].append(3 * Q + (g + v13) % Q)

    for g in range(Q):
        u = 2 * Q + g
        adj[u].append(0 * Q + (g - v02) % Q)
        adj[u].append(1 * Q + (g - v12) % Q)
        adj[u].append(3 * Q + (g + v23) % Q)

    for g in range(Q):
        u = 3 * Q + g
        adj[u].append(0 * Q + (g - v03) % Q)
        adj[u].append(1 * Q + (g - v13) % Q)
        adj[u].append(2 * Q + (g - v23) % Q)

    # BFS from each vertex to find shortest cycle
    min_girth = 100
    start_time = time.time()

    for start in range(N):
        if time.time() > deadline:
            break
        dist = [-1] * N
        dist[start] = 0
        queue = deque([start])
        while queue:
            u = queue.popleft()
            if dist[u] >= 7:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                    if min_girth < 8:
                        return min_girth

    return min_girth

def _build_k4_lift(voltages, Q):
    # Build the edge list
    edges = []
    v01, v02, v03, v12, v13, v23 = voltages

    for g in range(Q):
        u = 0 * Q + g
        v = 1 * Q + (g + v01) % Q
        edges.append((u, v))
        v = 2 * Q + (g + v02) % Q
        edges.append((u, v))
        v = 3 * Q + (g + v03) % Q
        edges.append((u, v))

    for g in range(Q):
        u = 1 * Q + g
        v = 2 * Q + (g + v12) % Q
        edges.append((u, v))
        v = 3 * Q + (g + v13) % Q
        edges.append((u, v))

    for g in range(Q):
        u = 2 * Q + g
        v = 3 * Q + (g + v23) % Q
        edges.append((u, v))

    return 4 * Q, edges
