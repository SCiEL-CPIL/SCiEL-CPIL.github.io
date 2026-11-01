import time
import random
import numpy as np

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    Uses voltage assignment on K_{4,4} for k=4.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 second margin

    if k != 4:
        return _seed_construct(k, g)

    # Base graph: K_{4,4}
    # Vertices: 0..3 (left), 4..7 (right)
    base_vertices = 8
    base_edges = []
    for i in range(4):
        for j in range(4):
            base_edges.append((i, 4 + j))

    best_n = 320
    best_edges = None

    # Moore bound for (4,10): 1 + 4 + 12 + 36 + 108 = 261
    # So m >= ceil(261/8) = 33. We try m from 33 upward.
    # Target: find smallest m with a valid voltage assignment.

    # Precompute all cycles of length 4, 6, 8 in K_{4,4} as edge sequences.
    # Each cycle is a list of (u,v) directed edges forming a closed walk.
    # We'll store them as tuples of directed-edge indices.
    # Edge index: for edge (u,v) in base_edges, index = position in base_edges list.
    # We need directed edges: (u,v) and (v,u).

    # Build directed edge lookup: (u,v) -> (edge_idx, direction_sign)
    # For undirected edge e=(u,v) in base_edges, directed (u,v) has sign +1, (v,u) has sign -1.
    dir_edge_map = {}
    for idx, (u, v) in enumerate(base_edges):
        dir_edge_map[(u, v)] = (idx, 1)
        dir_edge_map[(v, u)] = (idx, -1)

    # Enumerate all simple cycles of length 4, 6, 8 in K_{4,4}
    # K_{4,4} is bipartite with parts L={0,1,2,3} and R={4,5,6,7}
    # A cycle of length 2c uses c left and c right vertices.

    def enumerate_cycles(c):
        """Enumerate all simple cycles of length 2c in K_{4,4}."""
        cycles = []
        left = [0, 1, 2, 3]
        right = [4, 5, 6, 7]

        if c == 2:
            # 4-cycles: choose 2 left, 2 right, and the 2 possible matchings
            for i in range(4):
                for j in range(i + 1, 4):
                    for a in range(4):
                        for b in range(a + 1, 4):
                            u1, u2 = left[i], left[j]
                            v1, v2 = right[a], right[b]
                            # Cycle 1: u1-v1-u2-v2-u1
                            cyc1 = [(u1, v1), (v1, u2), (u2, v2), (v2, u1)]
                            cycles.append(cyc1)
                            # Cycle 2: u1-v2-u2-v1-u1
                            cyc2 = [(u1, v2), (v2, u2), (u2, v1), (v1, u1)]
                            cycles.append(cyc2)
        elif c == 3:
            # 6-cycles: choose 3 left, 3 right, and all bijections (3! = 6)
            for i in range(4):
                for j in range(i + 1, 4):
                    for l in range(j + 1, 4):
                        for a in range(4):
                            for b in range(a + 1, 4):
                                for d in range(b + 1, 4):
                                    u1, u2, u3 = left[i], left[j], left[l]
                                    v1, v2, v3 = right[a], right[b], right[d]
                                    # All 6 permutations of right vertices
                                    from itertools import permutations
                                    for perm in permutations([v1, v2, v3]):
                                        w1, w2, w3 = perm
                                        cyc = [(u1, w1), (w1, u2), (u2, w2),
                                               (w2, u3), (u3, w3), (w3, u1)]
                                        cycles.append(cyc)
        elif c == 4:
            # 8-cycles: use all 4 left and all 4 right.
            # Fix u1=0 (left[0]) to break rotational symmetry.
            # Then enumerate permutations of remaining 3 left and all 4 right.
            # Actually, let's just enumerate all ways to pick an ordering.
            # Fix first left vertex to 0 to avoid duplicates (each cycle counted 8 times total, 4 rotations * 2 directions, fixing first left gives 4*2=8/8=1... let me just enumerate all and dedupe by frozenset of edges)
            # Simpler: enumerate all permutations of left (4!) and right (4!), fix u1=left[0] by only considering perms where first element is 0.
            from itertools import permutations
            left_perms = [p for p in permutations([0, 1, 2, 3]) if p[0] == 0]
            right_perms = list(permutations([4, 5, 6, 7]))
            seen = set()
            for lp in left_perms:
                for rp in right_perms:
                    u1, u2, u3, u4 = lp
                    v1, v2, v3, v4 = rp
                    cyc = [(u1, v1), (v1, u2), (u2, v2), (v2, u3),
                           (u3, v3), (v3, u4), (u4, v4), (v4, u1)]
                    # Canonical form: frozenset of undirected edges
                    key = frozenset(frozenset(e) for e in cyc)
                    if key not in seen:
                        seen.add(key)
                        cycles.append(cyc)

    cycles_4 = enumerate_cycles(2)
    cycles_6 = enumerate_cycles(3)
    cycles_8 = enumerate_cycles(4)

    # Convert cycles to edge-index-with-sign arrays for fast sum computation
    # For each cycle, store list of (edge_idx, sign)
    def cycle_to_edge_sigs(cycle):
        sigs = []
        for u, v in cycle:
            idx, sign = dir_edge_map[(u, v)]
            sigs.append((idx, sign))
        return sigs

    cycles_4_sigs = [cycle_to_edge_sigs(c) for c in cycles_4]
    cycles_6_sigs = [cycle_to_edge_sigs(c) for c in cycles_6]
    cycles_8_sigs = [cycle_to_edge_sigs(c) for c in cycles_8]

    # For fast checking, we can use numpy arrays
    # Each cycle becomes an array of (edge_idx, sign)
    # voltages is an array of length 16 (one per undirected edge)
    # Sum for a cycle = sum(sign_i * voltages[edge_idx_i]) mod m

    # Try m from 33 upward
    for m in range(33, 40):
        if time.time() > deadline:
            break

        found = False
        num_attempts = 2000
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break

            # Random voltage assignment
            voltages = np.random.randint(1, m, size=16)  # non-zero

            # Check all cycles
            valid = True

            # Check 4-cycles
            for sigs in cycles_4_sigs:
                s = 0
                for idx, sign in sigs:
                    s += sign * voltages[idx]
                if s % m == 0:
                    valid = False
                    break
            if not valid:
                continue

            # Check 6-cycles
            for sigs in cycles_6_sigs:
                s = 0
                for idx, sign in sigs:
                    s += sign * voltages[idx]
                if s % m == 0:
                    valid = False
                    break
            if not valid:
                continue

            # Check 8-cycles
            for sigs in cycles_8_sigs:
                s = 0
                for idx, sign in sigs:
                    s += sign * voltages[idx]
                if s % m == 0:
                    valid = False
                    break
            if not valid:
                continue

            # All checks passed! Construct the lifted graph
            n = base_vertices * m
            edges = []
            for idx, (u, v) in enumerate(base_edges):
                val = voltages[idx]
                for g_elem in range(m):
                    u_new = u * m + g_elem
                    v_new = v * m + (g_elem + val) % m
                    edges.append((u_new, v_new))

            # Verify girth with non-backtracking walks
            if _check_girth_fast(n, edges, g):
                best_n = n
                best_edges = edges
                found = True
                break

        if found:
            break  # Found valid graph for this m, which is the smallest m tried

    if best_edges is not None:
        return best_n, best_edges

    # Fallback
    return _seed_construct(k, g)

def _check_girth_fast(n, edges, g):
    """
    Check girth >= g using non-backtracking walks from every vertex.
    A non-backtracking walk of length L < g starting at any vertex
    that returns to the start indicates a cycle of length <= L < g.
    We do BFS on the non-backtracking line graph.
    """
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # For each vertex, do BFS on non-backtracking walks
    # State: (current_vertex, previous_vertex)
    # We track visited states to avoid cycles in the BFS itself.
    # If we reach (start, prev) where prev is a neighbor of start, that's a cycle.
    # Actually, a non-backtracking walk of length L from start returns to start
    # iff the last step is from some neighbor of start to start.
    # So we check: can we reach state (start, v) for any neighbor v of start,
    # in fewer than g-1 steps? (The walk has L steps, and we need L < g)

    # More precisely: a closed non-backtracking walk of length L starting at s
    # ends at s. The state after L steps is (s, prev) where prev is the vertex
    # at step L-1. We need L < g.
    # So for each start vertex s, we do BFS on states (u, v) where v is the previous vertex.
    # Initial states: (s, v) for each neighbor v of s, at distance 1.
    # We want to know if any state (s, v) is reached at distance < g-1.
    # Wait, the walk length is the number of edges. If we start at s and take 1 step to v,
    # the state is (v, s) at distance 1. We want to return to s, so the state should be (s, v)
    # for some neighbor v. The distance in the BFS is the walk length.
    # So we check if (s, v) is reachable in < g steps.

    # To make this efficient, we can do a multi-source BFS or just check each start.
    # For n up to ~300, this should be fast enough.

    for start in range(n):
        # BFS on non-backtracking states
        # State: (current, prev)
        # dist[state] = length of walk
        visited = set()
        queue = []
        for v in adj[start]:
            state = (v, start)
            visited.add(state)
            queue.append((state, 1))

        head = 0
        found_short = False
        while head < len(queue):
            (u, prev), d = queue[head]
            head += 1

            # If we're back at start, we found a cycle of length d
            if u == start:
                if d < g:
                    return False
                continue  # Don't expand further from a closed walk

            # Expand: go from u to w, where w != prev
            for w in adj[u]:
                if w == prev:
                    continue
                new_state = (w, u)
                if new_state not in visited:
                    visited.add(new_state)
                    if d + 1 < g:  # Only enqueue if we haven't exceeded girth
                        queue.append((new_state, d + 1))

    return True

def _check_girth(n, edges, g):
    """Check if graph has girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and u != parent[v]:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True

def _seed_construct(k, g):
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
