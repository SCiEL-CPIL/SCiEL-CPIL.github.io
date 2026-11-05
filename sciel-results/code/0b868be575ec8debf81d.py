import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 105  # 45s safety margin before 150s kill

    # Voltage-graph approach over K4,4
    # K4,4: U={0,1,2,3}, V={4,5,6,7}
    # Voltage x[u][v] for edge u->v (u in U, v in V)
    # Lift: 8n vertices, (side, idx, lift) where side 0=U, 1=V

    # Precompute non-backtracking closed walks of length 4, 6, 8 in K4,4
    def get_nb_walks(length):
        walks = []
        for start in range(8):
            # DFS with non-backtracking constraint
            stack = [(start, -1, [start])]
            while stack:
                curr, prev, path = stack.pop()
                if len(path) == length + 1:
                    if path[-1] == start:
                        # Verify non-backtracking: path[i] != path[i-2] for all i>=2
                        valid = True
                        for i in range(2, len(path)):
                            if path[i] == path[i-2]:
                                valid = False
                                break
                        if valid:
                            walks.append(path)
                    continue
                for nxt in range(8):
                    # Must be adjacent in K4,4: one in U, one in V
                    if (curr < 4 and nxt >= 4) or (curr >= 4 and nxt < 4):
                        if nxt != prev:  # non-backtracking
                            stack.append((nxt, curr, path + [nxt]))
        return walks

    walks_4 = get_nb_walks(4)
    walks_6 = get_nb_walks(6)
    walks_8 = get_nb_walks(8)

    # For each walk, precompute the sign pattern and vertex pairs
    # voltage contribution: +x[u][v] if U->V, -x[u][v] if V->U
    def precompute_walk_info(walks):
        info = []
        for w in walks:
            terms = []  # list of (u, v, sign) where sign=+1 for U->V, -1 for V->U
            for i in range(len(w) - 1):
                u, v = w[i], w[i+1]
                if u < 4:  # U to V
                    terms.append((u, v - 4, 1))
                else:  # V to U
                    terms.append((v - 4, u, -1))
            info.append(terms)
        return info

    walk_info_4 = precompute_walk_info(walks_4)
    walk_info_6 = precompute_walk_info(walks_6)
    walk_info_8 = precompute_walk_info(walks_8)

    def check_voltages(x, n):
        """Check if voltage assignment x (4x4) has no zero-sum walk of length 4,6,8 mod n."""
        for terms in walk_info_4:
            s = 0
            for u, v, sign in terms:
                s += sign * x[u][v]
            if s % n == 0:
                return False
        for terms in walk_info_6:
            s = 0
            for u, v, sign in terms:
                s += sign * x[u][v]
            if s % n == 0:
                return False
        for terms in walk_info_8:
            s = 0
            for u, v, sign in terms:
                s += sign * x[u][v]
            if s % n == 0:
                return False
        return True

    def construct_graph(x, n):
        """Build the lifted graph from voltage assignment."""
        edges = []
        for u in range(4):
            for v in range(4):
                volt = x[u][v]
                for i in range(n):
                    u_id = u * n + i
                    v_id = 4 * n + v * n + ((i + volt) % n)
                    edges.append((u_id, v_id))
        return 8 * n, edges

    def verify_girth(n_graph, edges, required_girth):
        """Verify girth >= required_girth using BFS from each vertex."""
        adj = [[] for _ in range(n_graph)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        min_girth = float('inf')
        for start in range(n_graph):
            dist = [-1] * n_graph
            parent = [-1] * n_graph
            q = deque([start])
            dist[start] = 0
            while q:
                curr = q.popleft()
                for nei in adj[curr]:
                    if dist[nei] == -1:
                        dist[nei] = dist[curr] + 1
                        parent[nei] = curr
                        q.append(nei)
                    elif nei != parent[curr]:
                        cycle_len = dist[curr] + dist[nei] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                            if min_girth < required_girth:
                                return False
        return min_girth >= required_girth

    # Search for voltage assignments
    # Try n from 37 upward (Moore bound for (4,10) is 243, so n=31 gives 248 vertices)
    # Actually 8*31=248, 8*30=240 < 243 so n>=31
    best_result = None

    for n in range(31, 60):
        if time.time() > deadline:
            break
        target_vertices = 8 * n

        # Try random and local search
        for attempt in range(200):
            if time.time() > deadline:
                break
            # Random initial assignment
            x = [[random.randint(0, n-1) for _ in range(4)] for _ in range(4)]

            # Local improvement: try to fix violations
            improved = True
            max_iter = 50
            while improved and max_iter > 0:
                improved = False
                max_iter -= 1
                for u in range(4):
                    for v in range(4):
                        if time.time() > deadline:
                            break
                        old_val = x[u][v]
                        for new_val in range(n):
                            if new_val == old_val:
                                continue
                            x[u][v] = new_val
                            if check_voltages(x, n):
                                improved = True
                                break
                            x[u][v] = old_val
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break
                if time.time() > deadline:
                    break

            if check_voltages(x, n):
                n_graph, edges = construct_graph(x, n)
                if verify_girth(n_graph, edges, g):
                    if best_result is None or n_graph < best_result[0]:
                        best_result = (n_graph, edges)
                    # Found valid, try smaller n
                    break

    if best_result is not None:
        return best_result

    # Fallback: optimized LUW construction
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

    rule_a = [0] * (d + 1)
    rule_b = [0] * (d + 1)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [0] * (d + 1)
            y[1] = first
            if side == 0:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = add(x[c], mul(y[a], x[b]))
            else:
                for c in range(2, d + 1):
                    a, b = rule_a[c], rule_b[c]
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (0,) * (d + 1))
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
