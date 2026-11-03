"""Deterministic algebraic construction for regular graphs of large girth."""
import time
import random

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = 115.0  # seconds; evaluator kills at 150s, keep margin

    # Use the LUW-type algebraic construction for all cases
    q = k

    if q == 4:
        table = [[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]]
        add = lambda x,y: x^y
        sub = lambda x,y: x^y
        mul = lambda x,y: table[x][y]
    else:
        add = lambda x,y: (x+y)%q
        sub = lambda x,y: (x-y)%q
        mul = lambda x,y: x*y%q

    def build_rule(d_val):
        """Precompute the LUW rule table for a given depth."""
        rule = [None, None, (1,1), (2,1), (1,2)]
        for c in range(5, d_val+1):
            r = (c-5) % 4
            c0 = c - r
            rule.append([(1,c0-2),(c0-1,1),(c0,1),(1,c0+1)][r])
        return rule

    def build_graph(d_val, rule):
        """Build the LUW graph for given depth and rule table."""
        start = (0, (None,) + (0,)*d_val)
        index = {start: 0}
        order = [start]
        edges = []

        for vertex in order:
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0]*(d_val-1)
                for c in range(2, d_val+1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                other = (1-side, tuple(y))
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if side == 0:
                    edges.append((index[vertex], index[other]))

        return len(order), edges

    def verify_girth_fast(n, edges, min_girth):
        """Fast girth verification using non-backtracking walks."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        max_depth = (min_girth - 1) // 2
        for start in range(n):
            # Non-backtracking BFS: state is (node, parent)
            visited = {}
            queue = [(start, -1)]
            depth = 0
            found_cycle = False
            while queue and depth < max_depth:
                next_queue = []
                for (u, pu) in queue:
                    for v in adj[u]:
                        if v == pu:
                            continue
                        if v == start and depth + 1 >= min_girth:
                            # This would be a valid cycle of length >= min_girth
                            continue
                        if v == start and depth + 1 < min_girth:
                            # Short cycle found
                            return False
                        state = (v, u)
                        if state not in visited:
                            visited[state] = depth + 1
                            next_queue.append(state)
                queue = next_queue
                depth += 1
        return True

    def verify_girth(n, edges, min_girth):
        """Verify girth >= min_girth using non-backtracking BFS from each vertex."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        max_dist = (min_girth - 1) // 2
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = [start]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                du = dist[u]
                if du >= max_dist:
                    continue
                pu = parent[u]
                for v in adj[u]:
                    if v == pu:
                        continue
                    dv = dist[v]
                    if dv == -1:
                        dist[v] = du + 1
                        parent[v] = u
                        queue.append(v)
                    elif dv >= du:
                        cycle_len = du + dv + 1
                        if cycle_len < min_girth:
                            return False
        return True

    # Choose the minimal depth d such that the LUW construction has girth >= g
    # The LUW construction with depth d typically has girth around 2*d + 1 or 2*d + 2
    # Start with a conservative estimate and verify
    d_min = max(3, (g + 1) // 2 - 1)

    best_n = None
    best_edges = None

    for d_try in range(d_min, d_min + 8):
        if time.time() - t0 > deadline - 10:
            break
        rule = build_rule(d_try)
        n, edges = build_graph(d_try, rule)

        # Quick size check: if n is already larger than best, skip verification
        if best_n is not None and n >= best_n:
            break

        if verify_girth(n, edges, g):
            best_n = n
            best_edges = edges
            # If this is the first valid one, we can try smaller d if possible
            # but since we're going up, keep searching for smaller n (shouldn't happen)
            break

    if best_n is not None:
        return best_n, best_edges

    # Fallback: return the last constructed graph
    rule = build_rule(d_min)
    n, edges = build_graph(d_min, rule)
    return n, edges
# EVOLVE-BLOCK-END
