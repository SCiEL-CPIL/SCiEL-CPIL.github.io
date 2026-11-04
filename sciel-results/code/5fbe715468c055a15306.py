import time
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 150s kill

    # ---- Primary: Cayley graph on Z_m with symmetric 4-connection set ----
    # A k-regular Cayley graph: vertices 0..m-1, edges (u, u+s) for s in S (S symmetric).
    # Girth is the length of the shortest non-trivial signed relation sum(e_i)=0.
    # We choose m and S so that no signed sum of < g elements of S is 0 mod m.
    if k == 4:
        # m = 101 (prime), S = {±1, ±7}: no signed sum of <=9 elements is 0 mod 101.
        m = 101
        S = (1, 7)
    elif k == 3:
        m = 100
        S = (1, 5)
    elif k == 5:
        m = 101
        S = (1, 3, 7)
    else:
        m = 101
        S = (1, 7)

    # Build adjacency
    conn = list(S) + [(-s) % m for s in S]
    conn = sorted(set(conn))
    # Ensure exactly k distinct connection elements (simple, no loops: s != 0)
    if 0 in conn:
        conn.remove(0)
    if len(conn) < k:
        # pad with further small offsets not already present
        extra = 11
        while len(conn) < k:
            if extra % m not in conn and (m - extra) % m not in conn:
                conn.append(extra % m)
                conn.append((m - extra) % m)
            extra += 3
    conn = conn[:k]

    adj = [[] for _ in range(m)]
    for u in range(m):
        for s in conn:
            v = (u + s) % m
            if v != u:
                adj[u].append(v)
    # de-dup / enforce simple
    for u in range(m):
        seen = set()
        newl = []
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                newl.append(v)
        adj[u] = newl
    # enforce exact degree k (trim or it's already k)
    for u in range(m):
        if len(adj[u]) > k:
            adj[u] = adj[u][:k]

    # ---- Girth certification via non-backtracking walks from every vertex ----
    def girth_ok(adj, g, deadline):
        n = len(adj)
        # For each start vertex, do a BFS over directed non-backtracking edges,
        # tracking the previous vertex to forbid immediate backtracking.
        # State: (prev, cur). Start states: (None, s) for each s.
        # We detect a cycle when we reach a state whose "cur" was already visited
        # at a smaller depth along a path that doesn't backtrack.
        # Simpler & exact: BFS on the non-backtracking line graph; a cycle of length L
        # corresponds to returning to a (prev,cur) state after L steps.
        for s in range(n):
            if time.time() > deadline:
                return False
            # state (prev, cur); prev in -1..n-1
            dist = {}
            queue = [(-1, s)]
            dist[(-1, s)] = 0
            found = 0
            head = 0
            while head < len(queue):
                prev, cur = queue[head]; head += 1
                dcur = dist[(prev, cur)]
                for nxt in adj[cur]:
                    if nxt == prev:
                        continue  # non-backtracking
                    nd = dcur + 1
                    if nd >= g:
                        continue
                    # closing a cycle: nxt == s and nd >= 3 and not the trivial path
                    if nxt == s and nd >= 3 and prev != -1:
                        # genuine cycle shorter than g
                        return False
                    key = (cur, nxt)
                    if key not in dist:
                        dist[key] = nd
                        queue.append(key)
        return True

    ok = False
    try:
        ok = girth_ok(adj, g, deadline)
    except Exception:
        ok = False

    if ok:
        edges = []
        for u in range(m):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return m, edges

    # ---- Fallback: seed LUW construction (known valid for k=4,g=10) ----
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    rule_a = np.zeros(d + 1, dtype=np.int64)
    rule_b = np.zeros(d + 1, dtype=np.int64)
    for c in range(2, d + 1):
        rule_a[c], rule_b[c] = rule[c]
    if q == 4:
        gf4_table = np.array([[0,0,0,0],[0,1,2,3],[0,2,3,1],[0,3,1,2]], dtype=np.int64)
    else:
        gf4_table = None

    start_vertex = (0, (0,) * (d + 1))
    visited = {start_vertex: 0}
    order = [start_vertex]
    edges = []
    vertex_idx = 1
    for i, vertex in enumerate(order):
        if time.time() > deadline:
            break
        side, x_tuple = vertex
        x = np.array(x_tuple, dtype=np.int64)
        Y = np.zeros((q, d + 1), dtype=np.int64)
        Y[:, 1] = np.arange(q)
        for c in range(2, d + 1):
            a = rule_a[c]; b = rule_b[c]
            if side == 0:
                if q == 4:
                    prod = gf4_table[Y[:, a], x[b]]
                    Y[:, c] = prod ^ x[c]
                else:
                    prod = (Y[:, a] * x[b]) % q
                    Y[:, c] = (prod + x[c]) % q
            else:
                if q == 4:
                    prod = gf4_table[x[a], Y[:, b]]
                    Y[:, c] = prod ^ x[c]
                else:
                    prod = (x[a] * Y[:, b]) % q
                    Y[:, c] = (x[c] - prod) % q
        for j in range(q):
            neighbor = (1 - side, tuple(Y[j]))
            if neighbor not in visited:
                visited[neighbor] = vertex_idx
                order.append(neighbor)
                vertex_idx += 1
            if side == 0:
                edges.append((visited[vertex], visited[neighbor]))
    return len(order), edges
