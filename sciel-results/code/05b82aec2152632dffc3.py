import time
import heapq
from random import Random


def construct():
    start = time.time()
    DEADLINE = 140.0
    SPECS = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def build_graph(edges):
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        return adj

    def min_cost_flow_lp(N, succ, cap, cost, s, t):
        # min-cost flow of value N via simplex on the standard LP form.
        # x[e] in [0, cap[e]], flow conservation, minimize sum cost[e]*x[e].
        import numpy as np

        m = len(succ)
        nodes = [s, t] + list(range(1, m + 1))  # 0=s, 1=t, 2..=edges
        idx = {s: 0, t: 1}
        for e in range(1, m + 1):
            idx[e] = e + 1
        V = len(nodes)
        # constraints: conservation for each node (V rows), then lower/upper bounds
        # We use: A_eq x = b for conservation; bounds 0<=x<=cap handled via slacks.
        # Build full standard form with explicit slack/surplus vars.
        # Variables: x (m) + surplus for conservation (V) + slacks for upper (m)
        # conservation: for node i: sum_in x - sum_out x = 0 (s,t included, s supplies N)
        # Actually treat as: out - in = b where b[s]=N, b[t]=-N, else 0.
        # Introduce surplus vars for each conservation eq (>= form not needed; use =).
        # For equality, simplex needs slacks on both sides -> use 2 vars per eq.
        # Simpler: convert equality to two inequalities.
        # Row count: V*2 (eq split) + m (upper)
        A = []
        b = []
        # For each conservation eq: out-in = b_i  ->  out-in <= b_i and -(out-in) <= -b_i
        cons = {}  # node -> (list of (coef, varcol))
        for i in range(V):
            cons[i] = []
        # var columns: 0..m-1 are x[e]
        for e in range(m):
            for (u, w) in succ[e]:
                cons[u].append((1.0, e))
                cons[w].append((-1.0, e))
        bvec = [0.0] * V
        bvec[s] = N
        bvec[t] = -N
        for i in range(V):
            row1 = [0.0] * m
            for c, e in cons[i]:
                row1[e] += c
            A.append(row1)
            b.append(bvec[i])
            row2 = [-x for x in row1]
            A.append(row2)
            b.append(-bvec[i])
        # upper bounds: x[e] <= cap[e]
        for e in range(m):
            row = [0.0] * m
            row[e] = 1.0
            A.append(row)
            b.append(cap[e])
        c = [cost[e] for e in range(m)]
        mrows = len(A)
        # Build standard form: A x + s = b, s>=0, x>=0
        # objective: c x + 0 s
        # Use scipy linprog? Not guaranteed. Implement two-phase simplex.
        # We'll implement a compact two-phase simplex.
        def simplex(A, b, c):
            # A: m x n, b: m, c: n (minimize c^T x s.t. A x <= b, x>=0, b>=0)
            m_ = len(b)
            n_ = len(c)
            # tableau: rows m_, cols n_ + m_ (slacks) + 1 (rhs)
            T = []
            for i in range(m_):
                row = A[i] + [0.0] * m_ + [b[i]]
                row[n_ + i] = 1.0
                T.append(row)
            obj = c + [0.0] * m_ + [0.0]
            basis = list(range(n_, n_ + m_))
            # Phase 1 if needed: check b>=0
            if any(bi < -1e-9 for bi in b):
                # add artificial var for each negative row
                # Simplify: multiply that row by -1 (flips inequality) and adjust
                for i in range(m_):
                    if b[i] < -1e-9:
                        T[i] = [-x for x in T[i]]
                        b[i] = -b[i]
                        obj = obj  # unchanged (we just flipped constraint)
                # recompute A,b from T
                for i in range(m_):
                    b[i] = T[i][-1]
            # Now b>=0. Iterate.
            def pivot(T, basis, r, ccol):
                pv = T[r][ccol]
                for j in range(len(T[0])):
                    T[r][j] /= pv
                for i in range(len(T)):
                    if i != r and abs(T[i][ccol]) > 1e-12:
                        f = T[i][ccol]
                        for j in range(len(T[0])):
                            T[i][j] -= f * T[r][j]
                basis[r] = ccol

            for _ in range(200000):
                # find entering: most negative reduced cost in obj row (last row)
                o = T[-1]
                enter = -1
                best = -1e-12
                for j in range(len(c)):
                    if o[j] < best:
                        best = o[j]
                        enter = j
                if enter < 0:
                    break
                # ratio test
                leave = -1
                best_ratio = float('inf')
                for i in range(len(T) - 1):
                    if T[i][enter] > 1e-12:
                        ratio = T[i][-1] / T[i][enter]
                        if ratio < best_ratio - 1e-12:
                            best_ratio = ratio
                            leave = i
                if leave < 0:
                    return None  # unbounded
                pivot(T, basis, leave, enter)
            x = [0.0] * n_
            for i, bv in enumerate(basis):
                if bv < n_:
                    x[bv] = T[i][-1]
            val = sum(c[j] * x[j] for j in range(n_))
            return val, x

        # We need objective row appended to T.
        T = []
        for i in range(mrows):
            T.append(A[i] + [0.0] * mrows + [b[i]])
        # add slack identity
        for i in range(mrows):
            T[i][m + i] = 1.0
        obj = c + [0.0] * mrows + [0.0]
        T.append(obj)
        basis = list(range(m, m + mrows))
        # ensure b>=0 by flipping
        for i in range(mrows):
            if T[i][-1] < -1e-9:
                T[i] = [-x for x in T[i]]
        # iterate
        def pivot(T, basis, r, ccol):
            pv = T[r][ccol]
            for j in range(len(T[0])):
                T[r][j] /= pv
            for i in range(len(T)):
                if i != r and abs(T[i][ccol]) > 1e-12:
                    f = T[i][ccol]
                    for j in range(len(T[0])):
                        T[i][j] -= f * T[r][j]
            basis[r] = ccol

        for _ in range(300000):
            o = T[-1]
            enter = -1
            best = -1e-12
            for j in range(m):
                if o[j] < best:
                    best = o[j]
                    enter = j
            if enter < 0:
                break
            leave = -1
            best_ratio = float('inf')
            for i in range(mrows):
                if T[i][enter] > 1e-12:
                    ratio = T[i][-1] / T[i][enter]
                    if ratio < best_ratio - 1e-12:
                        best_ratio = ratio
                        leave = i
            if leave < 0:
                break
            pivot(T, basis, leave, enter)
        x = [0.0] * m
        for i, bv in enumerate(basis):
            if bv < m:
                x[bv] = max(0.0, T[i][-1])
        return x

    def solve_one(edges, time_budget):
        m = len(edges)
        if m == 0:
            return []
        # map edge -> index
        eidx = {}
        for i, (u, v) in enumerate(edges):
            eidx[(u, v)] = i
        # successors: edge e can be followed by edge e2 if they share a vertex
        # and the shared vertex is an endpoint of e2 (i.e., the path continues).
        # We build a directed graph on edge-nodes: e -> e2 if they share exactly one
        # vertex and e2's other vertex is the "next". Represent edge e as oriented.
        # To keep it tractable, allow e->e2 if |{u,v} ∩ {u2,v2}| = 1 and e2 != e.
        succ = [[] for _ in range(m)]
        cost = [0.0] * m
        for i in range(m):
            u, v = edges[i]
            for j in range(m):
                if i == j:
                    continue
                u2, v2 = edges[j]
                inter = {u, v} & {u2, v2}
                if len(inter) == 1:
                    succ[i].append((j, 0.0))
        # We want to select a set of "links" (i->j) forming a collection of paths
        # covering all edge-nodes, minimizing number of paths = m - (#links).
        # This is a maximum matching-like / min-cost flow on a bipartite-ish structure.
        # Model as: each edge-node has in-degree <=1 and out-degree <=1 in the link
        # selection, and no cycles allowed (paths only). Minimize paths = maximize links.
        # Relax cycles: solve max links with in/out <=1 (assignment), then break cycles.
        # Use min-cost flow: source -> left(i) [cap1], left(i)->right(j) [cap1, cost -1 if succ],
        # right(j)->sink [cap1]. Max flow with min cost (cost -1 per link) => max links.
        # Build network for min-cost flow via successive shortest path with potentials.
        # Nodes: S, T, L_i (m), R_j (m)
        S = 0
        T = 1
        L = lambda i: 2 + i
        R = lambda j: 2 + m + j
        V = 2 + 2 * m
        # adjacency list of (to, cap, cost, rev)
        graph = [[] for _ in range(V)]
        def add_edge(u, v, cap, cost):
            graph[u].append([v, cap, cost, len(graph[v])])
            graph[v].append([u, 0, -cost, len(graph[u]) - 1])
        for i in range(m):
            add_edge(S, L(i), 1, 0)
            add_edge(R(i), T, 1, 0)
            for (j, c) in succ[i]:
                add_edge(L(i), R(j), 1, -1)  # -1 cost to maximize links
        # successive shortest path (Bellman-Ford for potentials since negative edges)
        import collections
        def bellman_ford(src):
            dist = [float('inf')] * V
            inq = [False] * V
            prev = [-1] * V
            dist[src] = 0
            q = collections.deque([src])
            inq[src] = True
            while q:
                u = q.popleft()
                inq[u] = False
                for (v, cap, c, rev) in graph[u]:
                    if cap > 0 and dist[u] + c < dist[v] - 1e-9:
                        dist[v] = dist[u] + c
                        prev[v] = u
                        if not inq[v]:
                            q.append(v)
                            inq[v] = True
            return dist, prev

        flow = 0
        total_cost = 0
        pot = [0.0] * V
        # initial potentials via bellman-ford from S
        dist, prev = bellman_ford(S)
        pot = dist
        while True:
            if time.time() > time_budget:
                break
            # Dijkstra with reduced costs
            dist = [float('inf')] * V
            dist[S] = 0
            prev = [-1] * V
            pq = [(0.0, S)]
            inq = [False] * V
            while pq:
                d, u = heapq.heappop(pq)
                if inq[u]:
                    continue
                inq[u] = True
                if d > dist[u] + 1e-9:
                    continue
                for k, (v, cap, c, rev) in enumerate(graph[u]):
                    if cap > 0:
                        rc = c + pot[u] - pot[v]
                        nd = d + rc
                        if nd < dist[v] - 1e-9:
                            dist[v] = nd
                            prev[v] = u
                            heapq.heappush(pq, (nd, v))
            if dist[T] == float('inf'):
                break
            # update potentials
            for i in range(V):
                if dist[i] < float('inf'):
                    pot[i] += dist[i]
            # augment 1 unit
            v = T
            while v != S:
                u = prev[v]
                # find edge index
                for k, (vv, cap, c, rev) in enumerate(graph[u]):
                    if vv == v and cap > 0:
                        graph[u][k][1] -= 1
                        graph[v][graph[u][k][3]][1] += 1
                        break
                v = u
            flow += 1
            total_cost += pot[T] - pot[S]
            if flow >= m:
                break
        # extract selected links: edges L(i)->R(j) with flow 1
        links = set()
        for i in range(m):
            for (v, cap, c, rev) in graph[L(i)]:
                # original forward edge has rev pointing to reverse; flow = reverse cap
                if v >= 2 + m and v < 2 + 2 * m:
                    j = v - (2 + m)
                    # flow on forward = reverse edge's cap
                    rev_edge = graph[v][rev]
                    if rev_edge[1] > 0:
                        links.add((i, j))
        # Now we have a set of links (i->j) where each node has out<=1, in<=1.
        # This forms paths and cycles. Break cycles by removing one link per cycle.
        # Build next/prev arrays
        nxt = [-1] * m
        prv = [-1] * m
        for (i, j) in links:
            nxt[i] = j
            prv[j] = i
        # find cycles and break
        visited = [False] * m
        for i in range(m):
            if visited[i]:
                continue
            # walk
            cur = i
            seen = {}
            step = 0
            while cur != -1 and not visited[cur]:
                seen[cur] = step
                visited[cur] = True
                step += 1
                cur = nxt[cur]
            if cur != -1 and cur in seen:
                # cycle found: break the link entering the first node of cycle
                # remove link (nxt[prev_of_start]?) simpler: remove one link in cycle
                # find a link (a,b) in cycle to remove
                # walk cycle
                cyc = []
                c2 = cur
                while True:
                    cyc.append(c2)
                    c2 = nxt[c2]
                    if c2 == cur:
                        break
                # remove link (cyc[-1], cyc[0])
                a = cyc[-1]
                b = cyc[0]
                nxt[a] = -1
                prv[b] = -1
        # Now build paths from the link structure
        paths = []
        used = [False] * m
        for i in range(m):
            if used[i]:
                continue
            # find start of this path (node with prv==-1)
            s = i
            while prv[s] != -1:
                s = prv[s]
            # walk
            path_edges = []
            cur = s
            while cur != -1:
                used[cur] = True
                path_edges.append(cur)
                cur = nxt[cur]
            if path_edges:
                paths.append(path_edges)
        # Convert edge-index lists to vertex lists
        result = []
        for pe in paths:
            verts = []
            prev_edge = None
            for ei in pe:
                u, v = edges[ei]
                if not verts:
                    verts = [u, v]
                else:
                    # orient: last vertex of current path must be one endpoint of this edge
                    last = verts[-1]
                    if last == u:
                        verts.append(v)
                    elif last == v:
                        verts.append(u)
                    else:
                        # shouldn't happen if links valid; fallback start new
                        break
            if len(verts) >= 2:
                result.append(verts)
        return result

    def greedy_fallback(edges):
        # simple: each edge its own path
        return [[u, v] for (u, v) in edges]

    results = []
    per_budget = (DEADLINE - (time.time() - start) - 5.0) / 3.0
    for spec in SPECS:
        if time.time() > DEADLINE - 3.0:
            results.append(None)
            continue
        edges = instance(*spec)
        t0 = time.time()
        try:
            paths = solve_one(edges, DEADLINE - 4.0)
        except Exception:
            paths = None
        if not paths:
            paths = greedy_fallback(edges)
        results.append(paths)

    # Fill any None with fallback
    for i, r in enumerate(results):
        if r is None:
            results[i] = greedy_fallback(instance(*SPECS[i]))
    return results
