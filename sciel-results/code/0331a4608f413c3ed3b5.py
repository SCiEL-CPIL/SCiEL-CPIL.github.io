import time
import random
from collections import deque
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def extract_cycles_milp(n, remaining_edges):
        if not remaining_edges:
            return []

        directed_edges = []
        edge_to_idx = {}
        for u, v in remaining_edges:
            directed_edges.append((u, v))
            directed_edges.append((v, u))
            edge_to_idx[(u, v)] = len(directed_edges) - 2
            edge_to_idx[(v, u)] = len(directed_edges) - 1
        
        m = len(directed_edges)
        if m == 0:
            return []

        c = [-1.0] * m

        A_rows = []
        b_lower = []
        b_upper = []

        for u in range(n):
            row = [0.0] * m
            for idx, (a, b) in enumerate(directed_edges):
                if a == u:
                    row[idx] = 1.0
            A_rows.append(row)
            b_lower.append(0.0)
            b_upper.append(1.0)

        for v in range(n):
            row = [0.0] * m
            for idx, (a, b) in enumerate(directed_edges):
                if b == v:
                    row[idx] = 1.0
            A_rows.append(row)
            b_lower.append(0.0)
            b_upper.append(1.0)

        for u, v in remaining_edges:
            row = [0.0] * m
            idx1 = edge_to_idx[(u, v)]
            idx2 = edge_to_idx[(v, u)]
            row[idx1] = 1.0
            row[idx2] = 1.0
            A_rows.append(row)
            b_lower.append(0.0)
            b_upper.append(1.0)

        if not A_rows:
            return []

        A = [A_rows]
        lb = [b_lower]
        ub = [b_upper]
        
        bounds = Bounds(lb=[0.0] * m, ub=[1.0] * m)
        integrality = [1] * m

        try:
            res = milp(c=c, constraints=LinearConstraint(A, lb, ub), bounds=bounds, integrality=integrality, options={"time_limit": 5.0})
            if not res.success:
                return []
            
            x = res.x
            if x is None:
                return []
            
            next_vertex = [-1] * n
            for i, (u, v) in enumerate(directed_edges):
                if x[i] > 0.5:
                    next_vertex[u] = v
            
            cycles = []
            visited = [False] * n
            for i in range(n):
                if not visited[i] and next_vertex[i] != -1:
                    cycle = []
                    cur = i
                    while not visited[cur]:
                        visited[cur] = True
                        cycle.append(cur)
                        cur = next_vertex[cur]
                    if len(cycle) >= 3:
                        cycles.append(cycle)
            
            return cycles

        except Exception:
            return []

    def find_longest_cycle(adj, n, remaining_edges):
        best_len = 0
        best_cycle = []

        for start in range(n):
            if not adj[start]:
                continue

            visited = {start: 0}
            parent = {start: None}
            in_stack = set([start])

            def dfs(u):
                nonlocal best_len, best_cycle
                for v in sorted(adj[u]):
                    e = (min(u, v), max(u, v))
                    if e not in remaining_edges:
                        continue
                    if v not in visited:
                        visited[v] = visited[u] + 1
                        parent[v] = u
                        in_stack.add(v)
                        dfs(v)
                        in_stack.remove(v)
                    elif v in in_stack:
                        length = visited[u] - visited[v] + 1
                        if length >= 3 and length > best_len:
                            cycle = [u]
                            cur = u
                            while cur != v:
                                cur = parent[cur]
                                cycle.append(cur)
                            best_cycle = cycle
                            best_len = length

            dfs(start)
            if time.time() > deadline:
                break

        return best_cycle

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))

    for spec in specs:
        if time.time() > deadline:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue

        edges = instance(*spec)
        n = spec[0]
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        pieces = []
        remaining_edges = set(edges)

        while remaining_edges and time.time() < deadline:
            cycles = extract_cycles_milp(n, remaining_edges)
            
            if not cycles:
                # Fallback to heuristic longest cycle
                best_cycle = find_longest_cycle(adj, n, remaining_edges)
                if best_cycle:
                    cycles = [best_cycle]
                else:
                    break
            
            for cycle in cycles:
                pieces.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    e = (min(u, v), max(u, v))
                    if e in remaining_edges:
                        remaining_edges.discard(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
            
            if time.time() > deadline:
                break

        for e in remaining_edges:
            pieces.append(list(e))

        results.append(pieces)

    return results
