"""MODEL-AND-OPTIMIZE: CSP solver with MRV, Forward Checking, and AC-3 initialization."""
# EVOLVE-BLOCK-START
import time
from collections import deque
import sys

def construct():
    sys.setrecursionlimit(10000)

    def instance(splits, seed, palette=5):
        from random import Random
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def ac3(n, adj, domains):
        """AC-3 arc consistency preprocessing."""
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        while queue:
            xi, xj = queue.popleft()
            di = domains[xi]
            dj = domains[xj]
            if not di:
                return False
            # For inequality constraint: value c in di is supported iff exists d in dj with d != c
            # This fails only if dj has exactly one value and it equals c
            if len(dj) == 1:
                only = dj[0]
                new_di = [c for c in di if c != only]
                if len(new_di) < len(di):
                    domains[xi] = new_di
                    if not new_di:
                        return False
                    for xk in adj[xi]:
                        if xk != xj:
                            queue.append((xk, xi))
        return True

    def solve_csp(n, edges, lists, deadline):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        domains = [list(l) for l in lists]
        for i in range(n):
            if not domains[i]:
                return None

        if not ac3(n, adj, domains):
            return None
        for i in range(n):
            if not domains[i]:
                return None

        assignment = [-1] * n
        blocked = [set() for _ in range(n)]

        def valid_color_count(v):
            dv = domains[v]
            bv = blocked[v]
            cnt = 0
            for c in dv:
                if c not in bv:
                    cnt += 1
            return cnt

        def backtrack(assigned_count):
            if time.time() > deadline:
                return False
            if assigned_count == n:
                return True

            # MRV: find unassigned vertex with fewest valid colors
            min_var = -1
            min_size = 4
            for v in range(n):
                if assignment[v] == -1:
                    cnt = valid_color_count(v)
                    if cnt < min_size:
                        min_size = cnt
                        min_var = v
                        if min_size == 0:
                            return False
                        if min_size == 1:
                            break

            if min_var == -1:
                return False

            dv = domains[min_var]
            bv = blocked[min_var]
            nbrs = adj[min_var]
            for c in dv:
                if c not in bv:
                    assignment[min_var] = c
                    for u in nbrs:
                        blocked[u].add(c)

                    if backtrack(assigned_count + 1):
                        return True

                    for u in nbrs:
                        blocked[u].discard(c)
                    assignment[min_var] = -1

            return False

        if backtrack(0):
            return assignment
        return None

    def build_result():
        params = ((20, 701), (40, 702), (60, 703), (100, 704))
        start_time = time.time()
        deadline = start_time + 110.0  # Safety margin before 120s limit
        results = []
        for splits, seed in params:
            n, edges, lists = instance(splits, seed)
            coloring = solve_csp(n, edges, lists, deadline)
            if coloring is None:
                # Fallback: greedy by degree
                adj = [[] for _ in range(n)]
                for u, v in edges:
                    adj[u].append(v)
                    adj[v].append(u)
                coloring = [-1] * n
                for v in sorted(range(n), key=lambda x: -len(adj[x])):
                    taken = set()
                    for u in adj[v]:
                        if coloring[u] != -1:
                            taken.add(coloring[u])
                    free = [c for c in lists[v] if c not in taken]
                    coloring[v] = free[0] if free else lists[v][0]
            results.append([int(c) for c in coloring])
        return results

    return build_result()

# EVOLVE-BLOCK-END
