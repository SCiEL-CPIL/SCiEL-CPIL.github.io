"""MODEL-AND-OPTIMIZE: Bipartite-aware CSP solver with AC-3, parity ordering, and MRV."""
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

    def bipartition(n, edges):
        """2-color the bipartite graph to get parity classes."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        color = [-1] * n
        for start in range(n):
            if color[start] != -1:
                continue
            color[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if color[v] == -1:
                        color[v] = 1 - color[u]
                        q.append(v)
        return color

    def ac3(n, adj, domains):
        """AC-3 arc consistency preprocessing for inequality constraints."""
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
            if len(dj) == 1:
                only = dj[0]
                if only in di:
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

        # Get bipartition for ordering
        parity = bipartition(n, edges)

        # Pre-compute: for each vertex, sort neighbours by domain size (ascending)
        # for more effective forward checking

        assignment = [-1] * n
        blocked = [set() for _ in range(n)]

        # Count unassigned neighbours for each vertex (for MRV tie-breaking)
        unassigned_count = [len(adj[v]) for v in range(n)]

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

            # MRV with parity-aware tie-breaking:
            # Prefer unassigned vertex with fewest valid colors;
            # break ties by parity class (alternate to balance), then by degree
            min_var = -1
            min_size = 4
            min_parity = -1
            min_deg = -1
            for v in range(n):
                if assignment[v] == -1:
                    cnt = valid_color_count(v)
                    if cnt < min_size:
                        min_size = cnt
                        min_var = v
                        min_parity = parity[v]
                        min_deg = len(adj[v])
                        if min_size == 0:
                            return False
                        if min_size == 1:
                            break
                    elif cnt == min_size:
                        # Tie-break: prefer vertex with fewer unassigned neighbours
                        # (more constrained) and alternate parity
                        deg = len(adj[v])
                        if deg < min_deg or (deg == min_deg and parity[v] != min_parity):
                            min_var = v
                            min_parity = parity[v]
                            min_deg = deg

            if min_var == -1:
                return False

            dv = domains[min_var]
            bv = blocked[min_var]
            nbrs = adj[min_var]

            # Order colors: try colors that block fewer unassigned neighbours first
            # (least-constraining value heuristic)
            def color_constraint(c):
                count = 0
                for u in nbrs:
                    if assignment[u] == -1 and c in domains[u]:
                        count += 1
                return count

            ordered_colors = sorted(
                [c for c in dv if c not in bv],
                key=lambda c: color_constraint(c)
            )

            for c in ordered_colors:
                assignment[min_var] = c
                for u in nbrs:
                    blocked[u].add(c)
                    unassigned_count[u] -= 1

                if backtrack(assigned_count + 1):
                    return True

                for u in nbrs:
                    blocked[u].discard(c)
                    unassigned_count[u] += 1
                assignment[min_var] = -1

            return False

        if backtrack(0):
            return assignment
        return None

    def solve_with_restarts(n, edges, lists, deadline):
        """Try multiple strategies within the time budget."""
        # Strategy 1: Full backtracking with AC-3 + MRV + parity ordering
        result = solve_csp(n, edges, lists, deadline)
        if result is not None:
            return result

        # Strategy 2: Greedy with backtracking on conflicts
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Try degree-ordered greedy, then repair
        color = [-1] * n
        for v in sorted(range(n), key=lambda x: -len(adj[x])):
            taken = set()
            for u in adj[v]:
                if color[u] != -1:
                    taken.add(color[u])
            free = [c for c in lists[v] if c not in taken]
            color[v] = free[0] if free else lists[v][0]

        # Check if valid; if not, try to repair
        valid = True
        for u, v in edges:
            if color[u] == color[v]:
                valid = False
                break
        if valid and all(color[v] in lists[v] for v in range(n)):
            return color

        # Strategy 3: Local search - start from greedy, flip colors to fix conflicts
        if time.time() < deadline:
            color = [-1] * n
            for v in sorted(range(n), key=lambda x: -len(adj[x])):
                taken = set()
                for u in adj[v]:
                    if color[u] != -1:
                        taken.add(color[u])
                free = [c for c in lists[v] if c not in taken]
                color[v] = free[0] if free else lists[v][0]

            # Local search: repeatedly pick a vertex in a conflict and try to recolor
            for _ in range(10000):
                if time.time() > deadline:
                    break
                # Find a conflict
                conflict_found = False
                for u, v in edges:
                    if color[u] == color[v]:
                        conflict_found = True
                        # Try to recolor u
                        for c in lists[u]:
                            if c == color[u]:
                                continue
                            # Check if c conflicts with any neighbour of u
                            ok = True
                            for w in adj[u]:
                                if color[w] == c:
                                    ok = False
                                    break
                            if ok:
                                color[u] = c
                                break
                        if not conflict_found:
                            break
                if not conflict_found:
                    break

            # Final validation
            valid = True
            for u, v in edges:
                if color[u] == color[v]:
                    valid = False
                    break
            if valid and all(color[v] in lists[v] for v in range(n)):
                return color

        return None

    def build_result():
        params = ((20, 701), (40, 702), (60, 703), (100, 704))
        start_time = time.time()
        deadline = start_time + 110.0  # Safety margin before 120s limit
        results = []
        for splits, seed in params:
            n, edges, lists = instance(splits, seed)
            coloring = solve_with_restarts(n, edges, lists, deadline)
            if coloring is None:
                # Last-resort fallback: greedy by degree
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
