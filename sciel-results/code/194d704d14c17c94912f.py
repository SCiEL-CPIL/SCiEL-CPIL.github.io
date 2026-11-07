"""MODEL-AND-OPTIMIZE: CSP solver with MRV, Forward Checking, and AC-3 initialization (bitmask domains)."""
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

    # Precompute popcount for 5-bit masks (colors 0..4)
    _popcount = [0] * 32
    for _i in range(32):
        _popcount[_i] = bin(_i).count('1')

    def ac3(n, adj, domains):
        """AC-3 arc consistency preprocessing using bitmask domains."""
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))
        while queue:
            xi, xj = queue.popleft()
            di = domains[xi]
            dj = domains[xj]
            if di == 0:
                return False
            # Inequality constraint: c in di is supported iff exists d in dj with d != c
            # Fails only if dj has exactly one bit and it equals c
            if dj != 0 and (dj & (dj - 1)) == 0:
                only_mask = dj
                new_di = di & ~only_mask
                if new_di != di:
                    domains[xi] = new_di
                    if new_di == 0:
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

        # Convert lists to bitmask domains
        domains = [0] * n
        for i in range(n):
            mask = 0
            for c in lists[i]:
                mask |= (1 << c)
            domains[i] = mask
            if mask == 0:
                return None

        if not ac3(n, adj, domains):
            return None
        for i in range(n):
            if domains[i] == 0:
                return None

        assignment = [-1] * n
        blocked = [0] * n  # bitmask of blocked colors per vertex

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
                    valid = domains[v] & ~blocked[v]
                    cnt = _popcount[valid]
                    if cnt < min_size:
                        min_size = cnt
                        min_var = v
                        if min_size == 0:
                            return False
                        if min_size == 1:
                            break

            if min_var == -1:
                return False

            valid = domains[min_var] & ~blocked[min_var]
            nbrs = adj[min_var]
            # Iterate through each valid color
            while valid:
                bit = valid & (-valid)  # lowest set bit
                valid ^= bit
                c = bit.bit_length() - 1

                assignment[min_var] = c
                # Block this color on neighbors
                for u in nbrs:
                    blocked[u] |= bit

                if backtrack(assigned_count + 1):
                    return True

                # Undo
                for u in nbrs:
                    blocked[u] &= ~bit
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
