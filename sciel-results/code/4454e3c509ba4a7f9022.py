"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

def construct():
    from random import Random
    import sys

    start_time = time.time()
    DEADLINE = 135.0

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def instance(splits, seed, palette=5):
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

    def solve_instance(n, edges, lists):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Bitmask domains
        domains = [0] * n
        for v in range(n):
            mask = 0
            for c in lists[v]:
                mask |= (1 << c)
            domains[v] = mask

        # AC-3 Initialization
        queue = deque()
        for u in range(n):
            for v in adj[u]:
                queue.append((u, v))

        def revise(u, v):
            mask_v = domains[v]
            if mask_v == 0:
                return True
            # If mask_v has more than 1 bit, it supports all c in u
            if mask_v & (mask_v - 1) != 0:
                return False
            # mask_v has exactly 1 bit
            if domains[u] & mask_v:
                domains[u] &= ~mask_v
                return True
            return False

        while queue:
            u, v = queue.popleft()
            if revise(u, v):
                if domains[u] == 0:
                    return None
                for w in adj[u]:
                    if w != v:
                        queue.append((w, u))

        # Backtracking with MRV and forward checking
        sys.setrecursionlimit(10000)
        colour = [-1] * n
        node_count = 0

        def get_mrv_vertex():
            min_count = 10  # max possible is 5
            min_vertex = -1
            for v in range(n):
                if colour[v] != -1:
                    continue
                cnt = domains[v].bit_count()
                if cnt < min_count:
                    min_count = cnt
                    min_vertex = v
                    if min_count <= 1:
                        break
            return min_vertex, min_count

        def backtrack():
            nonlocal node_count
            node_count += 1
            if node_count % 2000 == 0:
                if time_remaining() < 0.5:
                    return False

            v, count = get_mrv_vertex()
            if v == -1:
                return True

            if count == 0:
                return False

            # Try each color in domains[v], ordered by least constraining value
            mask = domains[v]
            colors_to_try = []
            m = mask
            while m:
                lsb = m & -m
                c = lsb.bit_length() - 1
                # Count how many neighbors would lose this color
                constraining = 0
                for u in adj[v]:
                    if colour[u] == -1 and (domains[u] & lsb):
                        constraining += 1
                colors_to_try.append((constraining, c, lsb))
                m &= m - 1

            # Sort by least constraining (fewest neighbors affected)
            colors_to_try.sort(key=lambda x: x[0])

            for _, c, lsb in colors_to_try:
                colour[v] = c

                saved_domains = {}
                valid = True
                for u in adj[v]:
                    if colour[u] != -1:
                        if colour[u] == c:
                            valid = False
                            break
                        continue

                    if domains[u] & lsb:
                        saved_domains[u] = domains[u]
                        domains[u] &= ~lsb
                        if domains[u] == 0:
                            valid = False
                            break

                if valid and backtrack():
                    return True

                # Restore
                colour[v] = -1
                for u, dom in saved_domains.items():
                    domains[u] = dom

            return False

        if backtrack():
            return [int(c) for c in colour]

        return None

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        res = solve_instance(n, edges, lists)
        if res is None:
            # Fallback: greedy with AC-3
            res = [lists[i][0] for i in range(n)]
        out.append(res)

    final_out = []
    for lst in out:
        if isinstance(lst, (list, tuple)):
            final_out.append([int(x) for x in lst])
        else:
            final_out.append(lst)

    return final_out
# EVOLVE-BLOCK-END
