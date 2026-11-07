"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
from random import Random


def construct():
    sys.setrecursionlimit(100000)
    start = time.time()
    TOTAL_BUDGET = 140.0  # explicit guard, safety margin below the 150s kill

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

    def _solve(n, edges, lists, time_limit):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        degree = [len(adj[i]) for i in range(n)]

        # Initialize domains as bitmasks for faster manipulation
        # Bit i set means color i is available
        domains = [0] * n
        for i, l in enumerate(lists):
            mask = 0
            for c in l:
                mask |= (1 << c)
            domains[i] = mask
        
        color = [-1] * n
        uncolored_arr = [1] * n  # 1 if uncolored, 0 if colored
        deadline = time.time() + time_limit
        state = {'count': 0, 'ok': False}
        remaining_count = n

        def forward_check(v, c, domains, uncolored_arr, adj, trail):
            """Assign color c to v, propagate constraints to uncolored neighbors.
            Returns True on success, False on failure. trail is list of (neighbor, old_domain) for undo."""
            mask_c = 1 << c
            for u in adj[v]:
                if uncolored_arr[u]:
                    old_dom = domains[u]
                    if old_dom & mask_c:
                        new_dom = old_dom & ~mask_c
                        domains[u] = new_dom
                        trail.append((u, old_dom))
                        if new_dom == 0:
                            return False
            return True

        def undo(trail, domains):
            for u, old_dom in trail:
                domains[u] = old_dom

        def backtrack():
            nonlocal remaining_count
            if remaining_count == 0:
                state['ok'] = True
                return True

            state['count'] += 1
            if (state['count'] & 255) == 0 and time.time() > deadline:
                return False

            # MRV: pick uncolored vertex with fewest domain options
            best_v = -1
            best_len = 999
            best_deg = -1
            for v in range(n):
                if uncolored_arr[v]:
                    ln = domains[v].bit_count()
                    if ln == 0:
                        return False
                    if ln < best_len or (ln == best_len and degree[v] > best_deg):
                        best_len = ln
                        best_v = v
                        best_deg = degree[v]
                        if ln == 1:
                            break

            uncolored_arr[best_v] = 0
            remaining_count -= 1
            
            # Iterate through available colors in best_v's domain
            mask = domains[best_v]
            while mask:
                # Extract lowest set bit
                lsb = mask & -mask
                c = lsb.bit_length() - 1
                mask ^= lsb
                
                color[best_v] = c
                trail = []
                if forward_check(best_v, c, domains, uncolored_arr, adj, trail):
                    if backtrack():
                        return True
                    undo(trail, domains)
                color[best_v] = -1

            uncolored_arr[best_v] = 1
            remaining_count += 1
            return False

        if backtrack() and state['ok']:
            return [int(c) for c in color]
        return None

    def _greedy(n, adj, lists):
        # Use a smarter greedy: process in order of decreasing degree,
        # and pick the color that minimizes conflicts with uncolored neighbors
        order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
        color = [-1] * n
        for v in order:
            taken = set()
            for u in adj[v]:
                if color[u] != -1:
                    taken.add(color[u])
            # Prefer colors not in taken; if all are taken, pick first available
            c = -1
            for cc in lists[v]:
                if cc not in taken:
                    c = cc
                    break
            if c == -1:
                # All colors in list conflict; pick the one with fewest future conflicts
                best_conflicts = 999
                for cc in lists[v]:
                    conflicts = 0
                    for u in adj[v]:
                        if color[u] == -1 and cc in lists[u]:
                            conflicts += 1
                    if conflicts < best_conflicts:
                        best_conflicts = conflicts
                        c = cc
            color[v] = c
        return [int(c) for c in color]

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        remaining = TOTAL_BUDGET - (time.time() - start)
        n, edges, lists = instance(splits, seed)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        if remaining > 2.0:
            tl = max(0.2, min(30.0, remaining * 0.7))
            res = _solve(n, edges, lists, tl)
            if res is not None:
                out.append(res)
                continue

        # Fallback: best-effort greedy (may score 0 for a broken instance, but is structurally valid).
        out.append(_greedy(n, adj, lists))

    # Hardening: ensure top-level is a list of 4 lists of Python ints with correct lengths.
    # (A wrong length/type makes the whole result invalid, so guard it.)
    lengths = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, _, _ = instance(splits, seed)
        lengths.append(n)
    for i in range(len(out)):
        if not isinstance(out[i], list):
            out[i] = list(out[i])
        if len(out[i]) != lengths[i]:
            out[i] = (out[i] + [0] * lengths[i])[:lengths[i]]
        out[i] = [int(x) for x in out[i]]
    return out
# EVOLVE-BLOCK-END
