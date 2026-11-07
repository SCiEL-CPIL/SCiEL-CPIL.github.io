"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import time
from random import Random


def construct():
    sys.setrecursionlimit(100000)
    start = time.time()
    TOTAL_BUDGET = 145.0  # explicit guard, safety margin below the 150s kill

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

        # Initialize domains as bitmasks (integers) for speed
        # Color c corresponds to bit c (1 << c)
        domains = [0] * n
        for i in range(n):
            mask = 0
            for c in lists[i]:
                mask |= (1 << c)
            domains[i] = mask
        
        color = [-1] * n
        uncolored = list(range(n))
        deadline = time.time() + time_limit
        state = {'count': 0, 'ok': False}

        def is_deadline():
            return time.time() > deadline

        def pop_count(x):
            return bin(x).count("1")

        def forward_check(v, c_mask):
            """Assign colour c (bitmask) to v, prune neighbors' domains. Returns (success, undo_info)."""
            undo_info = []
            for u in adj[v]:
                if color[u] != -1:
                    continue
                # u is uncolored; prune c from domains[u]
                old_dom = domains[u]
                new_dom = old_dom & ~c_mask
                if new_dom == 0:
                    # Conflict: domain becomes empty
                    # Undo
                    for (uu, old_d) in undo_info:
                        domains[uu] = old_d
                    return False, None
                if new_dom != old_dom:
                    undo_info.append((u, old_dom))
                    domains[u] = new_dom
            return True, undo_info

        def backtrack():
            if not uncolored:
                state['ok'] = True
                return True

            state['count'] += 1
            if (state['count'] & 1023) == 0 and is_deadline():
                return False

            # Select variable with smallest domain (MRV), tie-break by degree
            best_idx = -1
            best_len = 999
            best_deg = -1
            for i, v in enumerate(uncolored):
                ln = pop_count(domains[v])
                if ln == 0:
                    return False
                if ln < best_len or (ln == best_len and degree[v] > best_deg):
                    best_len = ln
                    best_idx = i
                    best_deg = degree[v]
                    if ln == 1:
                        break

            if best_idx == -1:
                return False

            best_v = uncolored[best_idx]
            
            # Swap-pop to remove best_v from uncolored efficiently
            uncolored[best_idx] = uncolored[-1]
            uncolored.pop()

            # Get list of available values (colors) for best_v
            dom_mask = domains[best_v]
            values = []
            temp_mask = dom_mask
            while temp_mask:
                # Get lowest set bit
                lsb = temp_mask & -temp_mask
                c = lsb.bit_length() - 1
                values.append(c)
                temp_mask ^= lsb

            # Order values: prefer those that cause least constraint on neighbors
            # (least constraining value heuristic)
            def value_score(c):
                c_mask = 1 << c
                cnt = 0
                for u in adj[best_v]:
                    if color[u] == -1 and (domains[u] & c_mask):
                        cnt += 1
                return cnt

            values.sort(key=value_score)

            for c in values:
                c_mask = 1 << c
                ok, undo_info = forward_check(best_v, c_mask)
                if not ok:
                    continue

                color[best_v] = c

                if backtrack():
                    return True

                # Undo
                color[best_v] = -1
                for (u, old_d) in undo_info:
                    domains[u] = old_d

            # Restore best_v to uncolored
            uncolored.append(best_v)
            return False

        # Initial check: any empty domain?
        for i in range(n):
            if domains[i] == 0:
                return None

        if backtrack() and state['ok']:
            return [int(c) for c in color]
        return None

    def _greedy(n, adj, lists):
        order = sorted(range(n), key=lambda v: (-len(adj[v]), v))
        color = [-1] * n
        for v in order:
            taken = set()
            for u in adj[v]:
                if color[u] != -1:
                    taken.add(color[u])
            c = -1
            for cc in lists[v]:
                if cc not in taken:
                    c = cc
                    break
            if c == -1:
                for cc in lists[v]:
                    c = cc
                    break
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

        solved = False
        if remaining > 2.0:
            # Allocate time proportionally, but ensure at least 0.5s for small instances
            # Larger instances get more time.
            # Splits: 20, 40, 60, 100 -> n: 88, 168, 248, 408
            # We'll just give a fixed chunk, say 20s each if available, else scale.
            tl = max(0.5, min(25.0, remaining * 0.4))
            res = _solve(n, edges, lists, tl)
            if res is not None:
                out.append(res)
                solved = True

        if not solved:
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
