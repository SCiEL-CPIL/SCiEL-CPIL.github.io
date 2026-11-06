"""Strong edge colouring via DSATUR greedy on the conflict graph + incremental
local search with class merging.  Structural change: DSATUR dynamically selects
the most-constrained vertex first, producing tighter initial colourings that
reduce the number of colours before local-search refinement.
"""
import time
import heapq
from random import Random
import sys
from collections import defaultdict


def construct():
    t0 = time.time()
    DEADLINE = t0 + 115.0

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict(edges):
        m = len(edges)
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        conf = [set() for _ in range(m)]
        v_to_e = {}
        for i, (u, v) in enumerate(edges):
            v_to_e.setdefault(u, []).append(i)
            v_to_e.setdefault(v, []).append(i)
        for i in range(m):
            u, v = edges[i]
            near = {u, v} | (adj.get(u, set()) | adj.get(v, set()))
            for x in near:
                for j in v_to_e.get(x, []):
                    if j != i and j > i:
                        conf[i].add(j)
                        conf[j].add(i)
        conf_list = [sorted(c) for c in conf]
        return conf_list

    def dsatur_greedy(conf, m, rng=None):
        """DSATUR: pick vertex with max saturation (distinct neighbor colors),
        tie-break by degree in conflict graph."""
        col = [-1] * m
        # degree in conflict graph
        cdeg = [len(conf[i]) for i in range(m)]
        # For each vertex, track set of colors used by colored neighbors
        neighbor_colors = [set() for _ in range(m)]
        colored = [False] * m
        num_colored = 0
        # Use a heap: (-saturation, -degree, vertex)
        heap = [(-0, -cdeg[i], i) for i in range(m)]
        heapq.heapify(heap)

        for _ in range(m):
            # Pop until we find an uncolored vertex
            while heap:
                neg_sat, neg_deg, v = heapq.heappop(heap)
                if not colored[v]:
                    break
            else:
                break

            colored[v] = True
            num_colored += 1
            sat = -neg_sat

            # Find the smallest available color
            used = neighbor_colors[v]
            c = 0
            while c in used:
                c += 1
            col[v] = c

            # Update neighbors
            for nb in conf[v]:
                if not colored[nb]:
                    neighbor_colors[nb].add(c)
                    new_sat = len(neighbor_colors[nb])
                    heapq.heappush(heap, (-new_sat, -cdeg[nb], nb))

        return col

    def k_of(col):
        return max(col) + 1 if col else 0

    def reindex(col):
        present = sorted(set(col))
        if not present:
            return col
        remap = {c: i for i, c in enumerate(present)}
        for i in range(len(col)):
            col[i] = remap[col[i]]
        return col

    def is_valid_move(col, e, new_c, conf):
        for j in conf[e]:
            if col[j] == new_c:
                return False
        return True

    def local_improve_fast(col, conf, m, deadline, rng):
        """Local search with incremental class tracking."""
        col = list(col)
        # Build class structure: list of sets, indexed by color
        cls = defaultdict(set)
        for i, c in enumerate(col):
            cls[c].add(i)
        current_k = max(cls.keys()) + 1 if cls else 0

        def try_merge_all():
            nonlocal current_k
            merged = True
            while merged and time.time() < deadline:
                merged = False
                cols = sorted(cls.keys())
                for idx in range(len(cols)):
                    c1 = cols[idx]
                    if c1 not in cls:
                        continue
                    s1 = cls[c1]
                    for j in range(idx + 1, len(cols)):
                        c2 = cols[j]
                        if c2 not in cls:
                            continue
                        s2 = cls[c2]
                        ok = True
                        for e in s2:
                            for nb in conf[e]:
                                if nb in s1:
                                    ok = False
                                    break
                            if not ok:
                                break
                        if ok:
                            for e in s2:
                                col[e] = c1
                            s1.update(s2)
                            del cls[c2]
                            current_k -= 1
                            merged = True
                            break
                    if merged:
                        break

        def try_single_moves():
            nonlocal current_k
            moved = True
            while moved and time.time() < deadline:
                moved = False
                cols = sorted(cls.keys())
                if len(cols) < 2:
                    break
                for _ in range(5000):
                    e = rng.randrange(m)
                    c = col[e]
                    for new_c in range(c):
                        if new_c not in cls:
                            continue
                        if is_valid_move(col, e, new_c, conf):
                            col[e] = new_c
                            cls[c].discard(e)
                            if not cls[c]:
                                del cls[c]
                                if c == current_k - 1:
                                    current_k -= 1
                            cls[new_c].add(e)
                            moved = True
                            break
                    if moved:
                        break

        def try_swaps():
            nonlocal current_k
            swapped = True
            while swapped and time.time() < deadline:
                swapped = False
                cols = sorted(cls.keys())
                if len(cols) < 2:
                    break
                for _ in range(3000):
                    e1 = rng.randrange(m)
                    e2 = rng.randrange(m)
                    if e1 == e2:
                        continue
                    c1, c2 = col[e1], col[e2]
                    if c1 == c2:
                        continue
                    # Check swap validity
                    valid = True
                    for j in conf[e1]:
                        if j != e2 and col[j] == c2:
                            valid = False
                            break
                    if valid:
                        for j in conf[e2]:
                            if j != e1 and col[j] == c1:
                                valid = False
                                break
                    if valid:
                        col[e1], col[e2] = c2, c1
                        cls[c1].remove(e1)
                        cls[c1].add(e2)
                        cls[c2].remove(e2)
                        cls[c2].add(e1)
                        swapped = True
                        break

        # Iteratively apply all move types
        improved = True
        guard = 0
        while improved and time.time() < deadline and guard < 200:
            improved = False
            guard += 1
            before_k = current_k
            try_merge_all()
            if current_k < before_k:
                improved = True
            before_k = current_k
            try_single_moves()
            if current_k < before_k:
                improved = True
            before_k = current_k
            try_swaps()
            if current_k < before_k:
                improved = True
            # After swaps, try merging again
            before_k = current_k
            try_merge_all()
            if current_k < before_k:
                improved = True

        return reindex(col)

    def sa_with_classes(col, conf, m, deadline, rng):
        """Simulated annealing with incremental class tracking."""
        col = list(col)
        best = list(col)
        best_k = max(col) + 1 if col else 0
        current = list(col)
        cls = defaultdict(set)
        for i, c in enumerate(current):
            cls[c].add(i)
        current_k = max(cls.keys()) + 1 if cls else 0
        no_improve = 0
        cdeg = [len(conf[i]) for i in range(m)]

        for it in range(500000):
            if time.time() > deadline:
                break
            if no_improve > 5000:
                break
            if len(cls) < 2:
                break

            e = rng.randrange(m)
            c = current[e]
            cols_list = list(cls.keys())
            if len(cols_list) < 2:
                break
            new_c = rng.choice([x for x in cols_list if x != c])

            if is_valid_move(current, e, new_c, conf):
                cls[c].discard(e)
                if not cls[c]:
                    del cls[c]
                    if c == current_k - 1:
                        current_k -= 1
                cls[new_c].add(e)
                current[e] = new_c

                if current_k < best_k:
                    best_k = current_k
                    best = list(current)
                    no_improve = 0
                elif current_k < current_k:
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1

        return reindex(best)

    results = []
    for idx, (n, d, seed) in enumerate(((40, 4, 101), (60, 5, 102), (80, 6, 103))):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            results.append([])
            continue
        conf = build_conflict(edges)
        cdeg = [len(c) for c in conf]
        order_base = list(range(m))
        deg_order = sorted(order_base, key=lambda i: -cdeg[i])

        best = None
        best_k = m + 1

        def consider(col):
            nonlocal best, best_k
            k = k_of(col)
            if k < best_k:
                best_k = k
                best = list(col)

        rng = Random(999 + seed)
        remaining = 3 - idx

        # DSATUR greedy (deterministic)
        consider(dsatur_greedy(conf, m))

        # Degree-ordered greedy as backup
        col_deg = [-1] * m
        for i in deg_order:
            taken = set()
            for j in conf[i]:
                c = col_deg[j]
                if c >= 0:
                    taken.add(c)
            c = 0
            while c in taken:
                c += 1
            col_deg[i] = c
        consider(col_deg)

        # Randomized DSATUR-like restarts: perturb order with random tie-breaking
        restart_deadline = DEADLINE - 25.0 * remaining
        while time.time() < restart_deadline:
            # Randomized greedy with degree-based priority + random tie-breaking
            col_r = [-1] * m
            uncolored = list(range(m))
            rng.shuffle(uncolored)
            # Sort by degree descending, but with random perturbation
            order = sorted(uncolored, key=lambda i: (-cdeg[i], rng.random()))
            for i in order:
                taken = set()
                for j in conf[i]:
                    c = col_r[j]
                    if c >= 0:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                col_r[i] = c
            consider(col_r)

        # Local improvement on best
        ls_deadline = min(DEADLINE - 15.0 * (remaining - 1), restart_deadline + 20.0)
        if best is not None and time.time() < ls_deadline:
            best = local_improve_fast(best, conf, m, ls_deadline, rng)
            consider(best)

        # SA for further refinement
        sa_deadline = DEADLINE - 10.0 * (remaining - 1)
        if best is not None and time.time() < sa_deadline:
            sa_res = sa_with_classes(best, conf, m, sa_deadline, rng)
            consider(sa_res)
            # Final local polish
            if best is not None and time.time() < sa_deadline:
                final_dl = DEADLINE - 5.0 * (remaining - 1)
                best = local_improve_fast(best, conf, m, final_dl, rng)
                consider(best)

        if best is None:
            best = dsatur_greedy(conf, m)
        results.append(best)
    return results
