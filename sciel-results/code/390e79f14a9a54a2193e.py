"""Strong edge colouring via bitmask conflict representation, DSATUR greedy,
and local search with merges, single-edge moves, and 2-swaps.

Each colour class must be an induced matching. Two edges conflict iff they share
a vertex OR are joined by an edge. We build a bitmask conflict graph once, then
run DSATUR-style greedy restarts (deterministic + randomized) and a local search
that merges compatible colour classes, moves single edges to lower colours, and
performs 2-swaps to enable further merges. A hard deadline guard returns the best
found before the 150 s kill.
"""
import time
from random import Random


def construct():
    t0 = time.time()
    DEADLINE = t0 + 145.0  # safety margin under the 150 s kill

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

    def build_conflict_bitmasks(edges):
        """Build conflict bitmask for each edge using Python ints as bitsets."""
        m = len(edges)
        # vertex -> list of edge indices incident
        v2e = {}
        for i, (u, v) in enumerate(edges):
            v2e.setdefault(u, []).append(i)
            v2e.setdefault(v, []).append(i)

        # edge_set for quick lookup: (u,v) -> edge index
        edge_idx = {}
        for i, (u, v) in enumerate(edges):
            edge_idx[(u, v)] = i
            edge_idx[(v, u)] = i

        conflict = [0] * m  # bitmask of conflicting edges

        # 1) Share a vertex
        for u, lst in v2e.items():
            for a in range(len(lst)):
                ea = lst[a]
                for b in range(a + 1, len(lst)):
                    eb = lst[b]
                    conflict[ea] |= 1 << eb
                    conflict[eb] |= 1 << ea

        # 2) Joined by an edge
        for i, (u, v) in enumerate(edges):
            lu = v2e.get(u, [])
            lv = v2e.get(v, [])
            for a in lu:
                if a == i:
                    continue
                for b in lv:
                    if b == i or b == a:
                        continue
                    conflict[a] |= 1 << b
                    conflict[b] |= 1 << a

        return conflict

    def dsatur_greedy(edges, conflict, rng, time_budget=None):
        """DSATUR-style greedy: pick the most-constrained uncoloured edge,
        assign the smallest feasible colour. Returns (colour_list, num_colours)."""
        m = len(edges)
        colour = [-1] * m
        col_masks = []  # bitmask of edges in each colour
        used = 0
        uncoloured = (1 << m) - 1

        # Precompute degree in conflict graph for tie-breaking
        conf_deg = [bin(c).count('1') for c in conflict]

        while uncoloured and (time_budget is None or time.time() - t0 < DEADLINE):
            # Pick edge with highest saturation (distinct neighbour colours),
            # tie-break by conflict degree
            best_e = -1
            best_sat = -1
            best_deg = -1
            # Iterate over uncoloured edges
            e = 0
            mask = uncoloured
            while mask:
                if mask & 1:
                    if colour[e] == -1:
                        # Count distinct colours among coloured neighbours
                        sat = 0
                        cm = conflict[e] & ~uncoloured  # coloured neighbours
                        # Count distinct colour classes: for each colour, check if any neighbour has it
                        for c in range(used):
                            if col_masks[c] & cm:
                                sat += 1
                        if sat > best_sat or (sat == best_sat and conf_deg[e] > best_deg):
                            best_sat = sat
                            best_deg = conf_deg[e]
                            best_e = e
                mask >>= 1
                e += 1

            if best_e == -1:
                break

            ce = 1 << best_e
            uncoloured &= ~ce
            placed = False
            for c in range(used):
                if col_masks[c] & conflict[best_e] == 0:
                    colour[best_e] = c
                    col_masks[c] |= ce
                    placed = True
                    break
            if not placed:
                colour[best_e] = used
                col_masks.append(ce)
                used += 1

        return colour, used

    def greedy_first_fit(edges, conflict, order):
        """Greedy first-fit using conflict bitmasks. Returns (colour_list, num_colours)."""
        m = len(edges)
        colour = [-1] * m
        used = 0
        col_masks = []

        for e in order:
            ce = 1 << e
            placed = False
            for c in range(used):
                if col_masks[c] & conflict[e] == 0:
                    colour[e] = c
                    col_masks[c] |= ce
                    placed = True
                    break
            if not placed:
                colour[e] = used
                col_masks.append(ce)
                used += 1
        return colour, used

    def local_search(edges, conflict, colour, used, rng, time_budget=5.0):
        """Local search: merges, single-edge moves, and 2-swaps.
        Returns (colour, used)."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()

        for _ in range(2000):
            if time.time() - start > time_budget:
                break
            improved = False

            # Build colour classes and their bitmasks
            classes = {}
            class_masks = {}
            for e in range(m):
                c = col_of[e]
                if c >= 0:
                    if c not in classes:
                        classes[c] = []
                        class_masks[c] = 0
                    classes[c].append(e)
                    class_masks[c] |= 1 << e

            if not classes:
                break

            maxc = max(classes.keys())

            # 1) Try merging colour classes (highest into lower)
            for c in range(maxc, -1, -1):
                if c not in classes:
                    continue
                cls_c = classes[c]
                mask_c = class_masks[c]

                for d in range(c):
                    if d not in classes:
                        continue
                    mask_d = class_masks[d]

                    ok = True
                    for e in cls_c:
                        if conflict[e] & mask_d:
                            ok = False
                            break

                    if ok:
                        for e in cls_c:
                            col_of[e] = d
                        classes[d].extend(cls_c)
                        class_masks[d] |= mask_c
                        del classes[c]
                        del class_masks[c]
                        improved = True
                        break

                if improved:
                    break

            if improved:
                continue

            # 2) Try moving a single edge to a lower colour
            edges_list = list(range(m))
            rng.shuffle(edges_list)

            for e in edges_list:
                ce = col_of[e]
                if ce <= 0:
                    continue

                best_d = None
                for d in range(ce):
                    if d not in classes:
                        continue
                    mask_d = class_masks[d]
                    if conflict[e] & mask_d == 0:
                        best_d = d
                        break

                if best_d is not None:
                    col_of[e] = best_d
                    classes[best_d].append(e)
                    class_masks[best_d] |= 1 << e
                    classes[ce].remove(e)
                    class_masks[ce] &= ~(1 << e)
                    if not classes[ce]:
                        del classes[ce]
                        del class_masks[ce]
                    improved = True
                    break

            if improved:
                continue

            # 3) Try 2-swaps: move e to colour d, move f from d to colour ce
            for _ in range(20):
                e = rng.randrange(m)
                ce = col_of[e]
                if ce <= 0:
                    continue

                lower_colors = [d for d in range(ce) if d in classes and classes[d]]
                if not lower_colors:
                    continue
                d = rng.choice(lower_colors)

                if not classes[d]:
                    continue
                f = rng.choice(classes[d])

                # Check if e can go to d (excluding f)
                mask_d_without_f = class_masks[d] & ~(1 << f)
                if conflict[e] & mask_d_without_f:
                    continue

                # Check if f can go to ce (excluding e)
                mask_ce_without_e = class_masks[ce] & ~(1 << e)
                if conflict[f] & mask_ce_without_e:
                    continue

                # Perform swap
                col_of[e] = d
                col_of[f] = ce
                classes[d].append(e)
                class_masks[d] |= 1 << e
                classes[ce].append(f)
                class_masks[ce] |= 1 << f
                classes[d].remove(f)
                class_masks[d] &= ~(1 << f)
                classes[ce].remove(e)
                class_masks[ce] &= ~(1 << e)
                improved = True
                break

            if not improved:
                # No improvement in this iteration; break to avoid infinite loop
                break

        # Reindex to contiguous 0..k-1
        present = sorted(set(col_of))
        remap = {c: i for i, c in enumerate(present)}
        for i in range(m):
            col_of[i] = remap[col_of[i]]
        used = len(present)
        return col_of, used

    results = []
    for idx, (n, d, seed) in enumerate(((40, 4, 101), (60, 5, 102), (80, 6, 103))):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            results.append([])
            continue

        conflict = build_conflict_bitmasks(edges)
        cdeg = [bin(c).count('1') for c in conflict]
        order_base = list(range(m))
        deg_order = sorted(order_base, key=lambda i: -cdeg[i])

        best = None
        best_k = m + 1

        def consider(col, used):
            nonlocal best, best_k
            if used < best_k:
                best_k = used
                best = col

        # DSATUR greedy (deterministic)
        col, used = dsatur_greedy(edges, conflict, Random(0))
        consider(col, used)

        # Deterministic first-fit orders
        col, used = greedy_first_fit(edges, conflict, order_base)
        consider(col, used)
        col, used = greedy_first_fit(edges, conflict, deg_order)
        consider(col, used)
        col, used = greedy_first_fit(edges, conflict, sorted(order_base, key=lambda i: cdeg[i]))
        consider(col, used)

        # Randomized DSATUR restarts
        rng = Random(999 + seed)
        remaining = 3 - idx
        restart_deadline = DEADLINE - 25.0 * remaining
        while time.time() < restart_deadline:
            col, used = dsatur_greedy(edges, conflict, rng)
            consider(col, used)

        # Local search on the best
        if best is not None:
            ls_deadline = DEADLINE - 25.0 * (remaining - 1)
            time_budget = max(1.0, min(10.0, ls_deadline - time.time()))
            best, _ = local_search(edges, conflict, best, best_k, rng, time_budget)
            # Reindex
            present = sorted(set(best))
            remap = {c: i for i, c in enumerate(present)}
            for i in range(m):
                best[i] = remap[best[i]]

        results.append(best)

    return results
