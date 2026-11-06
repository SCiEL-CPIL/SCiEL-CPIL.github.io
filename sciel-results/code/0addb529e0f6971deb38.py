"""Strong edge colouring: bitmask-accelerated greedy + local search + SA."""
import time
import math
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
        v2e = {}
        for i, (u, v) in enumerate(edges):
            v2e.setdefault(u, []).append(i)
            v2e.setdefault(v, []).append(i)

        conflict = [0] * m

        # 1) Share a vertex
        for u, lst in v2e.items():
            for a in range(len(lst)):
                for b in range(a + 1, len(lst)):
                    conflict[lst[a]] |= 1 << lst[b]
                    conflict[lst[b]] |= 1 << lst[a]

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

    def greedy_dsatur(edges, conflict):
        """DSATUR-style greedy: pick edge with most conflicts among uncolored, 
        then assign lowest feasible color. Returns (colour_list, num_colours)."""
        m = len(edges)
        colour = [-1] * m
        used = 0
        col_masks = []
        uncolored = set(range(m))
        
        # Precompute conflict degree for ordering
        conf_deg = [bin(conflict[e]).count('1') for e in range(m)]
        
        while uncolored:
            # Pick uncolored edge with highest conflict degree
            e = max(uncolored, key=lambda x: conf_deg[x])
            uncolored.discard(e)
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

    def local_search(edges, conflict, colour, used, rng, time_budget=3.0):
        """Local search with incremental class tracking: try edge moves, swaps, and class merges."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()

        # Build initial class structures
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

        maxc = max(classes.keys()) if classes else 0
        num_classes = len(classes)

        for _ in range(2000):
            if time.time() - start > time_budget:
                break
            improved = False

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
                        maxc = max(classes.keys()) if classes else 0
                        num_classes -= 1
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
                        maxc = max(classes.keys()) if classes else 0
                        num_classes -= 1
                    improved = True
                    break

            if improved:
                continue

            # 3) Try 1-swap
            if rng.random() < 0.7:
                for _ in range(20):
                    e = rng.randrange(m)
                    ce = col_of[e]
                    if ce <= 0:
                        continue

                    lower_colors = [d for d in range(ce) if d in classes and classes[d]]
                    if not lower_colors:
                        continue
                    d = rng.choice(lower_colors)

                    f = rng.choice(classes[d])

                    mask_d_without_f = class_masks[d] & ~(1 << f)
                    if conflict[e] & mask_d_without_f:
                        continue

                    mask_ce_without_e = class_masks[ce] & ~(1 << e)
                    if conflict[f] & mask_ce_without_e:
                        continue

                    col_of[e] = d
                    col_of[f] = ce

                    classes[d].remove(f)
                    classes[d].append(e)
                    class_masks[d] = (class_masks[d] & ~(1 << f)) | (1 << e)

                    classes[ce].remove(e)
                    classes[ce].append(f)
                    class_masks[ce] = (class_masks[ce] & ~(1 << e)) | (1 << f)

                    improved = True
                    break

            # 4) Try 2-swap (move two edges between classes to enable merge)
            if not improved and rng.random() < 0.3 and len(classes) >= 2:
                cls_list = list(classes.keys())
                if len(cls_list) >= 2:
                    a, b = rng.sample(cls_list, 2)
                    if len(classes[a]) >= 1 and len(classes[b]) >= 1:
                        e = rng.choice(classes[a])
                        f = rng.choice(classes[b])
                        
                        mask_b_without_f = class_masks[b] & ~(1 << f)
                        if conflict[e] & mask_b_without_f:
                            continue
                        mask_a_without_e = class_masks[a] & ~(1 << e)
                        if conflict[f] & mask_a_without_e:
                            continue
                        
                        col_of[e] = b
                        col_of[f] = a
                        classes[b].remove(f)
                        classes[b].append(e)
                        class_masks[b] = (class_masks[b] & ~(1 << f)) | (1 << e)
                        classes[a].remove(e)
                        classes[a].append(f)
                        class_masks[a] = (class_masks[a] & ~(1 << e)) | (1 << f)
                        improved = True

            if not improved:
                break

        # Renumber compactly
        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)

    def fast_local_search(edges, conflict, colour, used, rng, time_budget=1.0):
        """Fast local search: only merge and single-edge moves, no swaps."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()

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

        maxc = max(classes.keys()) if classes else 0

        for _ in range(500):
            if time.time() - start > time_budget:
                break
            improved = False

            # Merge
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
                        maxc = max(classes.keys()) if classes else 0
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # Single edge move
            for _ in range(5):
                e = rng.randrange(m)
                ce = col_of[e]
                if ce <= 0:
                    continue
                for d in range(ce):
                    if d not in classes:
                        continue
                    if conflict[e] & class_masks[d] == 0:
                        col_of[e] = d
                        classes[d].append(e)
                        class_masks[d] |= 1 << e
                        classes[ce].remove(e)
                        class_masks[ce] &= ~(1 << e)
                        if not classes[ce]:
                            del classes[ce]
                            del class_masks[ce]
                            maxc = max(classes.keys()) if classes else 0
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)

    def simulated_annealing(edges, conflict, colour, used, rng, time_budget=5.0):
        """SA with bitmask-based validity checks."""
        m = len(edges)
        best_colour = colour[:]
        best_used = used
        cur_colour = colour[:]
        cur_used = used
        start = time.time()
        T = 10.0

        # Precompute class masks for current state
        def build_class_masks(col):
            cm = {}
            for e in range(m):
                c = col[e]
                if c not in cm:
                    cm[c] = 0
                cm[c] |= 1 << e
            return cm

        cur_class_masks = build_class_masks(cur_colour)

        while time.time() - start < time_budget:
            if rng.random() < 0.5:
                e = rng.randrange(m)
                old_c = cur_colour[e]
                new_c = rng.randrange(used + 1)
                if new_c == old_c:
                    continue
                
                # Check validity using class mask
                if new_c in cur_class_masks:
                    if cur_class_masks[new_c] & conflict[e]:
                        continue
                
                cur_colour[e] = new_c
                cur_class_masks[new_c] = cur_class_masks.get(new_c, 0) | (1 << e)
                cur_class_masks[old_c] &= ~(1 << e)
                if cur_class_masks[old_c] == 0:
                    del cur_class_masks[old_c]
            else:
                e1 = rng.randrange(m)
                e2 = rng.randrange(m)
                if e1 == e2:
                    continue
                c1 = cur_colour[e1]
                c2 = cur_colour[e2]
                if c1 == c2:
                    continue

                # Check if e1 can go to c2
                mask_c2_without_e2 = cur_class_masks[c2] & ~(1 << e2)
                if mask_c2_without_e2 & conflict[e1]:
                    continue
                # Check if e2 can go to c1
                mask_c1_without_e1 = cur_class_masks[c1] & ~(1 << e1)
                if mask_c1_without_e1 & conflict[e2]:
                    continue

                cur_colour[e1] = c2
                cur_colour[e2] = c1
                cur_class_masks[c2] = (cur_class_masks[c2] & ~(1 << e2)) | (1 << e1)
                cur_class_masks[c1] = (cur_class_masks[c1] & ~(1 << e1)) | (1 << e2)

            cur_used = len(cur_class_masks)

            if cur_used < best_used:
                best_colour = cur_colour[:]
                best_used = cur_used

            # Quick local search
            cur_colour, cur_used = fast_local_search(edges, conflict, cur_colour, cur_used, rng, time_budget=0.3)
            cur_class_masks = build_class_masks(cur_colour)

            if cur_used < best_used:
                best_colour = cur_colour[:]
                best_used = cur_used

            T *= 0.93

        return best_colour, best_used

    def two_swap_search(edges, conflict, colour, used, rng, time_budget=3.0):
        """Try 2-swap moves to enable merges."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()

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

        maxc = max(classes.keys()) if classes else 0

        for _ in range(800):
            if time.time() - start > time_budget:
                break
            improved = False

            if len(classes) < 2:
                break

            cls_list = list(classes.keys())
            a, b = rng.sample(cls_list, 2)

            if not classes[a] or not classes[b]:
                continue
            e = rng.choice(classes[a])
            f = rng.choice(classes[b])

            mask_b_without_f = class_masks[b] & ~(1 << f)
            if conflict[e] & mask_b_without_f:
                continue

            mask_a_without_e = class_masks[a] & ~(1 << e)
            if conflict[f] & mask_a_without_e:
                continue

            col_of[e] = b
            col_of[f] = a

            classes[b].remove(f)
            classes[b].append(e)
            class_masks[b] = (class_masks[b] & ~(1 << f)) | (1 << e)

            classes[a].remove(e)
            classes[a].append(f)
            class_masks[a] = (class_masks[a] & ~(1 << e)) | (1 << f)

            improved = True

            # Try to merge after swap
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
                    for edge_c in cls_c:
                        if conflict[edge_c] & mask_d:
                            ok = False
                            break

                    if ok:
                        for edge_c in cls_c:
                            col_of[edge_c] = d
                        classes[d].extend(cls_c)
                        class_masks[d] |= mask_c
                        del classes[c]
                        del class_masks[c]
                        maxc = max(classes.keys()) if classes else 0
                        improved = True
                        break
                if improved:
                    break

        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)

    def final_polish(edges, conflict, colour, used, rng, time_budget=5.0):
        """Final high-precision polish: aggressive merge attempts with 2-swap."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()

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

        maxc = max(classes.keys()) if classes else 0
        num_classes = len(classes)

        for _ in range(3000):
            if time.time() - start > time_budget:
                break
            improved = False

            # Try all pairs of classes for merge
            cls_list = sorted(classes.keys(), reverse=True)
            for i, c in enumerate(cls_list):
                if c not in classes:
                    continue
                cls_c = classes[c]
                mask_c = class_masks[c]
                for d in cls_list[i+1:]:
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
                        maxc = max(classes.keys()) if classes else 0
                        num_classes -= 1
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # 2-swap to enable future merges
            if len(classes) >= 2:
                cls_list = list(classes.keys())
                a, b = rng.sample(cls_list, 2)
                if classes[a] and classes[b]:
                    e = rng.choice(classes[a])
                    f = rng.choice(classes[b])
                    mask_b_without_f = class_masks[b] & ~(1 << f)
                    if not (conflict[e] & mask_b_without_f):
                        mask_a_without_e = class_masks[a] & ~(1 << e)
                        if not (conflict[f] & mask_a_without_e):
                            col_of[e] = b
                            col_of[f] = a
                            classes[b].remove(f)
                            classes[b].append(e)
                            class_masks[b] = (class_masks[b] & ~(1 << f)) | (1 << e)
                            classes[a].remove(e)
                            classes[a].append(f)
                            class_masks[a] = (class_masks[a] & ~(1 << e)) | (1 << f)
                            improved = True

            if not improved:
                break

        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = gen_instance(*spec)
        conflict = build_conflict_bitmasks(edges)
        m = len(edges)
        best_colour = None
        best_used = m
        rng = Random(42)

        # DSATUR greedy first
        colour, used = greedy_dsatur(edges, conflict)
        if used < best_used:
            best_colour = colour[:]
            best_used = used

        # Multiple randomized restarts with greedy
        n_restarts = 150
        for r in range(n_restarts):
            if time.time() > DEADLINE - 15.0:
                break
            order = list(range(m))
            rng.shuffle(order)
            colour, used = greedy_first_fit(edges, conflict, order)
            if used < best_used:
                best_colour = colour[:]
                best_used = used

        # Local compression on best
        if best_colour is not None:
            best_colour, best_used = local_search(edges, conflict, best_colour, best_used, rng, time_budget=8.0)

        # Two-swap search for further improvement
        if time.time() < DEADLINE - 25.0 and best_colour is not None:
            best_colour, best_used = two_swap_search(edges, conflict, best_colour, best_used, rng, time_budget=8.0)

        # Simulated annealing for further improvement
        if time.time() < DEADLINE - 25.0 and best_colour is not None:
            best_colour, best_used = simulated_annealing(edges, conflict, best_colour, best_used, rng, time_budget=10.0)

        # Final polish
        if time.time() < DEADLINE - 15.0 and best_colour is not None:
            best_colour, best_used = final_polish(edges, conflict, best_colour, best_used, rng, time_budget=5.0)

        # More targeted restarts if time permits
        if time.time() < DEADLINE - 20.0:
            for r in range(80):
                if time.time() > DEADLINE - 15.0:
                    break
                order = list(range(m))
                rng.shuffle(order)
                colour, used = greedy_first_fit(edges, conflict, order)
                if used <= best_used:
                    colour, used = fast_local_search(edges, conflict, colour, used, rng, time_budget=1.0)
                    if used < best_used:
                        best_colour = colour[:]
                        best_used = used

        if best_colour is None:
            best_colour, best_used = greedy_first_fit(edges, conflict, list(range(m)))

        out.append(best_colour)

    return out
