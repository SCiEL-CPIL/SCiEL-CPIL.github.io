"""Strong edge colouring: bitmask-accelerated greedy + local search with annealing."""
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

    def greedy_first_fit(edges, conflict, order):
        """Greedy first-fit using conflict bitmasks. Returns (colour_list, num_colours)."""
        m = len(edges)
        colour = [-1] * m
        used = 0
        # For each colour, maintain bitmask of edges already assigned
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
        """Local search: try edge moves, swaps, and class merges. Returns (colour, used)."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()
        
        for _ in range(500):
            if time.time() - start > time_budget:
                break
            improved = False
            
            # Build colour classes
            classes = {}
            for e in range(m):
                c = col_of[e]
                if c >= 0:
                    classes.setdefault(c, []).append(e)
            
            maxc = max(classes.keys()) if classes else 0
            
            # 1) Try merging colour classes (highest into lower)
            for c in range(maxc, -1, -1):
                if c not in classes:
                    continue
                cls_c = classes[c]
                for d in range(c):
                    if d not in classes:
                        continue
                    cls_d = classes[d]
                    # Check no conflicts between the two classes
                    ok = True
                    for e in cls_c:
                        mask_d = 0
                        for f in cls_d:
                            mask_d |= 1 << f
                        if conflict[e] & mask_d:
                            ok = False
                            break
                    if ok:
                        for e in cls_c:
                            col_of[e] = d
                        del classes[c]
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # 2) Try moving a single edge to a lower colour
            for e in range(m):
                ce = col_of[e]
                if ce <= 0:
                    continue
                for d in range(ce):
                    if d not in classes:
                        continue
                    cls_d = classes[d]
                    mask_d = 0
                    for f in cls_d:
                        mask_d |= 1 << f
                    if conflict[e] & mask_d == 0:
                        col_of[e] = d
                        classes[d].append(e)
                        classes[ce].remove(e)
                        if not classes[ce]:
                            del classes[ce]
                        improved = True
                        break
                if improved:
                    break
            
            if improved:
                continue
            
            # 3) Try 1-swap: move e to colour d, move f from d to colour ce
            for e in range(m):
                ce = col_of[e]
                if ce <= 0:
                    continue
                for d in range(ce):
                    if d not in classes:
                        continue
                    cls_d = classes[d]
                    cls_ce = classes[ce]
                    for f in cls_d:
                        if f == e:
                            continue
                        # e can go to d: e must not conflict with any in cls_d except f
                        mask_d_without_f = 0
                        for g in cls_d:
                            if g != f:
                                mask_d_without_f |= 1 << g
                        if conflict[e] & mask_d_without_f:
                            continue
                        # f can go to ce: f must not conflict with any in cls_ce except e
                        mask_ce_without_e = 0
                        for g in cls_ce:
                            if g != e:
                                mask_ce_without_e |= 1 << g
                        if conflict[f] & mask_ce_without_e:
                            continue
                        # Perform swap
                        col_of[e] = d
                        col_of[f] = ce
                        cls_d.remove(f)
                        cls_d.append(e)
                        cls_ce.remove(e)
                        cls_ce.append(f)
                        improved = True
                        break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                break
        
        # Renumber compactly
        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)

    def simulated_annealing(edges, conflict, colour, used, rng, time_budget=5.0):
        """SA-style perturbation and local search."""
        m = len(edges)
        best_colour = colour[:]
        best_used = used
        cur_colour = colour[:]
        cur_used = used
        start = time.time()
        T = 10.0
        
        while time.time() - start < time_budget:
            # Perturb: pick a random edge and change its colour
            e = rng.randrange(m)
            old_c = cur_colour[e]
            new_c = rng.randrange(used + 1)
            if new_c == old_c:
                continue
            cur_colour[e] = new_c
            
            # Check validity and compute new used count
            valid = True
            for f in range(m):
                if f == e:
                    continue
                if cur_colour[f] == new_c and (conflict[e] >> f) & 1:
                    valid = False
                    break
            
            if not valid:
                # Revert
                cur_colour[e] = old_c
                continue
            
            # Compute actual used
            present = set(cur_colour)
            cur_used = len(present)
            
            if cur_used < best_used:
                best_colour = cur_colour[:]
                best_used = cur_used
            elif cur_used > best_used:
                # Accept with probability exp(-(delta)/T)
                delta = cur_used - best_used
                if rng.random() > (2.718281828 ** (-delta / T)):
                    cur_colour[e] = old_c
                    cur_used = best_used
            # Run quick local search
            cur_colour, cur_used = local_search(edges, conflict, cur_colour, cur_used, rng, time_budget=0.5)
            
            if cur_used < best_used:
                best_colour = cur_colour[:]
                best_used = cur_used
            
            T *= 0.95
        
        return best_colour, best_used

    def two_swap_search(edges, conflict, colour, used, rng, time_budget=3.0):
        """Try 2-swap moves: swap edges between two colour classes to enable merges."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()
        
        for _ in range(200):
            if time.time() - start > time_budget:
                break
            improved = False
            
            # Build colour classes
            classes = {}
            for e in range(m):
                c = col_of[e]
                if c >= 0:
                    classes.setdefault(c, []).append(e)
            
            maxc = max(classes.keys()) if classes else 0
            
            # Try swapping one edge from class A with one from class B, then check if a merge is possible
            for a in range(maxc + 1):
                if a not in classes:
                    continue
                for b in range(a):
                    if b not in classes:
                        continue
                    cls_a = classes[a]
                    cls_b = classes[b]
                    
                    # Try swapping edge e from cls_a with edge f from cls_b
                    for e in cls_a:
                        for f in cls_b:
                            # After swap: e goes to b, f goes to a
                            # Check e can go to b (no conflicts with cls_b minus f)
                            ok_e = True
                            for g in cls_b:
                                if g == f:
                                    continue
                                if (conflict[e] >> g) & 1:
                                    ok_e = False
                                    break
                            if not ok_e:
                                continue
                            
                            # Check f can go to a (no conflicts with cls_a minus e)
                            ok_f = True
                            for g in cls_a:
                                if g == e:
                                    continue
                                if (conflict[f] >> g) & 1:
                                    ok_f = False
                                    break
                            if not ok_f:
                                continue
                            
                            # Perform swap
                            col_of[e] = b
                            col_of[f] = a
                            cls_a.remove(e)
                            cls_a.append(f)
                            cls_b.remove(f)
                            cls_b.append(e)
                            improved = True
                            break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if not improved:
                # Try merging after potential swaps
                for c in range(maxc, -1, -1):
                    if c not in classes:
                        continue
                    cls_c = classes[c]
                    for d in range(c):
                        if d not in classes:
                            continue
                        cls_d = classes[d]
                        ok = True
                        mask_d = 0
                        for f in cls_d:
                            mask_d |= 1 << f
                        for e in cls_c:
                            if conflict[e] & mask_d:
                                ok = False
                                break
                        if ok:
                            for e in cls_c:
                                col_of[e] = d
                            del classes[c]
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        # Renumber compactly
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
        
        # Multiple randomized restarts with greedy
        n_restarts = 120
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
            best_colour, best_used = local_search(edges, conflict, best_colour, best_used, rng, time_budget=5.0)
        
        # Two-swap search for further improvement
        if time.time() < DEADLINE - 20.0 and best_colour is not None:
            best_colour, best_used = two_swap_search(edges, conflict, best_colour, best_used, rng, time_budget=5.0)
        
        # Simulated annealing for further improvement
        if time.time() < DEADLINE - 20.0 and best_colour is not None:
            best_colour, best_used = simulated_annealing(edges, conflict, best_colour, best_used, rng, time_budget=8.0)
        
        # More targeted restarts if time permits
        if time.time() < DEADLINE - 25.0:
            for r in range(80):
                if time.time() > DEADLINE - 15.0:
                    break
                order = list(range(m))
                rng.shuffle(order)
                colour, used = greedy_first_fit(edges, conflict, order)
                if used <= best_used:
                    colour, used = local_search(edges, conflict, colour, used, rng, time_budget=2.0)
                    if used < best_used:
                        best_colour = colour[:]
                        best_used = used
        
        if best_colour is None:
            best_colour, best_used = greedy_first_fit(edges, conflict, list(range(m)))
        
        out.append(best_colour)
    
    return out
