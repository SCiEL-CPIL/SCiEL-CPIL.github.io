"""Strong edge colouring: bitmask-accelerated greedy + local search with annealing."""
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
        
        for _ in range(1000):
            if time.time() - start > time_budget:
                break
            improved = False
            
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
            
            # 3) Try 1-swap
            if rng.random() < 0.5:
                for _ in range(10):
                    e = rng.randrange(m)
                    ce = col_of[e]
                    if ce <= 0:
                        continue
                    
                    lower_colors = [d for d in range(ce) if d in classes]
                    if not lower_colors:
                        continue
                    d = rng.choice(lower_colors)
                    
                    if not classes[d]:
                        continue
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
            
            if not improved:
                break
        
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
            if rng.random() < 0.5:
                e = rng.randrange(m)
                old_c = cur_colour[e]
                new_c = rng.randrange(used + 1)
                if new_c == old_c:
                    continue
                cur_colour[e] = new_c
                
                valid = True
                for f in range(m):
                    if f == e:
                        continue
                    if cur_colour[f] == new_c and (conflict[e] >> f) & 1:
                        valid = False
                        break
                
                if not valid:
                    cur_colour[e] = old_c
                    continue
            else:
                e1 = rng.randrange(m)
                e2 = rng.randrange(m)
                if e1 == e2:
                    continue
                c1 = cur_colour[e1]
                c2 = cur_colour[e2]
                if c1 == c2:
                    continue
                
                cur_colour[e1] = c2
                cur_colour[e2] = c1
                
                valid = True
                for f in range(m):
                    if f == e1:
                        continue
                    if cur_colour[f] == c2 and (conflict[e1] >> f) & 1:
                        valid = False
                        break
                
                if valid:
                    for f in range(m):
                        if f == e2:
                            continue
                        if cur_colour[f] == c1 and (conflict[e2] >> f) & 1:
                            valid = False
                            break
                
                if not valid:
                    cur_colour[e1] = c1
                    cur_colour[e2] = c2
                    continue
            
            present = set(cur_colour)
            cur_used = len(present)
            
            if cur_used < best_used:
                best_colour = cur_colour[:]
                best_used = cur_used
            else:
                delta = cur_used - best_used
                if delta > 0:
                    if rng.random() > math.exp(-delta / T):
                        pass
            
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
        
        for _ in range(500):
            if time.time() - start > time_budget:
                break
            improved = False
            
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
            
            if maxc < 1:
                break
                
            a = rng.randrange(maxc + 1)
            b = rng.randrange(a) if a > 0 else 0
            if a == b or a not in classes or b not in classes:
                continue
                
            cls_a = classes[a]
            cls_b = classes[b]
            mask_a = class_masks[a]
            mask_b = class_masks[b]
            
            if not cls_a or not cls_b:
                continue
            e = rng.choice(cls_a)
            f = rng.choice(cls_b)
            
            mask_b_without_f = mask_b & ~(1 << f)
            if conflict[e] & mask_b_without_f:
                continue
            
            mask_a_without_e = mask_a & ~(1 << e)
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
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                pass
        
        present = sorted(set(col_of))
        remap = {old: new for new, old in enumerate(present)}
        for e in range(m):
            col_of[e] = remap[col_of[e]]
        return col_of, len(present)

    def branch_and_bound(edges, conflict, colour, used, rng, time_budget=5.0):
        """Branch-and-bound to reduce colour count by trying to eliminate top colours."""
        m = len(edges)
        col_of = colour[:]
        start = time.time()
        best_used = used
        
        # Build initial classes
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
        
        # Try to eliminate colours from highest to lowest
        for target_c in range(used - 1, -1, -1):
            if time.time() - start > time_budget:
                break
            if target_c not in classes:
                continue
            
            # For each edge in target_c, try to reassign it to another colour
            # Use backtracking with forward checking
            edges_to_reassign = list(classes[target_c])
            edges_to_reassign.sort(key=lambda e: sum(1 for f in range(m) if f != e and conflict[e] & (1 << f) and col_of[f] != target_c), reverse=True)
            
            def backtrack(idx, classes_copy, class_masks_copy):
                nonlocal best_used
                if time.time() - start > time_budget:
                    return False
                if idx == len(edges_to_reassign):
                    return True
                
                e = edges_to_reassign[idx]
                # Try assigning e to each available colour (not target_c)
                available = [c for c in classes_copy if c != target_c]
                rng.shuffle(available)
                
                for new_c in available:
                    mask_new = class_masks_copy[new_c]
                    if conflict[e] & mask_new:
                        continue
                    
                    # Try this assignment
                    classes_copy[new_c].append(e)
                    class_masks_copy[new_c] |= (1 << e)
                    
                    if backtrack(idx + 1, classes_copy, class_masks_copy):
                        return True
                    
                    # Undo
                    classes_copy[new_c].remove(e)
                    class_masks_copy[new_c] &= ~(1 << e)
                
                return False
            
            # Deep copy classes for backtracking
            classes_copy = {c: lst[:] for c, lst in classes.items()}
            class_masks_copy = dict(class_masks)
            
            if backtrack(0, classes_copy, class_masks_copy):
                # Update col_of from classes_copy
                for e in range(m):
                    col_of[e] = target_c  # default
                for c, lst in classes_copy.items():
                    for e in lst:
                        col_of[e] = c
                # Remove target_c if empty
                if not classes_copy.get(target_c, []):
                    del classes_copy[target_c]
                    del class_masks_copy[target_c]
                # Renumber
                present = sorted(set(col_of))
                remap = {old: new for new, old in enumerate(present)}
                for e in range(m):
                    col_of[e] = remap[col_of[e]]
                best_used = len(present)
                # Update classes and class_masks for next iteration
                classes = {}
                class_masks = {}
                for e in range(m):
                    c = col_of[e]
                    if c not in classes:
                        classes[c] = []
                        class_masks[c] = 0
                    classes[c].append(e)
                    class_masks[c] |= 1 << e
                continue
            else:
                # Can't eliminate target_c, continue to next
                continue
        
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
        
        if best_colour is not None:
            best_colour, best_used = local_search(edges, conflict, best_colour, best_used, rng, time_budget=10.0)
        
        if time.time() < DEADLINE - 20.0 and best_colour is not None:
            best_colour, best_used = two_swap_search(edges, conflict, best_colour, best_used, rng, time_budget=10.0)
        
        if time.time() < DEADLINE - 20.0 and best_colour is not None:
            best_colour, best_used = simulated_annealing(edges, conflict, best_colour, best_used, rng, time_budget=15.0)
        
        # Branch-and-bound to further reduce
        if time.time() < DEADLINE - 20.0 and best_colour is not None:
            best_colour, best_used = branch_and_bound(edges, conflict, best_colour, best_used, rng, time_budget=10.0)
        
        if time.time() < DEADLINE - 25.0:
            for r in range(100):
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
