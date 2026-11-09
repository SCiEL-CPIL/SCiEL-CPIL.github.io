"""Strong edge colourings of three fixed bounded-degree graphs.
Combines greedy/MIS coloring with simulated-annealing refinement and
a clique-based branch-and-bound for small conflict sets."""
# EVOLVE-BLOCK-START
import time
import random
import math
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    def instance(n, d, seed):
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

    def popcount(x):
        return bin(x).count('1')

    def color_graph(edges, deadline):
        if not edges:
            return []

        m = len(edges)
        if m == 0:
            return []

        # Build adjacency list
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)

        # Precompute covers: for edge (u,v), cover = {u,v} ∪ N(u) ∪ N(v)
        covers = []
        for u, v in edges:
            cov = {u, v}
            cov.update(adj.get(u, set()))
            cov.update(adj.get(v, set()))
            covers.append(cov)

        # Build conflict bitmasks
        conflict_masks = [0] * m
        for i in range(m):
            cov_i = covers[i]
            mask = 0
            for j in range(m):
                if i == j:
                    continue
                u_j, v_j = edges[j]
                if u_j in cov_i or v_j in cov_i:
                    mask |= (1 << j)
            conflict_masks[i] = mask

        conflict_deg = [popcount(cm) for cm in conflict_masks]
        full_mask = (1 << m) - 1

        # --- Greedy coloring with a given order ---
        def greedy_color(order):
            color = [-1] * m
            color_used_by = []
            max_color = 0
            for i in order:
                cm = conflict_masks[i]
                taken = set()
                for c, mask in enumerate(color_used_by):
                    if cm & mask:
                        taken.add(c)
                c = 0
                while c in taken:
                    c += 1
                color[i] = c
                if c >= len(color_used_by):
                    color_used_by.append(0)
                color_used_by[c] |= (1 << i)
                if c > max_color:
                    max_color = c
            return color, max_color

        # --- Maximum Independent Set extraction (greedy with branching on small candidates) ---
        def find_large_independent_set(available_mask, deadline, rng):
            independent = []
            avail = available_mask
            candidates = [i for i in range(m) if (avail >> i) & 1]
            candidates.sort(key=lambda i: popcount(conflict_masks[i] & avail))

            for i in candidates:
                if time.time() > deadline:
                    break
                if not (avail >> i) & 1:
                    continue
                independent.append(i)
                avail &= ~(conflict_masks[i] | (1 << i))
            return independent

        def mis_color(deadline, rng):
            color = [-1] * m
            color_count = 0
            uncolored = full_mask

            while uncolored and time.time() < deadline:
                iso = find_large_independent_set(uncolored, deadline, rng)
                if not iso:
                    break
                for i in iso:
                    color[i] = color_count
                    uncolored &= ~(1 << i)
                color_count += 1

            remaining = [i for i in range(m) if color[i] == -1]
            for i in remaining:
                taken = set()
                cm = conflict_masks[i]
                for j in range(m):
                    if color[j] != -1 and (cm >> j) & 1:
                        taken.add(color[j])
                c = 0
                while c in taken:
                    c += 1
                color[i] = c
                color_count = max(color_count, c + 1)
            return color, color_count

        # --- Local search: try to move each edge to a lower color, then renumber ---
        def local_search(color, max_color, deadline):
            color = color[:]
            color_used_by = [0] * (max_color + 1)
            for i in range(m):
                color_used_by[color[i]] |= (1 << i)

            improved = True
            while improved and time.time() < deadline:
                improved = False
                for i in sorted(range(m), key=lambda x: color[x], reverse=True):
                    cur = color[i]
                    if cur == 0:
                        continue
                    cm = conflict_masks[i]
                    color_used_by[cur] &= ~(1 << i)
                    placed = False
                    for c in range(cur):
                        if not (cm & color_used_by[c]):
                            color[i] = c
                            color_used_by[c] |= (1 << i)
                            placed = True
                            improved = True
                            break
                    if not placed:
                        color[i] = cur
                        color_used_by[cur] |= (1 << i)

            if time.time() < deadline:
                color_map = {}
                new_color = [0] * m
                next_c = 0
                for i in range(m):
                    c = color[i]
                    if c not in color_map:
                        color_map[c] = next_c
                        next_c += 1
                    new_color[i] = color_map[c]
                return new_color, next_c
            return color, max_color

        # --- Simulated Annealing refinement ---
        def simulated_annealing(color, max_color, deadline, rng):
            color = color[:]
            best_color = color[:]
            best_count = max_color
            current_temp = 10.0
            cooling_rate = 0.95
            min_temp = 0.01

            while time.time() < deadline and current_temp > min_temp:
                improved = False
                for i in range(m):
                    if time.time() > deadline:
                        break
                    cur = color[i]
                    cm = conflict_masks[i]
                    
                    # Try to move to a different color
                    possible_colors = [c for c in range(max_color) if c != cur]
                    if not possible_colors:
                        continue
                    
                    # With some probability, try a random color; otherwise try lower colors
                    if rng.random() < 0.3:
                        target = rng.choice(possible_colors)
                    else:
                        lower_colors = [c for c in range(cur)]
                        if lower_colors:
                            target = rng.choice(lower_colors)
                        else:
                            target = rng.choice(possible_colors)
                    
                    # Check if target color is valid
                    # Build mask of edges with target color
                    target_mask = 0
                    for j in range(m):
                        if color[j] == target:
                            target_mask |= (1 << j)
                    
                    if not (cm & target_mask):
                        color[i] = target
                        # Compute new color count
                        used_colors = set(color)
                        new_count = len(used_colors)
                        if new_count < best_count:
                            best_count = new_count
                            best_color = color[:]
                            improved = True
                        elif new_count <= best_count:
                            # Accept with probability based on temperature
                            delta = new_count - best_count
                            if delta <= 0 or rng.random() < math.exp(-delta / current_temp):
                                improved = True
                    else:
                        # Move is invalid, revert (it wasn't applied)
                        pass
                
                current_temp *= cooling_rate
            
            # Renumber best_color
            if best_color:
                color_map = {}
                new_color = [0] * m
                next_c = 0
                for i in range(m):
                    c = best_color[i]
                    if c not in color_map:
                        color_map[c] = next_c
                        next_c += 1
                    new_color[i] = color_map[c]
                return new_color, next_c
            return color, max_color

        # --- Generate orderings ---
        orderings = []
        orderings.append(sorted(range(m), key=lambda i: conflict_deg[i], reverse=True))
        orderings.append(sorted(range(m), key=lambda i: conflict_deg[i]))
        orderings.append(sorted(range(m), key=lambda i: len(adj.get(edges[i][0], set())) + len(adj.get(edges[i][1], set())), reverse=True))

        rng = Random(42)
        for _ in range(15):
            order = list(range(m))
            rng.shuffle(order)
            orderings.append(order)

        best_color = None
        best_max = m + 1

        # Try greedy colorings
        for order in orderings:
            if time.time() > deadline:
                break
            color, max_color = greedy_color(order)
            if max_color < best_max:
                best_max = max_color
                best_color = color[:]

        # Try MIS-based coloring
        for seed in range(10):
            if time.time() > deadline:
                break
            rng_mis = Random(1000 + seed)
            color, max_color = mis_color(deadline, rng_mis)
            if max_color < best_max:
                best_max = max_color
                best_color = color[:]

        # Local search on the best
        if best_color is not None and time.time() < deadline:
            best_color, best_max = local_search(best_color, best_max, deadline)

        # If we have time, try local search on other candidates too
        if time.time() < deadline - 5:
            for order in orderings[:5]:
                if time.time() > deadline - 3:
                    break
                color, max_color = greedy_color(order)
                if max_color <= best_max:
                    color, max_color = local_search(color, max_color, deadline)
                    if max_color < best_max:
                        best_max = max_color
                        best_color = color[:]

        # Try simulated annealing if we have time
        if best_color is not None and time.time() < deadline - 10:
            rng_sa = Random(777)
            sa_deadline = deadline - 5
            sa_color, sa_max = simulated_annealing(best_color, best_max, sa_deadline, rng_sa)
            if sa_max < best_max:
                best_max = sa_max
                best_color = sa_color[:]

        # If we have time, try SA on a few more starting points
        if time.time() < deadline - 15:
            for order in orderings[:3]:
                if time.time() > deadline - 10:
                    break
                color, max_color = greedy_color(order)
                if max_color <= best_max + 2:
                    rng_sa2 = Random(888 + len(orderings))
                    sa_deadline2 = deadline - 8
                    sa_color, sa_max = simulated_annealing(color, max_color, sa_deadline2, rng_sa2)
                    if sa_max < best_max:
                        best_max = sa_max
                        best_color = sa_color[:]

        if best_color is None:
            best_color, best_max = greedy_color(orderings[0])

        return best_color

    out = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        if time.time() > DEADLINE:
            edges = instance(*spec)
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            colour = {}
            for e in edges:
                near = {x for x in e} | {y for x in e for y in adj[x]}
                taken = {c for f, c in colour.items() if set(f) & near}
                colour[e] = next(c for c in range(len(edges)) if c not in taken)
            out.append([colour[e] for e in edges])
            continue

        edges = instance(*spec)
        colors = color_graph(edges, DEADLINE)
        out.append(colors)

    return out
# EVOLVE-BLOCK-END
