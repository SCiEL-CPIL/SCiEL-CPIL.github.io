import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflicts(edges, adj):
        """Build conflict sets: two edges conflict if they share a vertex or are joined by an edge."""
        n_edges = len(edges)
        edge_index = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(n_edges)]

        for i, (u, v) in enumerate(edges):
            # Case 1: edges sharing u or v
            for w in adj[u]:
                e1 = (min(u, w), max(u, w))
                if e1 in edge_index:
                    j = edge_index[e1]
                    if j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            for w in adj[v]:
                e1 = (min(v, w), max(v, w))
                if e1 in edge_index:
                    j = edge_index[e1]
                    if j != i:
                        conflicts[i].add(j)
                        conflicts[j].add(i)
            # Case 2: edges joined by an edge (distance-2 in line graph)
            # For edge (u,v), for each neighbor w of u, for each neighbor z of w:
            #   if edge (w,z) exists and z is not u or v, then (u,v) and (w,z) conflict
            for w in adj[u]:
                for z in adj[w]:
                    if z == u or z == v:
                        continue
                    e2 = (min(w, z), max(w, z))
                    if e2 in edge_index:
                        j = edge_index[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
            for w in adj[v]:
                for z in adj[w]:
                    if z == u or z == v:
                        continue
                    e2 = (min(w, z), max(w, z))
                    if e2 in edge_index:
                        j = edge_index[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)

        return conflicts

    def greedy_color(order, conflicts, n_edges):
        """Color edges in given order using first-fit."""
        color = [-1] * n_edges
        color_classes = []  # list of sets
        for idx in order:
            used = set()
            for c, cls in enumerate(color_classes):
                if conflicts[idx] & cls:
                    used.add(c)
            c = 0
            while c in used:
                c += 1
            color[idx] = c
            if c >= len(color_classes):
                color_classes.append(set())
            color_classes[c].add(idx)
        return color, len(color_classes)

    def local_search(color, conflicts, n_edges, max_iters=20000, time_budget=10.0):
        """Local search to reduce number of colors."""
        n_colors = max(color) + 1 if color else 0
        if n_colors == 0:
            return color, n_colors

        # Build color classes
        classes = [set() for _ in range(n_colors)]
        for i, c in enumerate(color):
            classes[c].add(i)

        start = time.time()
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() - start < time_budget:
            improved = False
            iters += 1

            # Try to remove each color class
            for c in range(n_colors):
                if len(classes[c]) == 0:
                    continue
                # Try to move all edges in class c to other classes
                moved = True
                temp_assign = {}
                for e in sorted(classes[c]):
                    placed = False
                    for c2 in range(n_colors):
                        if c2 == c:
                            continue
                        if not (conflicts[e] & classes[c2]):
                            temp_assign[e] = c2
                            placed = True
                            break
                    if not placed:
                        moved = False
                        break
                if moved:
                    for e, c2 in temp_assign.items():
                        classes[c].discard(e)
                        classes[c2].add(e)
                        color[e] = c2
                    n_colors -= 1
                    improved = True
                    break

            # Try pairwise swaps between colors to enable removal
            if not improved:
                for c1 in range(n_colors):
                    if len(classes[c1]) == 0:
                        continue
                    for c2 in range(c1 + 1, n_colors):
                        if len(classes[c2]) == 0:
                            continue
                        # Try swapping an edge from c1 to c2 and vice versa
                        for e1 in list(classes[c1]):
                            for e2 in list(classes[c2]):
                                if e1 in conflicts[e2] or e2 in conflicts[e1]:
                                    continue
                                # After swap: e1 goes to c2, e2 goes to c1
                                # Check if e1 fits in c2 (minus e2) and e2 fits in c1 (minus e1)
                                c2_temp = classes[c2] - {e2}
                                c1_temp = classes[c1] - {e1}
                                if not (conflicts[e1] & c2_temp) and not (conflicts[e2] & c1_temp):
                                    # Do the swap
                                    classes[c1].discard(e1)
                                    classes[c2].discard(e2)
                                    classes[c1].add(e2)
                                    classes[c2].add(e1)
                                    color[e1] = c2
                                    color[e2] = c1
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break

        return color, n_colors

    def advanced_local_search(color, conflicts, n_edges, time_budget=15.0):
        """More aggressive local search with color removal and swaps."""
        n_colors = max(color) + 1 if color else 0
        if n_colors == 0:
            return color, n_colors

        classes = [set() for _ in range(n_colors)]
        for i, c in enumerate(color):
            classes[c].add(i)

        start = time.time()
        while time.time() - start < time_budget:
            improved = False

            # Try to remove a color
            for c in range(n_colors):
                if len(classes[c]) == 0:
                    continue
                moved = True
                temp_assign = {}
                for e in sorted(classes[c]):
                    placed = False
                    for c2 in range(n_colors):
                        if c2 == c:
                            continue
                        if not (conflicts[e] & classes[c2]):
                            temp_assign[e] = c2
                            placed = True
                            break
                    if not placed:
                        moved = False
                        break
                if moved:
                    for e, c2 in temp_assign.items():
                        classes[c].discard(e)
                        classes[c2].add(e)
                        color[e] = c2
                    n_colors -= 1
                    improved = True
                    break

            if improved:
                continue

            # Try 2-swap moves
            found_swap = False
            for c1 in range(n_colors):
                if len(classes[c1]) == 0:
                    continue
                for c2 in range(c1 + 1, n_colors):
                    if len(classes[c2]) == 0:
                        continue
                    for e1 in list(classes[c1]):
                        for e2 in list(classes[c2]):
                            if e1 in conflicts[e2] or e2 in conflicts[e1]:
                                continue
                            c2_temp = classes[c2] - {e2}
                            c1_temp = classes[c1] - {e1}
                            if not (conflicts[e1] & c2_temp) and not (conflicts[e2] & c1_temp):
                                classes[c1].discard(e1)
                                classes[c2].discard(e2)
                                classes[c1].add(e2)
                                classes[c2].add(e1)
                                color[e1] = c2
                                color[e2] = c1
                                found_swap = True
                                break
                        if found_swap:
                            break
                    if found_swap:
                        break
                if found_swap:
                    break

            if not found_swap:
                # Try random perturbation to escape local optimum
                if n_colors > 0:
                    c = random.randrange(n_colors)
                    if classes[c]:
                        e = random.choice(list(classes[c]))
                        # Move e to a random valid color
                        valid_colors = [c2 for c2 in range(n_colors) if c2 != c and not (conflicts[e] & classes[c2])]
                        if valid_colors:
                            new_c = random.choice(valid_colors)
                            classes[c].discard(e)
                            classes[new_c].add(e)
                            color[e] = new_c

        return color, n_colors

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        edges = instance(*spec)
        n_edges = len(edges)

        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)

        conflicts = build_conflicts(edges, adj)
        conf_deg = [len(conflicts[i]) for i in range(n_edges)]

        best_color = None
        best_n = float('inf')

        # Strategy 1: Sort by descending conflict degree
        order1 = sorted(range(n_edges), key=lambda i: -conf_deg[i])
        color1, n1 = greedy_color(order1, conflicts, n_edges)
        color1, n1 = local_search(color1, conflicts, n_edges, max_iters=30000, time_budget=8.0)
        if n1 < best_n:
            best_n = n1
            best_color = color1[:]

        # Strategy 2: Multiple random orderings with local search
        rng = random.Random(12345)
        n_attempts = 300
        for attempt in range(n_attempts):
            if time.time() > deadline - 10:
                break
            order = list(range(n_edges))
            rng.shuffle(order)
            color, n = greedy_color(order, conflicts, n_edges)
            color, n = local_search(color, conflicts, n_edges, max_iters=15000, time_budget=4.0)
            if n < best_n:
                best_n = n
                best_color = color[:]

        # Strategy 3: Weighted random orderings (higher conflict degree = higher priority)
        for attempt in range(100):
            if time.time() > deadline - 10:
                break
            weights = [conf_deg[i] + 1 for i in range(n_edges)]
            total_w = sum(weights)
            order = []
            remaining = set(range(n_edges))
            while remaining:
                r = rng.random() * total_w
                cum = 0
                chosen = None
                for i in sorted(remaining, key=lambda x: -conf_deg[x]):
                    cum += weights[i]
                    if r <= cum:
                        chosen = i
                        break
                if chosen is None:
                    chosen = sorted(remaining, key=lambda x: -conf_deg[x])[0]
                order.append(chosen)
                remaining.remove(chosen)
                total_w -= weights[chosen]

            color, n = greedy_color(order, conflicts, n_edges)
            color, n = local_search(color, conflicts, n_edges, max_iters=15000, time_budget=4.0)
            if n < best_n:
                best_n = n
                best_color = color[:]

        # Strategy 4: Use advanced local search on the best solution
        if best_color is not None and time.time() < deadline - 15:
            color_adv, n_adv = advanced_local_search(best_color[:], conflicts, n_edges, time_budget=min(15.0, deadline - time.time() - 5))
            if n_adv < best_n:
                best_n = n_adv
                best_color = color_adv

        if best_color is None:
            best_color = list(range(n_edges))
            best_n = n_edges

        results.append([best_color[i] for i in range(n_edges)])

    return results
