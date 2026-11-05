import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

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

    def build_conflict_bitsets(edges, adj):
        """Build conflict graph as list of bitmasks (one per edge)."""
        m = len(edges)
        if m == 0:
            return []

        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [0] * m

        for i, (u, v) in enumerate(edges):
            # Edges sharing a vertex with (u,v)
            for x in adj[u]:
                for y in adj[x]:
                    if x == u or y == u:
                        continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i] |= (1 << j)
                            conflicts[j] |= (1 << i)
            for y in adj[v]:
                for x in adj[y]:
                    if x == v or y == v:
                        continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i] |= (1 << j)
                            conflicts[j] |= (1 << i)

        return conflicts

    def greedy_dsatur(conflicts, m, order_key='saturation'):
        """DSATUR-style greedy coloring using bitmasks."""
        if m == 0:
            return [], 0

        color = [-1] * m
        neighbor_colors = [0] * m  # bitmask of colors used by neighbors
        degrees = [bin(conflicts[i]).count('1') for i in range(m)]
        uncoloured = set(range(m))

        while uncoloured:
            # Find vertex with max saturation, break ties by degree
            best = None
            best_sat = -1
            best_deg = -1
            for v in uncoloured:
                s = bin(neighbor_colors[v]).count('1')
                if s > best_sat or (s == best_sat and degrees[v] > best_deg):
                    best = v
                    best_sat = s
                    best_deg = degrees[v]

            uncoloured.remove(best)

            # Find smallest available colour
            used = neighbor_colors[best]
            c = 0
            while used & (1 << c):
                c += 1

            color[best] = c

            # Update neighbours
            for nb in range(m):
                if color[nb] == -1 and (conflicts[best] >> nb) & 1:
                    neighbor_colors[nb] |= (1 << c)

        max_c = max(color) if color else 0
        return color, max_c + 1

    def greedy_with_order(conflicts, m, order):
        """Greedy coloring with a given vertex ordering."""
        if m == 0:
            return [], 0

        color = [-1] * m
        for i in order:
            forbidden = 0
            for j in range(m):
                if color[j] != -1 and (conflicts[i] >> j) & 1:
                    forbidden |= (1 << color[j])
            c = 0
            while forbidden & (1 << c):
                c += 1
            color[i] = c

        max_c = max(color) if color else 0
        return color, max_c + 1

    def local_search(conflicts, m, colors, max_colors, rng, max_iters=500):
        """Local search to reduce number of colors."""
        if m == 0:
            return colors, 0

        current = colors[:]
        current_max = max_colors
        conflict_lists = [[] for _ in range(m)]
        for i in range(m):
            mask = conflicts[i]
            lst = []
            for j in range(m):
                if (mask >> j) & 1:
                    lst.append(j)
            conflict_lists[i] = lst

        for _ in range(max_iters):
            improved = False
            max_c = max(current) if current else -1

            # Pass 1: try to move each vertex to a lower color
            for v in range(m):
                if current[v] == 0:
                    continue
                forbidden = 0
                for nb in conflict_lists[v]:
                    if current[nb] != -1:
                        forbidden |= (1 << current[nb])
                for c in range(current[v]):
                    if not (forbidden & (1 << c)):
                        current[v] = c
                        improved = True
                        break

            # Pass 2: try to eliminate highest color
            if max_c > 0:
                high = [v for v in range(m) if current[v] == max_c]
                for v in high:
                    if current[v] != max_c:
                        continue
                    forbidden = 0
                    for nb in conflict_lists[v]:
                        if current[nb] != -1:
                            forbidden |= (1 << current[nb])
                    for c in range(max_c):
                        if not (forbidden & (1 << c)):
                            current[v] = c
                            improved = True
                            break

            # Pass 3: swap-based moves
            if max_c > 0:
                for v in range(m):
                    if current[v] < max_c:
                        continue
                    for w in range(m):
                        if current[w] == max_c or w == v:
                            continue
                        # Try swapping colors of v and w
                        cv, cw = current[v], current[w]
                        if cv == cw:
                            continue
                        # Check if swap is valid
                        ok = True
                        for nb in conflict_lists[v]:
                            if nb != w and current[nb] == cw:
                                ok = False
                                break
                        if ok:
                            for nb in conflict_lists[w]:
                                if nb != v and current[nb] == cv:
                                    ok = False
                                    break
                        if ok:
                            current[v], current[w] = cw, cv
                            improved = True
                            break

            if not improved:
                break

            new_max = max(current) + 1 if current else 0
            if new_max < current_max:
                current_max = new_max

        return current, max(current) + 1 if current else 0

    def branch_and_bound(conflicts, m, upper_bound, rng, time_limit=5.0):
        """Branch and bound to find coloring with at most upper_bound-1 colors."""
        if m == 0:
            return [], 0

        start = time.time()
        conflict_lists = [[] for _ in range(m)]
        for i in range(m):
            mask = conflicts[i]
            lst = []
            for j in range(m):
                if (mask >> j) & 1:
                    lst.append(j)
            conflict_lists[i] = lst

        # Initial upper bound
        ub = upper_bound - 1  # We want at most ub colors (0-indexed)

        # Try to color with ub colors using B&B
        color = [-1] * m
        color_count = [0] * (ub + 1)

        def is_valid(v, c):
            for nb in conflict_lists[v]:
                if color[nb] == c:
                    return False
            return True

        def search(v, colors_used):
            nonlocal ub
            if time.time() - start > time_limit:
                return False

            if v == m:
                return True

            # Try colors in order
            for c in range(colors_used + 1):
                if c < colors_used and color_count[c] == 0:
                    continue  # Don't introduce unused color before using existing ones
                if c >= ub:
                    break
                if is_valid(v, c):
                    color[v] = c
                    color_count[c] += 1
                    next_v = v + 1
                    next_cu = max(colors_used, c + 1)
                    if search(next_v, next_cu):
                        return True
                    color_count[c] -= 1
                    color[v] = -1

            return False

        # Use a good ordering: sort by degree descending
        order = list(range(m))
        order.sort(key=lambda i: len(conflict_lists[i]), reverse=True)

        # Remap
        remap = {order[i]: i for i in range(m)}
        new_conflicts = [[] for _ in range(m)]
        for i in range(m):
            for j in conflict_lists[i]:
                new_conflicts[remap[i]].append(remap[j])

        # Rebuild for ordered version
        conflict_lists_ordered = new_conflicts
        color_ordered = [-1] * m
        color_count_ordered = [0] * (ub + 1)

        def is_valid_ordered(v, c):
            for nb in conflict_lists_ordered[v]:
                if color_ordered[nb] == c:
                    return False
            return True

        def search_ordered(v, colors_used):
            nonlocal ub
            if time.time() - start > time_limit:
                return False

            if v == m:
                return True

            for c in range(colors_used + 1):
                if c < colors_used and color_count_ordered[c] == 0:
                    continue
                if c >= ub:
                    break
                if is_valid_ordered(v, c):
                    color_ordered[v] = c
                    color_count_ordered[c] += 1
                    if search_ordered(v + 1, max(colors_used, c + 1)):
                        return True
                    color_count_ordered[c] -= 1
                    color_ordered[v] = -1

            return False

        if search_ordered(0, 0):
            # Convert back to original ordering
            result = [-1] * m
            for i in range(m):
                result[order[i]] = color_ordered[i]
            return result, max(color_ordered) + 1

        return None, None

    def color_instance(edges, adj, conflicts, rng):
        """Main coloring routine for one instance."""
        m = len(edges)
        if m == 0:
            return [], 0

        best_colors = None
        best_score = float('inf')

        # 1. DSATUR
        dsat_colors, dsat_score = greedy_dsatur(conflicts, m)
        if dsat_score < best_score:
            best_score = dsat_score
            best_colors = dsat_colors[:]

        # 2. Greedy with degree ordering
        order = list(range(m))
        order.sort(key=lambda i: bin(conflicts[i]).count('1'), reverse=True)
        g_colors, g_score = greedy_with_order(conflicts, m, order)
        if g_score < best_score:
            best_score = g_score
            best_colors = g_colors[:]

        # 3. Local search on best
        if best_colors is not None:
            ls_colors, ls_score = local_search(conflicts, m, best_colors, best_score, rng, max_iters=300)
            if ls_score < best_score:
                best_score = ls_score
                best_colors = ls_colors[:]

        # 4. Try B&B to beat current best
        if best_colors is not None and best_score > 1:
            # Try to color with best_score - 1 colors
            bb_colors, bb_score = branch_and_bound(conflicts, m, best_score, rng, time_limit=3.0)
            if bb_colors is not None and bb_score < best_score:
                best_score = bb_score
                best_colors = bb_colors[:]
                # Try one more reduction
                if best_score > 1:
                    bb2_colors, bb2_score = branch_and_bound(conflicts, m, best_score, rng, time_limit=3.0)
                    if bb2_colors is not None and bb2_score < best_score:
                        best_score = bb2_score
                        best_colors = bb2_colors[:]

        return best_colors, best_score

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))

    # Pre-generate graphs
    graphs = []
    for spec in specs:
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        conflicts = build_conflict_bitsets(edges, adj)
        graphs.append((edges, adj, conflicts))

    best_results = [[] for _ in graphs]
    best_scores = [float('inf')] * len(graphs)

    # Multiple iterations with different seeds
    num_iterations = 0
    max_iterations = 3000

    while time.time() < deadline - 5.0:
        num_iterations += 1
        if num_iterations > max_iterations:
            break

        iter_seed = 20000 + num_iterations
        rng = random.Random(iter_seed)

        for idx, (edges, adj, conflicts) in enumerate(graphs):
            if time.time() > deadline - 2.0:
                break

            colors, score = color_instance(edges, adj, conflicts, rng)

            if score < best_scores[idx]:
                best_scores[idx] = score
                best_results[idx] = colors

    # Ensure all results are populated
    for i in range(len(graphs)):
        if not best_results[i] or best_scores[i] == float('inf'):
            edges, adj, conflicts = graphs[i]
            rng = random.Random(0)
            colors, _ = color_instance(edges, adj, conflicts, rng)
            best_results[i] = colors

    return best_results
