import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    def build_conflict_graph(edges, adj):
        """Build conflict graph: two edges conflict if they share a vertex or are joined by an edge."""
        m = len(edges)
        if m == 0:
            return []

        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(m)]

        for i, (u, v) in enumerate(edges):
            # For each neighbor x of u, any edge (x, y) conflicts with (u, v)
            for x in adj[u]:
                for y in adj[x]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i].add(j)
            # For each neighbor y of v, any edge (x, y) conflicts with (u, v)
            for y in adj[v]:
                for x in adj[y]:
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i].add(j)

        return conflicts

    def greedy_color(conflicts, m, rng, strategy='largest_first'):
        """Greedy coloring with given strategy. Returns (colors, num_colors)."""
        if m == 0:
            return [], 0

        color_of_edge = [-1] * m

        if strategy == 'largest_first':
            order = list(range(m))
            order.sort(key=lambda i: len(conflicts[i]), reverse=True)
        elif strategy == 'random':
            order = list(range(m))
            rng.shuffle(order)
        elif strategy == 'smallest_first':
            order = list(range(m))
            order.sort(key=lambda i: len(conflicts[i]))
        elif strategy == 'largest_random_tie':
            order = list(range(m))
            random_vals = [rng.random() for _ in range(m)]
            order.sort(key=lambda i: (-len(conflicts[i]), random_vals[i]))
        else:
            order = list(range(m))
            order.sort(key=lambda i: len(conflicts[i]), reverse=True)

        for i in order:
            forbidden = set()
            for j in conflicts[i]:
                if color_of_edge[j] != -1:
                    forbidden.add(color_of_edge[j])
            c = 0
            while c in forbidden:
                c += 1
            color_of_edge[i] = c

        max_c = max(color_of_edge) if color_of_edge else 0
        return color_of_edge, max_c + 1

    def dsatur_coloring(conflicts, m):
        """DSATUR heuristic: pick vertex with max saturation, break ties by degree."""
        if m == 0:
            return [], 0

        color = [-1] * m
        neighbour_colors = [set() for _ in range(m)]
        degrees = [len(conflicts[i]) for i in range(m)]
        uncoloured = list(range(m))

        while uncoloured:
            best = None
            best_sat = -1
            best_deg = -1
            for v in uncoloured:
                s = len(neighbour_colors[v])
                if s > best_sat or (s == best_sat and degrees[v] > best_deg):
                    best = v
                    best_sat = s
                    best_deg = degrees[v]

            uncoloured.remove(best)

            used = neighbour_colors[best]
            c = 0
            while c in used:
                c += 1
            color[best] = c

            for nb in conflicts[best]:
                if color[nb] == -1:
                    neighbour_colors[nb].add(c)

        max_c = max(color) if color else 0
        return color, max_c + 1

    def _repack(colors, m):
        """Relabel colours to be contiguous 0..k-1. Returns (colors, k)."""
        if m == 0:
            return colors, 0
        present = {}
        for c in colors:
            if c not in present:
                present[c] = len(present)
        remap = [0] * m
        k = 0
        for i, c in enumerate(colors):
            if c not in present:
                present[c] = k
                k += 1
            remap[i] = present[c]
        return remap, k

    def local_search_improve(conflicts, m, colors, max_colors, rng, max_iters=300):
        """Reduce colour count by moving vertices to lower colours / eliminating top colour."""
        if m == 0:
            return colors, 0

        current_colors = colors[:]
        current_max = max_colors

        for _ in range(max_iters):
            improved = False
            nc = current_max

            # Move each vertex to a strictly lower available colour
            for v in range(m):
                current_c = current_colors[v]
                for c in range(current_c):
                    ok = True
                    for nb in conflicts[v]:
                        if current_colors[nb] == c:
                            ok = False
                            break
                    if ok:
                        current_colors[v] = c
                        improved = True
                        break

            # Eliminate the highest colour
            if nc > 1:
                max_c = nc - 1
                vertices_with_max = [v for v in range(m) if current_colors[v] == max_c]
                for v in vertices_with_max:
                    for c in range(max_c):
                        ok = True
                        for nb in conflicts[v]:
                            if current_colors[nb] == c:
                                ok = False
                                break
                        if ok:
                            current_colors[v] = c
                            improved = True
                            break

            if not improved:
                break

            new_max = max(current_colors) + 1 if current_colors else 0
            if new_max < current_max:
                current_max = new_max

        current_colors, k = _repack(current_colors, m)
        return current_colors, k

    def try_merge_colors(conflicts, m, colors, rng, max_attempts=300):
        """Merge the highest colour class into lower colours via moves and 2-swaps."""
        if m == 0:
            return colors, 0

        current_colors = colors[:]
        current_colors, nc = _repack(current_colors, m)

        if nc <= 1:
            return current_colors, nc

        for _ in range(max_attempts):
            max_c = nc - 1
            high_vertices = [v for v in range(m) if current_colors[v] == max_c]
            if not high_vertices:
                break

            moved_any = False
            # 1) Move high vertex to a lower colour
            for v in high_vertices:
                for c in range(max_c):
                    ok = True
                    for nb in conflicts[v]:
                        if current_colors[nb] == c:
                            ok = False
                            break
                    if ok:
                        current_colors[v] = c
                        moved_any = True
                        break

            if not moved_any:
                # 2) Swap a high vertex with a lower-coloured vertex
                for v in high_vertices:
                    if current_colors[v] != max_c:
                        continue
                    for w in range(m):
                        if current_colors[w] >= max_c:
                            continue
                        c_w = current_colors[w]
                        ok_v = True
                        for nb in conflicts[v]:
                            if nb != w and current_colors[nb] == c_w:
                                ok_v = False
                                break
                        if not ok_v:
                            continue
                        ok_w = True
                        for nb in conflicts[w]:
                            if nb != v and current_colors[nb] == max_c:
                                ok_w = False
                                break
                        if ok_w:
                            current_colors[v] = c_w
                            current_colors[w] = max_c
                            moved_any = True
                            break
                    if moved_any:
                        break

            if not moved_any:
                break

            current_colors, nc = _repack(current_colors, m)

        return current_colors, nc

    def color_graph(edges, adj, conflicts, rng):
        """Try multiple strategies and return the best (colors, num_colors)."""
        m = len(edges)
        if m == 0:
            return [], 0

        best_colors = None
        best_score = float('inf')

        candidates = []
        candidates.append(greedy_color(conflicts, m, rng, 'largest_first'))
        candidates.append(greedy_color(conflicts, m, rng, 'random'))
        candidates.append(greedy_color(conflicts, m, rng, 'smallest_first'))
        candidates.append(greedy_color(conflicts, m, rng, 'largest_random_tie'))
        candidates.append(dsatur_coloring(conflicts, m))

        for colors, score in candidates:
            if score < best_score:
                best_score = score
                best_colors = colors

        # Polish best candidate with local search + colour merging
        if best_colors is not None:
            improved_colors, improved_score = local_search_improve(
                conflicts, m, best_colors, best_score, rng, max_iters=300
            )
            if improved_score < best_score:
                best_score = improved_score
                best_colors = improved_colors

            merged_colors, merged_score = try_merge_colors(
                conflicts, m, best_colors, rng, max_attempts=300
            )
            if merged_score < best_score:
                best_score = merged_score
                best_colors = merged_colors

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
        conflicts = build_conflict_graph(edges, adj)
        graphs.append((edges, adj, conflicts))

    best_results = []
    best_scores = [float('inf')] * len(graphs)

    num_iterations = 0
    max_iterations = 5000

    while time.time() < deadline - 5.0:
        num_iterations += 1
        if num_iterations > max_iterations:
            break

        iter_seed = 10000 + num_iterations
        rng = random.Random(iter_seed)

        for idx, (edges, adj, conflicts) in enumerate(graphs):
            if time.time() > deadline - 2.0:
                break

            colors, score = color_graph(edges, adj, conflicts, rng)

            if score < best_scores[idx]:
                best_scores[idx] = score
                if len(best_results) <= idx:
                    while len(best_results) < len(graphs):
                        best_results.append([])
                best_results[idx] = colors

    # Ensure all results are populated
    for i in range(len(graphs)):
        if not best_results[i] or best_scores[i] == float('inf'):
            edges, adj, conflicts = graphs[i]
            rng = random.Random(0)
            colors, _ = color_graph(edges, adj, conflicts, rng)
            best_results[i] = colors

    return best_results
