import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 138.0  # Safety margin before 150s kill

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

    def build_conflict_graph_fast(edges, adj):
        """Build conflict graph using adjacency-based approach (faster for bounded degree)."""
        m = len(edges)
        if m == 0:
            return [set() for _ in range(m)]

        edge_to_idx = {e: i for i, e in enumerate(edges)}
        conflicts = [set() for _ in range(m)]

        for i, (u, v) in enumerate(edges):
            # For each neighbor x of u, any edge (x, y) conflicts with (u, v)
            for x in adj[u]:
                for y in adj[x]:
                    if x == u or y == u:
                        continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)
            # For each neighbor y of v, any edge (x, y) conflicts with (u, v)
            for y in adj[v]:
                for x in adj[y]:
                    if x == v or y == v:
                        continue
                    e2 = (min(x, y), max(x, y))
                    if e2 in edge_to_idx:
                        j = edge_to_idx[e2]
                        if j != i:
                            conflicts[i].add(j)
                            conflicts[j].add(i)

        return conflicts

    def dsatur_coloring(conflicts, m):
        """DSATUR heuristic for graph colouring."""
        if m == 0:
            return [], 0

        color = [-1] * m
        neighbour_colors = [set() for _ in range(m)]
        degrees = [len(conflicts[i]) for i in range(m)]
        uncoloured = set(range(m))

        while uncoloured:
            # Find vertex with max saturation degree, break ties by degree
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

            # Find smallest available colour
            used = neighbour_colors[best]
            c = 0
            while c in used:
                c += 1

            color[best] = c

            # Update neighbours
            for nb in conflicts[best]:
                if color[nb] == -1:
                    neighbour_colors[nb].add(c)

        max_c = max(color) if color else 0
        return color, max_c + 1

    def greedy_color_bitmask(conflicts, m, rng, strategy='largest_first'):
        """Greedy coloring using bitmask for fast color availability checks."""
        if m == 0:
            return [], 0

        color_of_edge = [-1] * m
        # Precompute neighbor lists as lists for faster iteration
        conflict_lists = [list(conflicts[i]) for i in range(m)]

        if strategy == 'largest_first':
            order = list(range(m))
            order.sort(key=lambda i: len(conflict_lists[i]), reverse=True)
        elif strategy == 'random':
            order = list(range(m))
            rng.shuffle(order)
        elif strategy == 'smallest_first':
            order = list(range(m))
            order.sort(key=lambda i: len(conflict_lists[i]))
        elif strategy == 'largest_random_tie':
            order = list(range(m))
            random_vals = [rng.random() for _ in range(m)]
            order.sort(key=lambda i: (-len(conflict_lists[i]), random_vals[i]))
        else:
            order = list(range(m))
            order.sort(key=lambda i: len(conflict_lists[i]), reverse=True)

        for i in order:
            # Use bitmask to find available color
            forbidden_mask = 0
            for j in conflict_lists[i]:
                if color_of_edge[j] != -1:
                    forbidden_mask |= (1 << color_of_edge[j])
            # Find lowest unset bit
            c = 0
            while forbidden_mask & (1 << c):
                c += 1
            color_of_edge[i] = c

        max_c = max(color_of_edge) if color_of_edge else 0
        return color_of_edge, max_c + 1

    def local_search_improve(conflicts, m, colors, max_colors, rng, max_iters=300):
        """Enhanced local search with swaps to reduce number of colors."""
        if m == 0:
            return colors, 0

        current_colors = colors[:]
        current_max = max_colors
        conflict_lists = [list(conflicts[i]) for i in range(m)]

        for _ in range(max_iters):
            improved = False
            max_c = max(current_colors) if current_colors else -1

            # Try to move each vertex to a lower colour
            for v in range(m):
                current_c = current_colors[v]
                if current_c == 0:
                    continue
                forbidden_mask = 0
                for nb in conflict_lists[v]:
                    if current_colors[nb] != -1:
                        forbidden_mask |= (1 << current_colors[nb])
                for c in range(current_c):
                    if not (forbidden_mask & (1 << c)):
                        current_colors[v] = c
                        improved = True
                        break

            # Try to eliminate the highest colour via moves
            if max_c > 0:
                high_vertices = [v for v in range(m) if current_colors[v] == max_c]
                for v in high_vertices:
                    forbidden_mask = 0
                    for nb in conflict_lists[v]:
                        if current_colors[nb] != -1:
                            forbidden_mask |= (1 << current_colors[nb])
                    for c in range(max_c):
                        if not (forbidden_mask & (1 << c)):
                            current_colors[v] = c
                            improved = True
                            break

            # Try swaps between high-colored vertices
            if max_c > 0:
                high_vertices = [v for v in range(m) if current_colors[v] == max_c]
                for i_idx, v in enumerate(high_vertices):
                    if current_colors[v] != max_c:
                        continue
                    for w in high_vertices[i_idx+1:]:
                        if current_colors[w] != max_c:
                            continue
                        # Try: v takes w's color (if different), w takes max_c
                        # But w already has max_c, so try: v takes some c < max_c, w stays
                        # Better: swap colors if v can take w's color and w can take v's
                        # Since both have max_c, swapping doesn't help directly.
                        # Instead: try to move v to a color that w has (if w had a different color)
                        # For now, just try moving v to any available lower color
                        forbidden_mask = 0
                        for nb in conflict_lists[v]:
                            if nb != w and current_colors[nb] != -1:
                                forbidden_mask |= (1 << current_colors[nb])
                        for c in range(max_c):
                            if not (forbidden_mask & (1 << c)):
                                current_colors[v] = c
                                improved = True
                                break
                        if improved:
                            break

            if not improved:
                break

            new_max = max(current_colors) + 1 if current_colors else 0
            if new_max < current_max:
                current_max = new_max

        return current_colors, max(current_colors) + 1 if current_colors else 0

    def try_merge_colors(conflicts, m, colors, rng, max_attempts=100):
        """Try to merge the highest color class into lower colors with smarter ordering."""
        if m == 0:
            return colors, 0

        current_colors = colors[:]
        nc = max(current_colors) + 1 if current_colors else 0
        conflict_lists = [list(conflicts[i]) for i in range(m)]
        
        # Precompute neighbor counts
        neighbor_counts = [len(conflict_lists[i]) for i in range(m)]

        if nc <= 1:
            return current_colors, nc

        max_c = nc - 1
        for _ in range(max_attempts):
            # Get vertices with max color, sorted by degree (low degree first - easier to move)
            high_vertices = [v for v in range(m) if current_colors[v] == max_c]
            if not high_vertices:
                break
            
            # Sort by degree ascending - low degree vertices have fewer constraints
            high_vertices.sort(key=lambda v: neighbor_counts[v])
            
            moved_any = False
            for v in high_vertices:
                if current_colors[v] != max_c:
                    continue
                forbidden_mask = 0
                for nb in conflict_lists[v]:
                    if current_colors[nb] != -1:
                        forbidden_mask |= (1 << current_colors[nb])
                for c in range(max_c):
                    if not (forbidden_mask & (1 << c)):
                        current_colors[v] = c
                        moved_any = True
                        break

            if not moved_any:
                break

            new_max = max(current_colors) + 1 if current_colors else 0
            if new_max < nc:
                nc = new_max
                max_c = nc - 1

        return current_colors, max(current_colors) + 1 if current_colors else 0

    def color_graph(edges, adj, conflicts, rng):
        """Try multiple strategies and return the best coloring."""
        m = len(edges)
        if m == 0:
            return [], 0

        best_colors = None
        best_score = float('inf')

        # Try DSATUR
        dsat_colors, dsat_score = dsatur_coloring(conflicts, m)
        if dsat_score < best_score:
            best_score = dsat_score
            best_colors = dsat_colors[:]

        # Try greedy strategies with bitmask
        strategies = ['largest_first', 'random', 'smallest_first', 'largest_random_tie']
        for strategy in strategies:
            colors, score = greedy_color_bitmask(conflicts, m, rng, strategy)
            if score < best_score:
                best_score = score
                best_colors = colors[:]

        # Try local search improvement on the best result
        if best_colors is not None:
            improved_colors, improved_score = local_search_improve(
                conflicts, m, best_colors, best_score, rng, max_iters=300
            )
            if improved_score < best_score:
                best_score = improved_score
                best_colors = improved_colors

            # Try color merging after local search
            merged_colors, merged_score = try_merge_colors(
                conflicts, m, best_colors, rng, max_attempts=100
            )
            if merged_score < best_score:
                best_score = merged_score
                best_colors = merged_colors

            # Final polish: another round of local search on the merged result
            if best_colors is not None:
                final_colors, final_score = local_search_improve(
                    conflicts, m, best_colors, best_score, rng, max_iters=100
                )
                if final_score < best_score:
                    best_score = final_score
                    best_colors = final_colors

        return best_colors, best_score

    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))

    # Pre-generate graphs with fast conflict building
    graphs = []
    for spec in specs:
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        conflicts = build_conflict_graph_fast(edges, adj)
        graphs.append((edges, adj, conflicts))

    best_results = []
    best_scores = [float('inf')] * len(graphs)

    # Run multiple iterations with different random seeds
    num_iterations = 0
    max_iterations = 3000

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
            
            # Early termination if we've reached the theoretical lower bound
            # For d=4: 5*16/4 = 20, for d=5: 5*25/4 = 31.25 -> 32, for d=6: 5*36/4 = 45
            # But we can do better than the conjecture bound in practice

    # Ensure all results are populated
    for i in range(len(graphs)):
        if not best_results[i] or best_scores[i] == float('inf'):
            edges, adj, conflicts = graphs[i]
            rng = random.Random(0)
            colors, _ = color_graph(edges, adj, conflicts, rng)
            best_results[i] = colors

    return best_results
