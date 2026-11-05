import time
import random
import math
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

    def simulated_annealing(conflicts, m, colors, rng, time_limit=5.0):
        """
        Simulated Annealing to minimize the number of distinct colors.
        Hard constraint: No two conflicting edges can have the same color.
        Soft objective: Minimize the count of distinct colors used.
        """
        if m == 0:
            return colors, 0

        # Start with a valid coloring
        current_colors = colors[:]
        current_colors, current_k = _repack(current_colors, m)
        
        # Precompute neighbor lists for fast conflict checking
        neighbors = [list(conflicts[i]) for i in range(m)]
        
        # Count of edges per color
        color_counts = [0] * current_k
        for c in current_colors:
            color_counts[c] += 1
            
        best_colors = current_colors[:]
        best_k = current_k
        
        start = time.time()
        T = 1.0
        cooling_rate = 0.9999
        min_T = 0.001
        max_color_index = current_k # We allow trying colors up to current_k (or slightly more to escape)
        
        while time.time() < start + time_limit and T > min_T:
            v = rng.randrange(m)
            old_c = current_colors[v]
            
            # Choose a new color
            # Bias towards existing colors to reduce k, but allow new ones if stuck
            if rng.random() < 0.8 and current_k > 0:
                new_c = rng.randrange(current_k)
            else:
                new_c = rng.randrange(max_color_index + 2) # Allow expanding palette slightly
            
            if new_c == old_c:
                continue
                
            # Check hard constraint: Is new_c available for v?
            valid = True
            for nb in neighbors[v]:
                if current_colors[nb] == new_c:
                    valid = False
                    break
                    
            if not valid:
                # If invalid, we could try to "fix" it by moving neighbors, but for now just reject.
                # A more powerful move would be to pick a neighbor with new_c and move it.
                # For simplicity and speed, we reject invalid moves.
                continue
                
            # Calculate change in number of distinct colors
            # If old_c count becomes 0, k decreases by 1
            # If new_c count was 0, k increases by 1
            delta_k = 0
            if color_counts[old_c] == 1:
                delta_k -= 1
            if new_c >= len(color_counts) or color_counts[new_c] == 0:
                if new_c >= len(color_counts):
                    # Extend counts array
                    while len(color_counts) <= new_c:
                        color_counts.append(0)
                delta_k += 1
                
            # Acceptance probability
            if delta_k < 0:
                accept = True
            elif delta_k == 0:
                accept = True # Neutral move, helps explore
            else:
                # delta_k > 0
                if T > 0:
                    prob = math.exp(-delta_k / T)
                    accept = rng.random() < prob
                else:
                    accept = False
                    
            if accept:
                # Apply move
                current_colors[v] = new_c
                color_counts[old_c] -= 1
                color_counts[new_c] += 1
                current_k += delta_k
                
                # Update max_color_index if we used a new color
                if new_c >= max_color_index:
                    max_color_index = new_c
                    
                # Track best
                if current_k < best_k:
                    best_k = current_k
                    best_colors = current_colors[:]
                    
            T *= cooling_rate
            
        best_colors, best_k_final = _repack(best_colors, m)
        return best_colors, best_k_final

    def color_graph(edges, adj, conflicts, rng, time_limit=10.0):
        """Try multiple strategies and return the best (colors, num_colors)."""
        m = len(edges)
        if m == 0:
            return [], 0

        best_colors = None
        best_score = float('inf')

        # 1. Generate initial candidates
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

        # 2. Polish best candidate with local search + colour merging
        if best_colors is not None:
            improved_colors, improved_score = local_search_improve(
                conflicts, m, best_colors, best_score, rng, max_iters=200
            )
            if improved_score < best_score:
                best_score = improved_score
                best_colors = improved_colors

            merged_colors, merged_score = try_merge_colors(
                conflicts, m, best_colors, rng, max_attempts=200
            )
            if merged_score < best_score:
                best_score = merged_score
                best_colors = merged_colors

        # 3. Simulated Annealing on the best candidate
        if best_colors is not None and time.time() < start_time + time_limit - 1.0:
            # Allocate remaining time to SA
            remaining = max(1.0, time_limit - (time.time() - start_time) - 1.0)
            sa_colors, sa_score = simulated_annealing(
                conflicts, m, best_colors, rng, time_limit=remaining
            )
            if sa_score < best_score:
                best_score = sa_score
                best_colors = sa_colors

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

    # Allocate time per graph
    total_time = 140.0
    time_per_graph = total_time / len(graphs)

    for idx, (edges, adj, conflicts) in enumerate(graphs):
        if time.time() > deadline - 2.0:
            # Fallback for remaining graphs
            rng = random.Random(0)
            colors, _ = greedy_color(conflicts, len(edges), rng, 'largest_first')
            best_results.append(colors)
            continue

        graph_start = time.time()
        time_limit_for_graph = min(time_per_graph, deadline - 2.0 - graph_start)
        
        # Run multiple restarts of the heuristic pipeline within the time limit
        best_local_colors = None
        best_local_score = float('inf')
        
        restart_count = 0
        while time.time() < graph_start + time_limit_for_graph - 0.5:
            restart_count += 1
            iter_seed = 10000 + idx * 1000 + restart_count
            rng = random.Random(iter_seed)
            
            # Limit time per single heuristic run to allow restarts
            single_run_limit = 5.0 if restart_count < 5 else (time.time() - graph_start + time_limit_for_graph - time.time() - 0.5)
            if single_run_limit < 0.1:
                break
                
            colors, score = color_graph(edges, adj, conflicts, rng, time_limit=single_run_limit)

            if score < best_local_score:
                best_local_score = score
                best_local_colors = colors
                
            # If we hit the theoretical lower bound or a very good score, we might stop early
            # For d=4, 5d^2/4 = 20. For d=5, 31.25 -> 32? No, 5*25/4 = 31.25. Integer bound is likely ceil.
            # Let's just let it run.

        if best_local_colors is not None:
            best_results.append(best_local_colors)
            best_scores[idx] = best_local_score
        else:
            # Fallback
            rng = random.Random(0)
            colors, _ = greedy_color(conflicts, len(edges), rng, 'largest_first')
            best_results.append(colors)
            best_scores[idx] = _

    return best_results

