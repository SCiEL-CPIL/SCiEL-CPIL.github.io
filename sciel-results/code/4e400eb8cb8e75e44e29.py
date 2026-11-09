"""Strong edge colouring of bounded-degree graphs via DSATUR + greedy + swap-based local search."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_MARGIN = 5  # seconds to finish before 150s wall-clock

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN

    def get_instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges), degree

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def precompute_conflicts(edges, adj):
        """Precompute conflict sets using vertex-edge indexing for speed."""
        n_edges = len(edges)
        conflicts = [set() for _ in range(n_edges)]
        edges_at_vertex = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            edges_at_vertex[u].append(i)
            edges_at_vertex[v].append(i)

        for i, (u1, v1) in enumerate(edges):
            # Edges sharing a vertex
            for e2_idx in edges_at_vertex[u1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)
            for e2_idx in edges_at_vertex[v1]:
                if e2_idx != i:
                    conflicts[i].add(e2_idx)

            # Distance-2 edges: neighbours of u1 or v1 (excluding the edge itself)
            for neighbor in adj[u1]:
                if neighbor == v1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx == i:
                        continue
                    u2, v2 = edges[e2_idx]
                    if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                        conflicts[i].add(e2_idx)
            for neighbor in adj[v1]:
                if neighbor == u1:
                    continue
                for e2_idx in edges_at_vertex[neighbor]:
                    if e2_idx == i:
                        continue
                    u2, v2 = edges[e2_idx]
                    if u2 != u1 and u2 != v1 and v2 != u1 and v2 != v1:
                        conflicts[i].add(e2_idx)

        return conflicts

    def conflicts_to_bitmask(conflicts, n_edges):
        """Convert conflict sets to bitmask integers for fast operations."""
        masks = [0] * n_edges
        for i in range(n_edges):
            mask = 0
            for j in conflicts[i]:
                mask |= (1 << j)
            masks[i] = mask
        return masks

    def dsatur_coloring(edges, conflicts, masks):
        """DSATUR heuristic: colour the most-saturated edge first."""
        n_edges = len(edges)
        color = [-1] * n_edges
        color_masks = [0] * (n_edges + 1)  # color_masks[c] = bitmask of edges with color c

        for _ in range(n_edges):
            if time.time() > deadline:
                break
            best_idx = -1
            best_sat = -1
            best_deg = -1
            for i in range(n_edges):
                if color[i] != -1:
                    continue
                # Count distinct colors among neighbors
                neighbor_color_mask = 0
                m = masks[i]
                while m:
                    low = m & (-m)
                    j = low.bit_length() - 1
                    if color[j] != -1:
                        neighbor_color_mask |= (1 << color[j])
                    m ^= low
                sat = neighbor_color_mask.bit_count()
                deg = masks[i].bit_count()
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_idx = i

            if best_idx == -1:
                break

            # Find smallest available color
            c = 0
            while (color_masks[c] & masks[best_idx]):
                c += 1
            color[best_idx] = c
            color_masks[c] |= (1 << best_idx)

        return color

    def greedy_coloring(edges, conflicts, order, masks):
        """Greedy coloring in given order using bitmask conflicts."""
        n_edges = len(edges)
        color = [-1] * n_edges
        color_masks = [0] * (n_edges + 1)

        for e_idx in order:
            c = 0
            while (color_masks[c] & masks[e_idx]):
                c += 1
            color[e_idx] = c
            color_masks[c] |= (1 << e_idx)

        return color

    def local_search(edges, conflicts, color, time_budget, masks):
        """Local search with move-to-lower-color, 2-swap, and repack moves using bitmasks."""
        n_edges = len(edges)
        color = list(color)
        color_count = max(color) + 1 if color else 0
        rng = random.Random(42)

        # Build color class bitmasks for fast lookup
        color_class_masks = [0] * (n_edges + 1)
        for i, c in enumerate(color):
            color_class_masks[c] |= (1 << i)

        start = time.time()
        max_iterations = 2000

        for iteration in range(max_iterations):
            if time.time() - start > time_budget:
                break
            if time.time() > deadline:
                break

            improved = False
            order = list(range(n_edges))
            rng.shuffle(order)

            # Phase 1: try to move each edge to a lower colour
            for e_idx in order:
                c_current = color[e_idx]
                if c_current == 0:
                    continue

                for new_c in range(c_current):
                    if not (color_class_masks[new_c] & masks[e_idx]):
                        color_class_masks[c_current] &= ~(1 << e_idx)
                        color_class_masks[new_c] |= (1 << e_idx)
                        color[e_idx] = new_c
                        improved = True
                        if c_current == color_count - 1 and color_class_masks[c_current] == 0:
                            color_count -= 1
                        break

            # Phase 2: aggressive 2-swap to eliminate top color
            if color_count > 1:
                top_color = color_count - 1
                top_mask = color_class_masks[top_color]
                if top_mask:
                    top_edges = []
                    m = top_mask
                    while m:
                        low = m & (-m)
                        e1 = low.bit_length() - 1
                        top_edges.append(e1)
                        m ^= low

                    for e1 in top_edges:
                        if time.time() - start > time_budget or time.time() > deadline:
                            break
                        c1 = color[e1]
                        if c1 != top_color:
                            continue
                        # Try swapping with every edge in every lower color
                        for c2 in range(top_color):
                            if c2 == 0:
                                continue
                            # Get edges in color c2
                            c2_mask = color_class_masks[c2]
                            if not c2_mask:
                                continue
                            m2 = c2_mask
                            while m2:
                                low2 = m2 & (-m2)
                                e2 = low2.bit_length() - 1
                                m2 ^= low2
                                # Check if e1 can move to c2 (excluding e2)
                                if not ((color_class_masks[c2] & ~(1 << e2)) & masks[e1]):
                                    # Check if e2 can move to c1 (excluding e1)
                                    if not ((color_class_masks[c1] & ~(1 << e1)) & masks[e2]):
                                        color_class_masks[c1] &= ~(1 << e1)
                                        color_class_masks[c2] &= ~(1 << e2)
                                        color_class_masks[c2] |= (1 << e1)
                                        color_class_masks[c1] |= (1 << e2)
                                        color[e1] = c2
                                        color[e2] = c1
                                        improved = True
                                        if color_class_masks[top_color] == 0:
                                            color_count -= 1
                                        break
                            if color_class_masks[top_color] == 0:
                                break

            # Phase 3: repack – rebuild coloring greedily in current edge order
            # This can sometimes reduce color count by finding better packing
            if color_count > 2 and iteration % 3 == 2:
                new_color = [-1] * n_edges
                new_color_masks = [0] * (color_count + 1)
                # Process edges in order of their current color (lowest first)
                for c in range(color_count):
                    cm = color_class_masks[c]
                    if not cm:
                        continue
                    m = cm
                    while m:
                        low = m & (-m)
                        e_idx = low.bit_length() - 1
                        m ^= low
                        # Assign to smallest available color
                        nc = 0
                        while (new_color_masks[nc] & masks[e_idx]):
                            nc += 1
                        if nc >= color_count:
                            nc = color_count - 1
                            # Force: find any valid color
                            for fc in range(color_count):
                                if not (new_color_masks[fc] & masks[e_idx]):
                                    nc = fc
                                    break
                        new_color[e_idx] = nc
                        new_color_masks[nc] |= (1 << e_idx)
                
                new_count = 0
                for nc in new_color:
                    if nc + 1 > new_count:
                        new_count = nc + 1
                
                if new_count < color_count:
                    color = new_color
                    color_count = new_count
                    color_class_masks = [0] * (n_edges + 1)
                    for i, c in enumerate(color):
                        color_class_masks[c] |= (1 << i)
                    improved = True

            if not improved:
                break

        return color, color_count

    def construct_induced_matching_coloring(edges, conflicts, masks, n_edges, time_budget):
        """
        Constructs a coloring by iteratively finding large induced matchings.
        This is a structural change: instead of coloring edges one by one,
        we form color classes (induced matchings) one by one.
        """
        remaining = set(range(n_edges))
        color = [-1] * n_edges
        current_color = 0
        rng = random.Random(42)
        
        start_time = time.time()
        
        while remaining:
            if time.time() - start_time > time_budget:
                # Fallback: assign remaining edges greedily to new colors
                for e in remaining:
                    color[e] = current_color
                    current_color += 1
                break
            
            # Try to find a large induced matching from 'remaining'
            # Use multiple random orders to find a good one
            best_matching = []
            # Adaptive number of trials based on time
            trials = 30
            
            for _ in range(trials):
                candidates = list(remaining)
                rng.shuffle(candidates)
                
                selected = []
                forbidden = set()
                
                for e in candidates:
                    if e in forbidden:
                        continue
                    selected.append(e)
                    forbidden.add(e)
                    # Add all conflicts of e to forbidden
                    m = masks[e]
                    while m:
                        low = m & (-m)
                        j = low.bit_length() - 1
                        forbidden.add(j)
                        m ^= low
                
                if len(selected) > len(best_matching):
                    best_matching = selected
            
            if not best_matching:
                # Should not happen if remaining is not empty
                best_matching = [next(iter(remaining))]
                
            # Assign current color to best_matching
            for e in best_matching:
                color[e] = current_color
                remaining.remove(e)
                
            current_color += 1
            
        return color, current_color

    def optimize_instance(n, d, seed, time_budget):
        """Optimize coloring for a single instance within time budget."""
        edges, degree = get_instance(n, d, seed)
        adj = build_adjacency(edges)
        conflicts = precompute_conflicts(edges, adj)
        masks = conflicts_to_bitmask(conflicts, len(edges))
        n_edges = len(edges)

        best_color = None
        best_count = float('inf')

        elapsed_start = time.time()

        # 1. DSATUR initial coloring
        dsatur_color = dsatur_coloring(edges, conflicts, masks)
        dsatur_count = max(dsatur_color) + 1 if dsatur_color else 0
        if dsatur_count < best_count:
            best_color = list(dsatur_color)
            best_count = dsatur_count

        # 2. Induced Matching Construction (New Structural Method)
        remaining = time_budget - (time.time() - elapsed_start)
        if remaining > 0.5:
            im_color, im_count = construct_induced_matching_coloring(edges, conflicts, masks, n_edges, remaining * 0.2)
            if im_count < best_count:
                best_color = list(im_color)
                best_count = im_count

        # 3. Local Search on best so far
        remaining = time_budget - (time.time() - elapsed_start)
        if remaining > 0.5 and best_color is not None:
            ls_color, ls_count = local_search(edges, conflicts, best_color, remaining * 0.4, masks)
            if ls_count < best_count:
                best_color = list(ls_color)
                best_count = ls_count

        # 4. Try multiple greedy orderings with local search
        remaining = time_budget - (time.time() - elapsed_start)
        n_restarts = max(5, int(remaining * 5))

        # Precompute orderings
        orderings = []
        # High-conflict-first
        conflict_counts = [masks[i].bit_count() for i in range(n_edges)]
        orderings.append(sorted(range(n_edges), key=lambda i: conflict_counts[i], reverse=True))
        # Low-conflict-first
        orderings.append(sorted(range(n_edges), key=lambda i: conflict_counts[i]))
        # Degree-based
        edge_degrees = [degree[u] + degree[v] for u, v in edges]
        orderings.append(sorted(range(n_edges), key=lambda i: edge_degrees[i], reverse=True))
        # Random shuffles with different seeds
        for s in range(n_restarts):
            rng = random.Random(seed + 1000 + s)
            shuffled = list(range(n_edges))
            rng.shuffle(shuffled)
            orderings.append(shuffled)

        for ordering in orderings:
            if time.time() > deadline:
                break
            if time.time() - elapsed_start > time_budget:
                break

            g_color = greedy_coloring(edges, conflicts, ordering, masks)
            g_count = max(g_color) + 1 if g_color else 0

            if g_count < best_count:
                best_color = list(g_color)
                best_count = g_count

            # Local search on this greedy solution
            remaining = time_budget - (time.time() - elapsed_start)
            if remaining > 0.3 and best_count > 1:
                ls_color, ls_count = local_search(edges, conflicts, g_color, remaining * 0.3, masks)
                if ls_count < best_count:
                    best_color = list(ls_color)
                    best_count = ls_count

        # Final polish: run local search on best solution with remaining time
        remaining = time_budget - (time.time() - elapsed_start)
        if remaining > 0.2 and best_color is not None:
            ls_color, ls_count = local_search(edges, conflicts, best_color, remaining * 0.85, masks)
            if ls_count < best_count:
                best_color = list(ls_color)
                best_count = ls_count

        if best_color is None:
            best_color = greedy_coloring(edges, conflicts, list(range(n_edges)), masks)
            best_count = max(best_color) + 1 if best_color else 0

        return best_color, best_count

    results = []
    instances = [(40, 4, 101), (60, 5, 102), (80, 6, 103)]
    remaining_time = deadline - time.time()

    for idx, (n, d, seed) in enumerate(instances):
        if time.time() > deadline:
            # Emergency fallback
            edges, _ = get_instance(n, d, seed)
            adj = build_adjacency(edges)
            conflicts = precompute_conflicts(edges, adj)
            masks = conflicts_to_bitmask(conflicts, len(edges))
            color = greedy_coloring(edges, conflicts, list(range(len(edges))), masks)
            results.append(color)
            continue

        time_budget = remaining_time / (len(instances) - idx)
        coloring, _ = optimize_instance(n, d, seed, time_budget)
        results.append(coloring)
        remaining_time = deadline - time.time()

    return results

# EVOLVE-BLOCK-END
