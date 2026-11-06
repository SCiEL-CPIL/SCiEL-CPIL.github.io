import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n = 20
    n_colors = 20
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)

    # Precompute all 4-cycles as tuples of 4 edge indices
    edge_to_idx = {}
    for i, (u, v) in enumerate(all_edges):
        edge_to_idx[(u, v)] = i
        edge_to_idx[(v, u)] = i

    all_4cycles = []
    for quad in combinations(range(n), 4):
        w, x, y, z = quad
        cycles = [
            [(w, x), (x, y), (y, z), (z, w)],
            [(w, x), (x, z), (z, y), (y, w)],
            [(w, y), (y, x), (x, z), (z, w)],
        ]
        for cyc in cycles:
            try:
                edge_idxs = tuple(sorted(edge_to_idx[e] for e in cyc))
                all_4cycles.append(edge_idxs)
            except KeyError:
                pass

    # For fast checking: for each edge, list of 4-cycles containing it
    edge_cycles = [[] for _ in range(num_edges)]
    for ci, cyc in enumerate(all_4cycles):
        for ei in cyc:
            edge_cycles[ei].append(ci)

    edge_degree = [len(edge_cycles[ei]) for ei in range(num_edges)]

    def can_add_edge(ei, c, edges_set, colors):
        """Check if adding edge ei with color c is valid."""
        for ci in edge_cycles[ei]:
            cyc = all_4cycles[ci]
            if all(e2 in edges_set for e2 in cyc if e2 != ei):
                cs = [colors.get(e2) for e2 in cyc if e2 != ei] + [c]
                if len(set(cs)) != 4:
                    return False
        return True

    def try_add_edge(ei, edges_set, colors):
        """Try to add edge ei with any valid color. Returns True if added."""
        valid_colors = list(range(n_colors))
        random.shuffle(valid_colors)
        for c in valid_colors:
            if can_add_edge(ei, c, edges_set, colors):
                edges_set.add(ei)
                colors[ei] = c
                return True
        return False

    def greedy_build():
        """Greedy edge addition with edge ordering biased towards high-degree edges."""
        edges_set = set()
        colors = {}
        remaining = list(range(num_edges))
        random.shuffle(remaining)
        remaining.sort(key=lambda ei: -edge_degree[ei] + random.uniform(-5, 5))

        for ei in remaining:
            if time.time() > deadline:
                break
            try_add_edge(ei, edges_set, colors)
        return edges_set, colors

    def local_search(edges_set, colors):
        """
        Local search: try to add missing edges.
        If an edge cannot be added, try to re-color existing edges to make it possible.
        """
        missing = [ei for ei in range(num_edges) if ei not in edges_set]
        random.shuffle(missing)

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            random.shuffle(missing)

            for ei in missing:
                if time.time() > deadline - 2.0:
                    break
                if ei in edges_set:
                    continue

                if try_add_edge(ei, edges_set, colors):
                    improved = True
                    missing.remove(ei)
                    continue

                conflicting_cycles = []
                for ci in edge_cycles[ei]:
                    cyc = all_4cycles[ci]
                    if all(e2 in edges_set for e2 in cyc if e2 != ei):
                        conflicting_cycles.append(cyc)

                if not conflicting_cycles:
                    continue

                random.shuffle(conflicting_cycles)
                for target_cyc in conflicting_cycles[:3]:
                    if time.time() > deadline - 2.0:
                        break
                    other_edges = [e2 for e2 in target_cyc if e2 != ei]
                    random.shuffle(other_edges)

                    for e_change in other_edges:
                        if time.time() > deadline - 2.0:
                            break
                        old_color = colors[e_change]
                        possible_colors = [c for c in range(n_colors) if c != old_color]
                        random.shuffle(possible_colors)

                        for new_c in possible_colors:
                            valid = True
                            for ci in edge_cycles[e_change]:
                                cyc = all_4cycles[ci]
                                if all(e2 in edges_set for e2 in cyc):
                                    cs = [colors.get(e2) for e2 in cyc if e2 != e_change] + [new_c]
                                    if len(set(cs)) != 4:
                                        valid = False
                                        break

                            if not valid:
                                continue

                            colors[e_change] = new_c

                            added = False
                            valid_colors_for_ei = list(range(n_colors))
                            random.shuffle(valid_colors_for_ei)
                            for c_ei in valid_colors_for_ei:
                                if can_add_edge(ei, c_ei, edges_set, colors):
                                    edges_set.add(ei)
                                    colors[ei] = c_ei
                                    added = True
                                    break

                            if added:
                                improved = True
                                missing.remove(ei)
                                break
                            else:
                                colors[e_change] = old_color

                        if improved:
                            break

                    if improved:
                        break

    def edge_swap_pass(edges_set, colors):
        """
        Try to remove an edge and add different edges to escape local optima.
        """
        if time.time() > deadline - 3.0:
            return edges_set, colors

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            present = list(edges_set)
            random.shuffle(present)

            for ei_remove in present[:10]:
                if time.time() > deadline - 2.0:
                    break

                removed_color = colors[ei_remove]
                edges_set.remove(ei_remove)
                del colors[ei_remove]

                missing = [ei for ei in range(num_edges) if ei not in edges_set]
                random.shuffle(missing)

                added_any = False
                for ei_add in missing[:20]:
                    if time.time() > deadline - 2.0:
                        break
                    if try_add_edge(ei_add, edges_set, colors):
                        added_any = True

                if added_any:
                    improved = True
                else:
                    edges_set.add(ei_remove)
                    colors[ei_remove] = removed_color

                if improved:
                    break

        return edges_set, colors

    def two_opt_recolor(edges_set, colors):
        """
        Try to swap colors of two edges to enable adding more edges.
        """
        if time.time() > deadline - 3.0:
            return edges_set, colors

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            present = list(edges_set)
            random.shuffle(present)
            missing = [ei for ei in range(num_edges) if ei not in edges_set]
            random.shuffle(missing)

            for ei_add in missing[:5]:
                if time.time() > deadline - 2.0:
                    break
                if ei_add in edges_set:
                    continue

                # Find conflicting cycles
                conflicting_cycles = []
                for ci in edge_cycles[ei_add]:
                    cyc = all_4cycles[ci]
                    if all(e2 in edges_set for e2 in cyc if e2 != ei_add):
                        conflicting_cycles.append(cyc)

                if not conflicting_cycles:
                    continue

                # Try swapping colors of two edges in a conflicting cycle
                random.shuffle(conflicting_cycles)
                for target_cyc in conflicting_cycles[:2]:
                    if time.time() > deadline - 2.0:
                        break
                    other_edges = [e2 for e2 in target_cyc if e2 != ei_add]
                    if len(other_edges) < 2:
                        continue

                    for i in range(len(other_edges)):
                        for j in range(i + 1, len(other_edges)):
                            e1 = other_edges[i]
                            e2 = other_edges[j]
                            c1 = colors[e1]
                            c2 = colors[e2]

                            # Swap colors
                            colors[e1] = c2
                            colors[e2] = c1

                            added = False
                            valid_colors_for_ei = list(range(n_colors))
                            random.shuffle(valid_colors_for_ei)
                            for c_ei in valid_colors_for_ei:
                                if can_add_edge(ei_add, c_ei, edges_set, colors):
                                    edges_set.add(ei_add)
                                    colors[ei_add] = c_ei
                                    added = True
                                    break

                            if added:
                                improved = True
                                break
                            else:
                                # Revert swap
                                colors[e1] = c1
                                colors[e2] = c2

                        if improved:
                            break

                    if improved:
                        break

                if improved:
                    break

    # Run multiple greedy restarts to find a good starting point
    best_edges_set = set()
    best_colors = {}
    best_count = 0

    # Phase 1: Greedy Builds
    num_restarts = 80
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        edges_set, colors = greedy_build()
        if len(edges_set) > best_count:
            best_count = len(edges_set)
            best_edges_set = edges_set.copy()
            best_colors = colors.copy()

    # Phase 2: Local Search on the best solution
    if time.time() < deadline - 5.0:
        local_search(best_edges_set, best_colors)
        best_count = len(best_edges_set)

    # Phase 2.5: Edge swap pass
    if time.time() < deadline - 4.0:
        edge_swap_pass(best_edges_set, best_colors)
        best_count = len(best_edges_set)

    # Phase 2.6: Two-opt recolor
    if time.time() < deadline - 4.0:
        two_opt_recolor(best_edges_set, best_colors)
        best_count = len(best_edges_set)

    # Phase 3: More restarts with local search if time permits
    if time.time() < deadline - 5.0:
        for r in range(80):
            if time.time() > deadline - 2.0:
                break
            edges_set, colors = greedy_build()
            local_search(edges_set, colors)
            if len(edges_set) > best_count:
                best_count = len(edges_set)
                best_edges_set = edges_set.copy()
                best_colors = colors.copy()

    # Final polish
    if time.time() < deadline - 2.0:
        local_search(best_edges_set, best_colors)
        edge_swap_pass(best_edges_set, best_colors)
        two_opt_recolor(best_edges_set, best_colors)
        best_count = len(best_edges_set)

    # Build result
    result = []
    for ei in best_edges_set:
        u, v = all_edges[ei]
        c = best_colors[ei]
        result.append([u, v, c])

    return result
