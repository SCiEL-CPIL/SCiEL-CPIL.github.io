import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    n = 20
    n_colors = 20
    # All possible edges
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
        # 3 distinct 4-cycles
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

    num_4cycles = len(all_4cycles)

    # For fast checking: for each edge, list of 4-cycles containing it
    edge_cycles = [[] for _ in range(num_edges)]
    for ci, cyc in enumerate(all_4cycles):
        for ei in cyc:
            edge_cycles[ei].append(ci)

    # Edge degree in 4-cycle hypergraph (for greedy ordering)
    edge_degree = [len(edge_cycles[ei]) for ei in range(num_edges)]

    # State:
    # edges_set: set of edge indices present in the graph
    # colors: dict mapping edge index to color

    def can_add_edge(ei, c, edges_set, colors):
        """Check if adding edge ei with color c is valid."""
        for ci in edge_cycles[ei]:
            cyc = all_4cycles[ci]
            # Check if the other 3 edges are present
            other_present = all(e2 in edges_set for e2 in cyc if e2 != ei)
            if other_present:
                cs = [colors[e2] for e2 in cyc if e2 != ei]
                if c in cs:
                    return False
        return True

    def get_valid_colors(ei, edges_set, colors):
        """Get all valid colors for edge ei."""
        valid = []
        for c in range(n_colors):
            if can_add_edge(ei, c, edges_set, colors):
                valid.append(c)
        return valid

    def try_add_edge(ei, edges_set, colors):
        """Try to add edge ei with any valid color. Returns True if added."""
        valid_colors = get_valid_colors(ei, edges_set, colors)
        if valid_colors:
            c = random.choice(valid_colors)
            edges_set.add(ei)
            colors[ei] = c
            return True
        return False

    def greedy_build():
        """Greedy edge addition with edge ordering biased towards high-degree edges."""
        edges_set = set()
        colors = {}
        remaining = list(range(num_edges))
        # Sort by degree with some randomness to break ties
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
            # Shuffle missing again to avoid bias
            random.shuffle(missing)

            for ei in missing:
                if time.time() > deadline - 2.0:
                    break
                if ei in edges_set:
                    continue

                # First, try simple addition
                if try_add_edge(ei, edges_set, colors):
                    improved = True
                    missing.remove(ei)
                    continue

                # If simple addition failed, try to fix conflicts by re-coloring
                # Identify conflicting cycles
                conflicting_cycles = []
                for ci in edge_cycles[ei]:
                    cyc = all_4cycles[ci]
                    if all(e2 in edges_set for e2 in cyc if e2 != ei):
                        # This cycle is complete except for ei
                        conflicting_cycles.append(cyc)

                if not conflicting_cycles:
                    continue

                # Try multiple conflicting cycles (up to 3) to increase chance of finding a fix
                random.shuffle(conflicting_cycles)
                for target_cyc in conflicting_cycles[:3]:
                    if time.time() > deadline - 2.0:
                        break
                    # The other 3 edges in this cycle
                    other_edges = [e2 for e2 in target_cyc if e2 != ei]
                    random.shuffle(other_edges)

                    for e_change in other_edges:
                        if time.time() > deadline - 2.0:
                            break
                        old_color = colors[e_change]
                        # Try changing e_change to a different color
                        possible_colors = [c for c in range(n_colors) if c != old_color]
                        random.shuffle(possible_colors)

                        for new_c in possible_colors:
                            # Check if changing e_change to new_c is valid for all cycles containing e_change
                            valid = True
                            for ci in edge_cycles[e_change]:
                                cyc = all_4cycles[ci]
                                if all(e2 in edges_set for e2 in cyc):
                                    cs = [colors[e2] for e2 in cyc if e2 != e_change]
                                    if new_c in cs:
                                        valid = False
                                        break

                            if not valid:
                                continue

                            # Temporarily set colors[e_change] = new_c
                            colors[e_change] = new_c

                            added = False
                            valid_colors_for_ei = get_valid_colors(ei, edges_set, colors)
                            if valid_colors_for_ei:
                                c_ei = random.choice(valid_colors_for_ei)
                                edges_set.add(ei)
                                colors[ei] = c_ei
                                added = True

                            if added:
                                improved = True
                                missing.remove(ei)
                                break
                            else:
                                # Revert color change
                                colors[e_change] = old_color

                        if improved:
                            break

                    if improved:
                        break

    def two_opt_recolor(edges_set, colors, max_steps=500):
        """
        Try to improve by re-coloring pairs of edges to free up space for missing edges.
        """
        missing = [ei for ei in range(num_edges) if ei not in edges_set]
        random.shuffle(missing)
        improved_any = False

        for step in range(max_steps):
            if time.time() > deadline - 2.0:
                break
            if not missing:
                break

            ei = missing[0]
            # Try to add ei
            if try_add_edge(ei, edges_set, colors):
                improved_any = True
                missing.remove(ei)
                continue

            # Find a pair of edges to recolor
            # Pick a random edge from the graph and a random missing edge's conflict
            conflicting_cycles = []
            for ci in edge_cycles[ei]:
                cyc = all_4cycles[ci]
                if all(e2 in edges_set for e2 in cyc if e2 != ei):
                    conflicting_cycles.append(cyc)

            if not conflicting_cycles:
                missing.remove(ei)
                continue

            target_cyc = random.choice(conflicting_cycles)
            other_edges = [e2 for e2 in target_cyc if e2 != ei]
            if len(other_edges) < 2:
                continue

            e1, e2 = random.sample(other_edges, 2)
            c1_old = colors[e1]
            c2_old = colors[e2]

            # Try swapping or changing colors
            for new_c1 in range(n_colors):
                if new_c1 == c1_old:
                    continue
                for new_c2 in range(n_colors):
                    if new_c2 == c2_old:
                        continue

                    # Check validity for e1
                    valid1 = True
                    for ci in edge_cycles[e1]:
                        cyc = all_4cycles[ci]
                        if all(x in edges_set for x in cyc):
                            cs = [colors[x] for x in cyc if x not in (e1, e2)]
                            if new_c1 in cs:
                                valid1 = False
                                break
                    if not valid1:
                        continue

                    # Check validity for e2
                    valid2 = True
                    for ci in edge_cycles[e2]:
                        cyc = all_4cycles[ci]
                        if all(x in edges_set for x in cyc):
                            cs = [colors[x] for x in cyc if x not in (e1, e2)]
                            if new_c2 in cs:
                                valid2 = False
                                break
                    if not valid2:
                        continue

                    # Check if e1 and e2 are in a common cycle
                    for ci in edge_cycles[e1]:
                        if e2 in all_4cycles[ci]:
                            cyc = all_4cycles[ci]
                            if all(x in edges_set for x in cyc):
                                cs = [colors[x] for x in cyc if x not in (e1, e2)]
                                if new_c1 == new_c2 or new_c1 in cs or new_c2 in cs:
                                    if new_c1 == new_c2:
                                        valid2 = False
                                        break
                                # Need distinct colors in the cycle
                                if len(set([new_c1, new_c2] + cs)) != len([new_c1, new_c2] + cs):
                                    valid2 = False
                                    break

                    if not valid2:
                        continue

                    colors[e1] = new_c1
                    colors[e2] = new_c2

                    if try_add_edge(ei, edges_set, colors):
                        improved_any = True
                        missing.remove(ei)
                        break
                    else:
                        colors[e1] = c1_old
                        colors[e2] = c2_old

            if time.time() > deadline - 2.0:
                break

        return improved_any

    # Run multiple greedy restarts to find a good starting point
    best_edges_set = set()
    best_colors = {}
    best_count = 0

    # Phase 1: Greedy Builds
    num_restarts = 50
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

    # Phase 3: Two-opt recoloring
    if time.time() < deadline - 3.0:
        two_opt_recolor(best_edges_set, best_colors)
        best_count = len(best_edges_set)

    # Phase 4: More restarts with local search if time permits
    if time.time() < deadline - 5.0:
        for r in range(50):
            if time.time() > deadline - 2.0:
                break
            edges_set, colors = greedy_build()
            # Quick local search on this new candidate
            local_search(edges_set, colors)
            if len(edges_set) > best_count:
                best_count = len(edges_set)
                best_edges_set = edges_set.copy()
                best_colors = colors.copy()
            # Two-opt on the best
            if time.time() < deadline - 2.0:
                two_opt_recolor(best_edges_set, best_colors)
                best_count = len(best_edges_set)

    # Build result
    result = []
    for ei in best_edges_set:
        u, v = all_edges[ei]
        c = best_colors[ei]
        result.append([u, v, c])

    return result
