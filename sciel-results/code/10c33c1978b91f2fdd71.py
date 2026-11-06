import random
import time
import itertools
import sys

def construct():
    n = 20
    num_colors = 20
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill

    # Precompute all edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    E = len(all_edges)
    edge_index = {}
    for i, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i

    # Precompute all 4-cycles (as tuples of 4 edge indices)
    cycles = []
    edge_to_cycles = [[] for _ in range(E)]

    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        # 3 distinct 4-cycles on vertices {a,b,c,d}
        # Cycle 1: a-b, b-c, c-d, d-a
        e1 = edge_index.get((a, b)); e2 = edge_index.get((b, c))
        e3 = edge_index.get((c, d)); e4 = edge_index.get((d, a))
        if None not in (e1, e2, e3, e4):
            cyc = (e1, e2, e3, e4)
            cycles.append(cyc)
            for e in cyc:
                edge_to_cycles[e].append(len(cycles) - 1)
        # Cycle 2: a-b, b-d, d-c, c-a
        e1 = edge_index.get((a, b)); e2 = edge_index.get((b, d))
        e3 = edge_index.get((d, c)); e4 = edge_index.get((c, a))
        if None not in (e1, e2, e3, e4):
            cyc = (e1, e2, e3, e4)
            cycles.append(cyc)
            for e in cyc:
                edge_to_cycles[e].append(len(cycles) - 1)
        # Cycle 3: a-c, c-b, b-d, d-a
        e1 = edge_index.get((a, c)); e2 = edge_index.get((c, b))
        e3 = edge_index.get((b, d)); e4 = edge_index.get((d, a))
        if None not in (e1, e2, e3, e4):
            cyc = (e1, e2, e3, e4)
            cycles.append(cyc)
            for e in cyc:
                edge_to_cycles[e].append(len(cycles) - 1)

    C = len(cycles)
    # Precompute for each cycle, the 4 edges and for each position, the other 3
    # cycle_other[cyc_idx][pos] = (other1, other2, other3)
    cycle_other = []
    for cyc in cycles:
        others = []
        for pos in range(4):
            o = [e for i, e in enumerate(cyc) if i != pos]
            others.append(tuple(o))
        cycle_other.append(others)

    # Also precompute: for each edge, list of (cycle_idx, position)
    edge_pos = [[] for _ in range(E)]
    for c_idx, cyc in enumerate(cycles):
        for pos, e in enumerate(cyc):
            edge_pos[e].append((c_idx, pos))

    # --- Backtracking with forward checking ---
    def backtrack_solve(order, max_depth):
        """Try to build a large valid subgraph via backtracking on edge order.
        Returns (edge_colors, edge_count) where edge_colors[e]=-1 if absent."""
        edge_colors = [-1] * E
        edge_count = 0
        # Track forbidden colors per edge due to complete cycles
        # forbidden[e] = set of colors that e cannot take
        forbidden = [set() for _ in range(E)]

        def check_add(e_idx, color):
            """Check if adding edge e_idx with color is consistent."""
            for c_idx, pos in edge_pos[e_idx]:
                o1, o2, o3 = cycle_other[c_idx][pos]
                c1 = edge_colors[o1]; c2 = edge_colors[o2]; c3 = edge_colors[o3]
                if c1 == -1 or c2 == -1 or c3 == -1:
                    continue
                if c1 == c2 or c2 == c3 or c1 == c3:
                    return False
                if color == c1 or color == c2 or color == c3:
                    return False
            return True

        def add_edge(e_idx, color):
            nonlocal edge_count
            edge_colors[e_idx] = color
            edge_count += 1
            # Forward checking: update forbidden sets for edges in cycles with e_idx
            for c_idx, pos in edge_pos[e_idx]:
                o1, o2, o3 = cycle_other[c_idx][pos]
                for oe in (o1, o2, o3):
                    if edge_colors[oe] != -1:
                        # This cycle is now complete; check the other 3 colors
                        cols = [edge_colors[x] for x in (o1, o2, o3) if x != oe]
                        # Actually we need the 3 colors of the OTHER edges in this cycle
                        # The cycle has edges: e_idx (just added) and o1,o2,o3
                        # For oe, the other 3 are e_idx and the two of o1,o2,o3 not equal to oe
                        all3 = [e_idx, o1, o2, o3]
                        others_of_oe = [x for x in all3 if x != oe]
                        cvals = [edge_colors[x] for x in others_of_oe]
                        if -1 not in cvals:
                            if cvals[0] == cvals[1] or cvals[1] == cvals[2] or cvals[0] == cvals[2]:
                                # Conflict! This cycle has duplicate colors among present edges
                                # But wait - e_idx was just checked valid, so this shouldn't happen
                                # unless the cycle already had a conflict. Skip.
                                pass
                            else:
                                forbidden[oe].update(cvals)

        def remove_edge(e_idx):
            nonlocal edge_count
            color = edge_colors[e_idx]
            edge_colors[e_idx] = -1
            edge_count -= 1
            # Undo forward checking
            for c_idx, pos in edge_pos[e_idx]:
                o1, o2, o3 = cycle_other[c_idx][pos]
                for oe in (o1, o2, o3):
                    if edge_colors[oe] != -1:
                        all3 = [e_idx, o1, o2, o3]
                        others_of_oe = [x for x in all3 if x != oe]
                        cvals = [edge_colors[x] for x in others_of_oe]
                        if -1 not in cvals:
                            forbidden[oe].difference_update(cvals)

        # Recursive backtrack
        sys.setrecursionlimit(10000)

        def bt(depth):
            nonlocal edge_count
            if depth >= len(order) or time.time() > deadline - 3:
                return edge_count

            e_idx = order[depth]

            # Try adding with various colors
            # Choose colors: try all, but prune if forbidden covers all
            allowed = [c for c in range(num_colors) if c not in forbidden[e_idx]]
            random.shuffle(allowed)

            best_sub = 0
            best_state = None

            for color in allowed:
                if time.time() > deadline - 3:
                    break
                if not check_add(e_idx, color):
                    continue
                # Save state
                saved_forbidden = [set() for _ in range(E)]
                for i in range(E):
                    saved_forbidden[i] = set(forbidden[i])

                add_edge(e_idx, color)
                sub = bt(depth + 1)
                if sub > best_sub:
                    best_sub = sub
                    best_state = (e_idx, color)
                # Undo
                remove_edge(e_idx)
                for i in range(E):
                    forbidden[i] = saved_forbidden[i]

            # Option: skip this edge
            skip_sub = bt(depth + 1)
            if skip_sub > best_sub:
                best_sub = skip_sub
                best_state = None

            return best_sub

        bt(0)
        return edge_colors, edge_count

    # --- Local search (simulated annealing) ---
    def local_search(edge_colors, edge_count, time_budget):
        """Improve a valid colored graph via local search.
        Moves: recolor an edge, add an edge, remove an edge."""
        # Build current state
        colors = list(edge_colors)
        count = edge_count
        best_colors = list(colors)
        best_count = count

        # Track forbidden per edge
        forbidden = [set() for _ in range(E)]
        for c_idx, cyc in enumerate(cycles):
            c0, c1, c2, c3 = cyc
            cols = [colors[e] for e in cyc]
            if -1 not in cols:
                if cols[0] == cols[1] or cols[1] == cols[2] or cols[0] == cols[2]:
                    # Invalid state, shouldn't happen
                    pass
                else:
                    for e in cyc:
                        forbidden[e].update(cols)

        def is_valid_recolor(e_idx, new_color):
            for c_idx, pos in edge_pos[e_idx]:
                o1, o2, o3 = cycle_other[c_idx][pos]
                c1 = colors[o1]; c2 = colors[o2]; c3 = colors[o3]
                if c1 == -1 or c2 == -1 or c3 == -1:
                    continue
                if c1 == c2 or c2 == c3 or c1 == c3:
                    return False
                if new_color == c1 or new_color == c2 or new_color == c3:
                    return False
            return True

        def is_valid_add(e_idx, new_color):
            if colors[e_idx] != -1:
                return False
            return is_valid_recolor(e_idx, new_color)

        end_time = time.time() + time_budget
        temp = 5.0
        cooling = 0.999

        present_edges = [i for i in range(E) if colors[i] != -1]
        absent_edges = [i for i in range(E) if colors[i] == -1]

        while time.time() < end_time:
            if temp < 0.1:
                temp = 0.1
            temp *= cooling

            # Choose a random move
            move_type = random.random()
            if move_type < 0.4 and present_edges:
                # Recolor
                e_idx = random.choice(present_edges)
                old_color = colors[e_idx]
                allowed = [c for c in range(num_colors) if c not in forbidden[e_idx] and c != old_color]
                if not allowed:
                    continue
                new_color = random.choice(allowed)
                if not is_valid_recolor(e_idx, new_color):
                    continue
                # Apply
                colors[e_idx] = new_color
                # Update forbidden
                for c_idx, pos in edge_pos[e_idx]:
                    o1, o2, o3 = cycle_other[c_idx][pos]
                    c1 = colors[o1]; c2 = colors[o2]; c3 = colors[o3]
                    if -1 not in (c1, c2, c3):
                        if c1 == c2 or c2 == c3 or c1 == c3:
                            pass
                        else:
                            for oe in (o1, o2, o3):
                                others = [x for x in (e_idx, o1, o2, o3) if x != oe]
                                cvals = [colors[x] for x in others]
                                if -1 not in cvals:
                                    forbidden[oe].update(cvals)
                # Delta is 0 (same edge count)
                delta = 0
                if count > best_count:
                    best_count = count
                    best_colors = list(colors)
                # Always accept (neutral move)
                # Undo
                colors[e_idx] = old_color
                for c_idx, pos in edge_pos[e_idx]:
                    o1, o2, o3 = cycle_other[c_idx][pos]
                    c1 = colors[o1]; c2 = colors[o2]; c3 = colors[o3]
                    if -1 not in (c1, c2, c3):
                        for oe in (o1, o2, o3):
                            others = [x for x in (e_idx, o1, o2, o3) if x != oe]
                            cvals = [colors[x] for x in others]
                            if -1 not in cvals:
                                forbidden[oe].difference_update(cvals)

            elif move_type < 0.8 and absent_edges:
                # Add edge
                e_idx = random.choice(absent_edges)
                allowed = [c for c in range(num_colors) if c not in forbidden[e_idx]]
                if not allowed:
                    continue
                new_color = random.choice(allowed)
                if not is_valid_add(e_idx, new_color):
                    continue
                # Apply
                colors[e_idx] = new_color
                count += 1
                present_edges.append(e_idx)
                absent_edges.remove(e_idx)
                # Update forbidden
                for c_idx, pos in edge_pos[e_idx]:
                    o1, o2, o3 = cycle_other[c_idx][pos]
                    c1 = colors[o1]; c2 = colors[o2]; c3 = colors[o3]
                    if -1 not in (c1, c2, c3):
                        for oe in (o1, o2, o3):
                            others = [x for x in (e_idx, o1, o2, o3) if x != oe]
                            cvals = [colors[x] for x in others]
                            if -1 not in cvals:
                                forbidden[oe].update(cvals)
                delta = 1
                if count > best_count:
                    best_count = count
                    best_colors = list(colors)
                # Accept if improving or with probability
                if delta <= 0 or random.random() < min(1.0, 1.0 / (1.0 + delta)):
                    # Keep
                    pass
                else:
                    # Undo
                    colors[e_idx] = -1
                    count -= 1
                    present_edges.remove(e_idx)
                    absent_edges.append(e_idx)
                    for c_idx, pos in edge_pos[e_idx]:
                        o1, o2, o3 = cycle_other[c_idx][pos]
                        c1 = colors[o1]; c2 = colors[o2]; c3 = colors[o3]
                        if -1 not in (c1, c2, c3):
                            for oe in (o1, o2, o3):
                                others = [x for x in (e_idx, o1, o2, o3) if x != oe]
                                cvals = [colors[x] for x in others]
                                if -1 not in cvals:
                                    forbidden[oe].difference_update(cvals)
            else:
                # Remove edge (rarely useful but helps escape)
                if present_edges:
                    e_idx = random.choice(present_edges)
                    old_color = colors[e_idx]
                    colors[e_idx] = -1
                    count -= 1
                    present_edges.remove(e_idx)
                    absent_edges.append(e_idx)
                    for c_idx, pos in edge_pos[e_idx]:
                        o1, o2, o3 = cycle_other[c_idx][pos]
                        c1 = colors[o1]; c2 = colors[o2]; c3 = colors[o3]
                        if -1 not in (c1, c2, c3):
                            for oe in (o1, o2, o3):
                                others = [x for x in (e_idx, o1, o2, o3) if x != oe]
                                cvals = [colors[x] for x in others]
                                if -1 not in cvals:
                                    forbidden[oe].difference_update(cvals)
                    delta = -1
                    # Always accept removal if it allows future additions (heuristic)
                    if random.random() < 0.3:
                        # Keep removal
                        pass
                    else:
                        # Undo
                        colors[e_idx] = old_color
                        count += 1
                        present_edges.append(e_idx)
                        absent_edges.remove(e_idx)
                        for c_idx, pos in edge_pos[e_idx]:
                            o1, o2, o3 = cycle_other[c_idx][pos]
                            c1 = colors[o1]; c2 = colors[o2]; c3 = colors[o3]
                            if -1 not in (c1, c2, c3):
                                for oe in (o1, o2, o3):
                                    others = [x for x in (e_idx, o1, o2, o3) if x != oe]
                                    cvals = [colors[x] for x in others]
                                    if -1 not in cvals:
                                        forbidden[oe].update(cvals)

        return best_colors, best_count

    # --- Main search ---
    best_edge_count = 0
    best_colors = None

    # Phase 1: Backtracking with different edge orderings
    edge_indices = list(range(E))

    # Try a few different orderings for backtracking
    max_bt_restarts = 30
    for restart in range(max_bt_restarts):
        if time.time() > deadline - 30:
            break

        random.shuffle(edge_indices)
        # Limit depth to keep it tractable
        max_depth = min(E, 100)
        order = edge_indices[:max_depth]

        colors, count = backtrack_solve(order, max_depth)
        if count > best_edge_count:
            best_edge_count = count
            best_colors = list(colors)

        # Local search from this result
        if time.time() < deadline - 20:
            ls_colors, ls_count = local_search(colors, count, min(5.0, deadline - 20 - time.time()))
            if ls_count > best_edge_count:
                best_edge_count = ls_count
                best_colors = list(ls_colors)

    # Phase 2: More greedy + local search for diversity
    if time.time() < deadline - 15:
        for restart in range(200):
            if time.time() > deadline - 15:
                break

            random.shuffle(edge_indices)
            edge_colors = [-1] * E
            edge_count = 0
            forbidden = [set() for _ in range(E)]

            for e_idx in edge_indices:
                allowed = [c for c in range(num_colors) if c not in forbidden[e_idx]]
                if not allowed:
                    continue
                # Check validity
                valid = True
                for c_idx, pos in edge_pos[e_idx]:
                    o1, o2, o3 = cycle_other[c_idx][pos]
                    c1 = edge_colors[o1]; c2 = edge_colors[o2]; c3 = edge_colors[o3]
                    if -1 not in (c1, c2, c3):
                        if c1 == c2 or c2 == c3 or c1 == c3:
                            valid = False
                            break
                        for ac in allowed:
                            if ac == c1 or ac == c2 or ac == c3:
                                allowed.remove(ac)
                                break
                        if not allowed:
                            valid = False
                            break
                if not valid:
                    continue
                chosen = random.choice(allowed)
                edge_colors[e_idx] = chosen
                edge_count += 1
                # Update forbidden
                for c_idx, pos in edge_pos[e_idx]:
                    o1, o2, o3 = cycle_other[c_idx][pos]
                    c1 = edge_colors[o1]; c2 = edge_colors[o2]; c3 = edge_colors[o3]
                    if -1 not in (c1, c2, c3):
                        for oe in (o1, o2, o3):
                            others = [x for x in (e_idx, o1, o2, o3) if x != oe]
                            cvals = [edge_colors[x] for x in others]
                            if -1 not in cvals:
                                forbidden[oe].update(cvals)

            if edge_count > best_edge_count:
                best_edge_count = edge_count
                best_colors = list(edge_colors)

            # Local search
            if time.time() < deadline - 10 and edge_count > 0:
                ls_colors, ls_count = local_search(edge_colors, edge_count, min(3.0, deadline - 10 - time.time()))
                if ls_count > best_edge_count:
                    best_edge_count = ls_count
                    best_colors = list(ls_colors)

    if best_colors is None:
        return []

    result = []
    for i in range(E):
        if best_colors[i] != -1:
            u, v = all_edges[i]
            result.append([u, v, best_colors[i]])

    return result
