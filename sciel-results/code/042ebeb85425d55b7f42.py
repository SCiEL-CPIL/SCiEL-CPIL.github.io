import random
import time
import itertools
from collections import defaultdict

import numpy as np
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    n = 20
    num_colors = 20

    # Precompute all 4-cycles and edge-to-cycle mappings
    all_cycles = []
    edge_to_cycles = defaultdict(list)

    for u, v, w, x in itertools.combinations(range(n), 4):
        # Three distinct 4-cycles on 4 vertices
        cycles = [
            [(u, v), (v, w), (w, x), (x, u)],
            [(u, v), (v, x), (x, w), (w, u)],
            [(u, w), (w, v), (v, x), (x, u)],
        ]
        for cycle in cycles:
            norm_cycle = [(min(e), max(e)) for e in cycle]
            if len(set(norm_cycle)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(idx)

    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]

    # ---- Maximum-weight independent set on the conflict graph ----
    # Nodes: (edge, color) pairs.  Two nodes conflict iff they lie in some
    # common 4-cycle and their colours violate the rainbow condition.
    #
    # Conflict types:
    #   (a) same edge, different colour            -> simple graph forbids both
    #   (b) same colour, different edges in a cycle -> duplicate colour in cycle
    #   (c) two different edges in a cycle whose other two edges share a colour
    #       (only relevant when those other edges are also selected)
    #
    # We build the full conflict matrix for types (a) and (b); type (c) is
    # enforced by the LP/branch-and-bound via cycle constraints.

    edge_index = {e: i for i, e in enumerate(all_edges)}
    num_edges = len(all_edges)

    # Flatten: node id = edge_idx * num_colors + color
    total_nodes = num_edges * num_colors

    # Precompute for each edge the set of (other_edge, other_color) that conflict
    # via same-colour-in-cycle (type b).  We store as a list of node ids.
    # For type (a): same edge different color -> num_colors-1 conflicts per node.

    # Build adjacency in CSR-like form for speed.
    # conflict_adj[node] = list of conflicting node ids (excluding self)
    # We'll build a numpy boolean matrix if memory allows, else use sets.
    # total_nodes = 190*20 = 3800, matrix is 3800x3800 ~ 14.4M bools ~ 14MB, fine.

    conflict_mat = np.zeros((total_nodes, total_nodes), dtype=bool)

    # Type (a): same edge, different colour
    for e_idx in range(num_edges):
        base = e_idx * num_colors
        for c1 in range(num_colors):
            for c2 in range(c1 + 1, num_colors):
                n1 = base + c1
                n2 = base + c2
                conflict_mat[n1, n2] = True
                conflict_mat[n2, n1] = True

    # Type (b): same colour, two edges that co-occur in some 4-cycle
    # For each colour c, for each 4-cycle, all pairs of edges in the cycle conflict.
    # But we only need to add conflicts for pairs not already present.
    # Iterate over cycles and for each pair of edges in the cycle, mark conflict
    # for the same-colour node pair.
    for cycle in all_cycles:
        # cycle is a list of 4 edges
        for i in range(4):
            for j in range(i + 1, 4):
                e1 = cycle[i]
                e2 = cycle[j]
                if e1 == e2:
                    continue
                e1_idx = edge_index[e1]
                e2_idx = edge_index[e2]
                for c in range(num_colors):
                    n1 = e1_idx * num_colors + c
                    n2 = e2_idx * num_colors + c
                    if not conflict_mat[n1, n2]:
                        conflict_mat[n1, n2] = True
                        conflict_mat[n2, n1] = True

    # Degree for each node (for branching heuristic)
    conflict_deg = conflict_mat.sum(axis=1).astype(np.int32)

    # Row sums of conflict_mat for LP bound
    # We'll use a greedy + LP relaxation bound in the B&B.

    def lp_bound(subset_nodes):
        """Compute LP relaxation bound for the MWIS on the induced subgraph of
        subset_nodes.  Uses a simple LP: max sum x_i s.t. x_i + x_j <= 1 for
        each conflict edge, 0 <= x_i <= 1.  Returns the optimal LP value.
        For speed, we solve a small LP only when the subset is small enough;
        otherwise use a greedy bound.
        """
        m = len(subset_nodes)
        if m <= 1:
            return float(m)
        if m > 300:
            # Use a fast greedy bound: sum of weights of nodes / (max degree + 1)
            # plus a simple packing bound
            # Actually, use a simple upper bound: m (trivial)
            # Better: use the fact that in a conflict graph, independence number
            # <= m / (avg_degree + 1) * ...  Let's just use m as a loose bound
            # and rely on the heuristic for pruning.
            return float(m)
        # Build small LP
        idx_list = list(subset_nodes)
        # Conflict edges within this subset
        sub_conf = conflict_mat[np.ix_(idx_list, idx_list)]
        # Upper triangle indices
        rows = []
        cols = []
        for i in range(m):
            for j in range(i + 1, m):
                if sub_conf[i, j]:
                    rows.append(i)
                    rows.append(j)
                    cols.append(len(rows) - 2)
                    cols.append(len(rows) - 1)
        # If no conflicts, bound = m
        if not rows:
            return float(m)
        # LP: max sum x_i s.t. x_i + x_j <= 1, 0 <= x_i <= 1
        c_obj = -np.ones(m)  # minimize -sum x_i
        A_ub = np.zeros((len(rows) // 2, m))
        for k in range(len(rows) // 2):
            A_ub[k, rows[2 * k]] = 1.0
            A_ub[k, rows[2 * k + 1]] = 1.0
        b_ub = np.ones(len(rows) // 2)
        bounds = [(0, 1)] * m
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                return float(-res.fun)
            else:
                return float(m)
        except Exception:
            return float(m)

    def branch_and_bound(nodes_list, best_known, time_limit):
        """Recursive B&B for MWIS on the conflict graph restricted to nodes_list.
        Returns the best solution found (list of node ids) and its size.
        Uses LP relaxation bound for pruning.
        """
        best_solution = list(best_known)
        best_size = len(best_known)
        nodes_arr = np.array(nodes_list, dtype=np.int32)
        stack = [(nodes_arr, list(best_known))]
        nodes_visited = 0

        while stack:
            if time.time() > time_limit:
                break
            current_nodes, current_sol = stack.pop()
            nodes_visited += 1
            m = len(current_nodes)
            if m == 0:
                continue
            # Bound
            ub = lp_bound(current_nodes)
            if ub + len(current_sol) <= best_size:
                continue
            # Pick a node to branch on: highest degree in the induced subgraph
            sub_conf = conflict_mat[np.ix_(current_nodes, current_nodes)]
            deg = sub_conf.sum(axis=1)
            # Pick node with max degree; tie-break by smallest index for determinism
            pivot_idx = int(np.argmax(deg))
            pivot_node = current_nodes[pivot_idx]

            # Branch 1: include pivot_node
            new_sol = current_sol + [pivot_node]
            # Remove pivot and all its neighbours from consideration
            neighbours_mask = sub_conf[pivot_idx]
            remaining = current_nodes[~neighbours_mask]
            remaining = remaining[remaining != pivot_node]
            if len(remaining) > 0:
                ub2 = lp_bound(remaining)
                if ub2 + len(new_sol) > best_size:
                    stack.append((remaining, new_sol))
            else:
                if len(new_sol) > best_size:
                    best_size = len(new_sol)
                    best_solution = new_sol

            # Branch 2: exclude pivot_node
            remaining2 = current_nodes[current_nodes != pivot_node]
            if len(remaining2) > 0:
                ub3 = lp_bound(remaining2)
                if ub3 + len(current_sol) > best_size:
                    stack.append((remaining2, current_sol))

        return best_solution, best_size, nodes_visited

    # Track forbidden colors per edge for fast lookups
    # edge_forbidden[e] = set of colors that would violate a 4-cycle constraint
    def compute_forbidden(e, edge_colors):
        """Compute the set of colors that are forbidden for edge e given current coloring."""
        forbidden = set()
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            other_edges = [oe for oe in cycle if oe != e]
            if not all(oe in edge_colors for oe in other_edges):
                continue
            other_colors = [edge_colors[oe] for oe in other_edges]
            if len(set(other_colors)) < 3:
                forbidden.update(range(num_colors))
            else:
                forbidden.update(other_colors)
        return forbidden

    def can_add_edge(e, c, edge_colors):
        """Check if adding edge e with color c keeps all 4-cycles containing e rainbow."""
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            other_edges = [oe for oe in cycle if oe != e]
            if not all(oe in edge_colors for oe in other_edges):
                continue
            other_colors = [edge_colors[oe] for oe in other_edges]
            if c in other_colors:
                return False
            if len(set(other_colors)) < 3:
                return False
        return True

    def find_valid_color(e, edge_colors):
        """Find a valid color for edge e, or None. Uses forbidden-set for speed."""
        forbidden = compute_forbidden(e, edge_colors)
        if len(forbidden) >= num_colors:
            return None
        available = [c for c in range(num_colors) if c not in forbidden]
        random.shuffle(available)
        return available[0] if available else None

    def get_conflicts(edge_colors):
        """Return dict edge -> number of conflicting cycles it participates in."""
        conflict_count = defaultdict(int)
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    for e in cycle:
                        conflict_count[e] += 1
        return conflict_count

    def repair(edge_colors, max_steps=500):
        """Remove edges to eliminate all conflicts, prioritizing high-conflict edges."""
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            conflict_count = get_conflicts(edge_colors)
            if not conflict_count:
                break
            max_count = max(conflict_count.values())
            candidates = [e for e, c in conflict_count.items() if c == max_count]
            e_to_remove = random.choice(candidates)
            del edge_colors[e_to_remove]

    def polish(edge_colors):
        """High-precision polish pass: greedily re-add every missing edge, then a
        targeted recolor-then-add sweep."""
        for _ in range(5):
            if time.time() > deadline:
                break
            progress = False
            for e in all_edges:
                if e in edge_colors:
                    continue
                c = find_valid_color(e, edge_colors)
                if c is not None:
                    edge_colors[e] = c
                    progress = True
            if not progress:
                break

        for e_rec in list(edge_colors.keys()):
            if time.time() > deadline:
                break
            old_c = edge_colors[e_rec]
            for new_c in random.sample(range(num_colors), min(8, num_colors - 1)):
                if new_c == old_c:
                    continue
                del edge_colors[e_rec]
                if can_add_edge(e_rec, new_c, edge_colors):
                    edge_colors[e_rec] = new_c
                    added = False
                    for e in all_edges:
                        if e not in edge_colors:
                            c = find_valid_color(e, edge_colors)
                            if c is not None:
                                edge_colors[e] = c
                                added = True
                    if added:
                        break
                else:
                    edge_colors[e_rec] = old_c
                    break
        return edge_colors

    def local_search(edge_colors, max_no_improve=800):
        """Local search with add / recolor / swap / double-swap moves and
        incremental missing-edge tracking."""
        best_count = len(edge_colors)
        best_state = dict(edge_colors)
        no_improve = 0
        temp = 1.0
        cooling = 0.999

        missing_set = set(all_edges) - set(edge_colors.keys())

        def save_if_better():
            nonlocal best_count, best_state
            if len(edge_colors) > best_count:
                best_count = len(edge_colors)
                best_state = dict(edge_colors)

        while time.time() < deadline:
            improved = False

            if missing_set:
                e_add = random.choice(tuple(missing_set))
                c = find_valid_color(e_add, edge_colors)
                if c is not None:
                    edge_colors[e_add] = c
                    missing_set.discard(e_add)
                    save_if_better()
                    no_improve = 0
                    continue

            if edge_colors:
                e_rec = random.choice(tuple(edge_colors.keys()))
                old_c = edge_colors[e_rec]
                new_c = random.randint(0, num_colors - 1)
                if new_c != old_c:
                    del edge_colors[e_rec]
                    if can_add_edge(e_rec, new_c, edge_colors):
                        edge_colors[e_rec] = new_c
                    else:
                        edge_colors[e_rec] = old_c

            if missing_set and edge_colors and random.random() < 0.5:
                e_remove = random.choice(tuple(edge_colors.keys()))
                e_add = random.choice(tuple(missing_set))
                del edge_colors[e_remove]
                missing_set.add(e_remove)
                c_add = find_valid_color(e_add, edge_colors)
                if c_add is not None:
                    edge_colors[e_add] = c_add
                    missing_set.discard(e_add)
                    save_if_better()
                    c_rem = find_valid_color(e_remove, edge_colors)
                    if c_rem is not None:
                        edge_colors[e_remove] = c_rem
                        missing_set.discard(e_remove)
                        save_if_better()
                        no_improve = 0
                        improved = True
                else:
                    c_rem = find_valid_color(e_remove, edge_colors)
                    if c_rem is not None:
                        edge_colors[e_remove] = c_rem
                        missing_set.discard(e_remove)

            if len(missing_set) >= 2 and len(edge_colors) >= 2 and random.random() < 0.2:
                existing_list = list(edge_colors.keys())
                missing_list = list(missing_set)
                e1, e2 = random.sample(existing_list, 2)
                e3, e4 = random.sample(missing_list, 2)
                del edge_colors[e1]
                del edge_colors[e2]
                missing_set.add(e1)
                missing_set.add(e2)
                c3 = find_valid_color(e3, edge_colors)
                c4 = find_valid_color(e4, edge_colors)
                if c3 is not None and c4 is not None:
                    edge_colors[e3] = c3
                    edge_colors[e4] = c4
                    missing_set.discard(e3)
                    missing_set.discard(e4)
                    save_if_better()
                    c1 = find_valid_color(e1, edge_colors)
                    c2 = find_valid_color(e2, edge_colors)
                    if c1 is not None:
                        edge_colors[e1] = c1
                        missing_set.discard(e1)
                    if c2 is not None:
                        edge_colors[e2] = c2
                        missing_set.discard(e2)
                    save_if_better()
                    no_improve = 0
                    improved = True
                else:
                    c1 = find_valid_color(e1, edge_colors)
                    c2 = find_valid_color(e2, edge_colors)
                    if c1 is not None:
                        edge_colors[e1] = c1
                        missing_set.discard(e1)
                    if c2 is not None:
                        edge_colors[e2] = c2
                        missing_set.discard(e2)

            if not improved:
                no_improve += 1
                temp *= cooling
            if no_improve > max_no_improve:
                break

        return best_state

    def validate_solution(edge_colors):
        """Verify every 4-cycle is rainbow. Returns True if valid."""
        for cycle in all_cycles:
            if all(e in edge_colors for e in cycle):
                colors = [edge_colors[e] for e in cycle]
                if len(set(colors)) < 4:
                    return False
        return True

    # Phase 1: Heuristic search to get a good initial solution
    best_result = None
    best_count = 0

    elapsed = time.time() - start_time
    remaining = deadline - start_time
    num_restarts = max(2, int(remaining / 18))
    num_restarts = min(num_restarts, 8)

    for restart in range(num_restarts):
        if time.time() > deadline - 40:  # reserve 40s for B&B
            break

        edge_colors = {}
        p_init = 0.7 + 0.05 * random.random()
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < p_init:
                    edge_colors[(u, v)] = random.randint(0, num_colors - 1)

        repair(edge_colors, max_steps=800)
        edge_colors = local_search(edge_colors, max_no_improve=600)

        if time.time() < deadline - 30:
            edge_colors = polish(edge_colors)
            repair(edge_colors, max_steps=200)

        if len(edge_colors) > best_count:
            best_count = len(edge_colors)
            best_result = dict(edge_colors)

    # Phase 2: Branch-and-bound on the conflict graph to improve
    if best_result is not None and time.time() < deadline - 10:
        # Convert best_result to node ids for B&B
        initial_nodes = []
        for (u, v), c in best_result.items():
            e_idx = edge_index[(u, v)]
            node_id = e_idx * num_colors + c
            initial_nodes.append(node_id)

        # Run B&B with time limit
        bb_time_limit = deadline - 5
        bb_solution, bb_size, nodes_visited = branch_and_bound(
            list(range(total_nodes)),
            initial_nodes,
            bb_time_limit
        )

        # Convert B&B solution back to edge_colors
        if bb_size > best_count:
            bb_edge_colors = {}
            for node_id in bb_solution:
                e_idx = node_id // num_colors
                c = node_id % num_colors
                e = all_edges[e_idx]
                bb_edge_colors[e] = c

            # Validate
            if validate_solution(bb_edge_colors):
                best_count = bb_size
                best_result = bb_edge_colors
            else:
                # If invalid (shouldn't happen if conflict graph is correct),
                # fall back to heuristic result
                pass

    # Final polish on the best state
    if best_result is not None and time.time() < deadline - 3:
        best_result = polish(best_result)
        repair(best_result, max_steps=200)
        if time.time() < deadline - 1:
            best_result = polish(best_result)

    if best_result is None:
        return []

    return [[u, v, c] for (u, v), c in best_result.items()]
