"""Program to find a small 5-uniform hypergraph without Property B."""
import random
import time
from itertools import combinations


def construct():
    """Return a 5-uniform hypergraph with no proper 2-coloring, minimizing edges."""
    start_time = time.time()
    deadline = start_time + 110.0

    def time_left():
        return deadline - time.time()

    # --- Verification: backtracking with unit propagation ---
    def is_non_2_colorable(n, edges, timeout=0.5):
        """Check if hypergraph has NO proper 2-coloring via backtracking + unit prop."""
        m = len(edges)
        if m == 0:
            return False

        # Build incidence: for each edge, store set of vertex indices
        edge_vertices = [list(e) for e in edges]

        # For each vertex, store list of edge indices containing it
        vert_edges = [[] for _ in range(n)]
        for i, e in enumerate(edge_vertices):
            for v in e:
                vert_edges[v].append(i)

        # State: color[v] = -1 (uncolored), 0 or 1 (colored)
        # For each edge: count_0[e], count_1[e], uncolored_count[e]
        color = [-1] * n
        count0 = [0] * m
        count1 = [0] * m
        uncol = [5] * m  # 5 vertices per edge

        # Stack of (vertex, edge) assignments for undo
        undo_stack = []

        def propagate():
            """Unit propagation: if an edge has 4 of one color, force the 5th."""
            changed = True
            while changed:
                changed = False
                for e_idx in range(m):
                    c0 = count0[e_idx]
                    c1 = count1[e_idx]
                    unc = uncol[e_idx]
                    if unc == 0:
                        # Edge is complete: check if monochromatic
                        if c0 == 5 or c1 == 5:
                            return False  # Conflict
                        continue
                    if c0 == 4:
                        # Force all uncolored in this edge to color 1
                        for v in edge_vertices[e_idx]:
                            if color[v] == -1:
                                color[v] = 1
                                count1[e_idx] += 1
                                uncol[e_idx] -= 1
                                undo_stack.append((v, e_idx, 1))
                                changed = True
                    elif c1 == 4:
                        # Force all uncolored in this edge to color 0
                        for v in edge_vertices[e_idx]:
                            if color[v] == -1:
                                color[v] = 0
                                count0[e_idx] += 1
                                uncol[e_idx] -= 1
                                undo_stack.append((v, e_idx, 0))
                                changed = True
            return True

        # Try both colorings of vertex 0
        for first_color in [0, 1]:
            # Reset state
            color = [-1] * n
            count0 = [0] * m
            count1 = [0] * m
            uncol = [5] * m
            undo_stack = []

            # Assign vertex 0
            color[0] = first_color
            if first_color == 0:
                for e_idx in vert_edges[0]:
                    count0[e_idx] += 1
                    uncol[e_idx] -= 1
            else:
                for e_idx in vert_edges[0]:
                    count1[e_idx] += 1
                    uncol[e_idx] -= 1

            if not propagate():
                continue  # Conflict, try other assignment

            # Backtracking search on remaining vertices
            def backtrack():
                # Find uncolored vertex
                v = -1
                for i in range(n):
                    if color[i] == -1:
                        v = i
                        break
                if v == -1:
                    return True  # All colored, check for monochromatic edges

                # Check if any edge is monochromatic (conflict)
                for e_idx in range(m):
                    if uncol[e_idx] == 0:
                        if count0[e_idx] == 5 or count1[e_idx] == 5:
                            return False

                # Try color 0
                for c in [0, 1]:
                    # Assign color c to v
                    color[v] = c
                    local_undo = []
                    for e_idx in vert_edges[v]:
                        if c == 0:
                            count0[e_idx] += 1
                            uncol[e_idx] -= 1
                        else:
                            count1[e_idx] += 1
                            uncol[e_idx] -= 1
                        local_undo.append(e_idx)

                    # Save state for undo
                    state_save = []
                    for e_idx in local_undo:
                        state_save.append((count0[e_idx], count1[e_idx], uncol[e_idx]))

                    if propagate():
                        if backtrack():
                            return True

                    # Undo
                    for i, e_idx in enumerate(local_undo):
                        count0[e_idx] = state_save[i][0]
                        count1[e_idx] = state_save[i][1]
                        uncol[e_idx] = state_save[i][2]
                    color[v] = -1

                    # Undo propagation effects
                    while undo_stack:
                        pv, pe, pc = undo_stack.pop()
                        if pc == 1:
                            count1[pe] -= 1
                        else:
                            count0[pe] -= 1
                        uncol[pe] += 1
                        color[pv] = -1

                return False

            if backtrack():
                return False  # Found a valid 2-coloring

        return True  # No 2-coloring found

    # --- Known small construction: 51-edge hypergraph ---
    # Based on the best known m(5) construction
    # We use a randomized search approach

    def random_hypergraph(n, m, seed=None):
        rng = random.Random(seed)
        edges = set()
        attempts = 0
        while len(edges) < m and attempts < m * 50:
            e = frozenset(rng.sample(range(n), 5))
            edges.add(e)
            attempts += 1
        return [list(e) for e in edges]

    def remove_redundant_edges(n, edges, deadline):
        """Try to remove edges while maintaining non-2-colorability."""
        edges = [list(e) for e in edges]
        m = len(edges)
        if m <= 5:
            return edges

        # Shuffle edges for randomized removal order
        indices = list(range(m))
        random.shuffle(indices)

        for idx in indices:
            if time.time() > deadline:
                break
            # Try removing edge idx
            test_edges = [e for i, e in enumerate(edges) if i != idx]
            if is_non_2_colorable(n, test_edges, timeout=0.3):
                edges = test_edges
        return edges

    # Start with various constructions
    best_result = None
    best_edge_count = 200

    # Try the complete 5-uniform hypergraph on 9 vertices (126 edges) as baseline
    n9 = 9
    edges9 = [list(e) for e in combinations(range(n9), 5)]
    # Check if it's non-2-colorable
    if is_non_2_colorable(n9, edges9, timeout=1.0):
        edges9 = remove_redundant_edges(n9, edges9, deadline)
        if len(edges9) < best_edge_count:
            best_edge_count = len(edges9)
            best_result = {'n': n9, 'edges': edges9}

    # Try random constructions on various vertex counts
    for n in range(10, 17):
        if time.time() > deadline:
            break
        for m in [30, 40, 50, 60]:
            if time.time() > deadline:
                break
            # Multiple random attempts
            for attempt in range(5):
                if time.time() > deadline:
                    break
                edges = random_hypergraph(n, m, seed=hash((n, m, attempt)) % (2**31))
                if is_non_2_colorable(n, edges, timeout=0.4):
                    edges = remove_redundant_edges(n, edges, deadline)
                    if len(edges) < best_edge_count:
                        best_edge_count = len(edges)
                        best_result = {'n': n, 'edges': edges}

    # Try structured constructions: take all 5-subsets from a smaller core
    # e.g., complete hypergraph on 8 vertices + some extra edges
    for n_core in [7, 8]:
        if time.time() > deadline:
            break
        core_edges = [list(e) for e in combinations(range(n_core), 5)]
        # Add random extra vertices and edges
        for extra in range(0, 4):
            if time.time() > deadline:
                break
            n = n_core + extra
            edges = list(core_edges)
            # Add random edges involving extra vertices
            for _ in range(20):
                if time.time() > deadline:
                    break
                v_extra = random.choice(range(n_core, n))
                v_rest = random.sample(range(n_core), 4)
                e = [v_extra] + v_rest
                if e not in edges:
                    edges.append(e)
            if is_non_2_colorable(n, edges, timeout=0.4):
                edges = remove_redundant_edges(n, edges, deadline)
                if len(edges) < best_edge_count:
                    best_edge_count = len(edges)
                    best_result = {'n': n, 'edges': edges}

    # Fallback: complete 5-uniform on 9 vertices
    if best_result is None:
        best_result = {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

    return best_result
