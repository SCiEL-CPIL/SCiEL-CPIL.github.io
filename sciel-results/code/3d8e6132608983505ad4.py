import time
import random
from itertools import combinations


def _is_non_2colorable(n, edges, deadline=None):
    """
    Check if a 5-uniform hypergraph is NOT 2-colorable using backtracking
    with unit propagation.
    Returns True if the hypergraph is NOT 2-colorable.
    """
    if deadline is None:
        deadline = time.time() + 10

    if not edges:
        return False  # Empty hypergraph is trivially 2-colorable

    m = len(edges)
    edge_verts = [tuple(e) for e in edges]

    # For each vertex, which edges contain it
    vert_in_edges = [[] for _ in range(n)]
    for i, ev in enumerate(edge_verts):
        for v in ev:
            vert_in_edges[v].append(i)

    # State: assignment[v] in {-1, 0, 1}
    # For each edge, track counts of color 0 and color 1 among assigned vertices
    # and list of unassigned vertices

    def _check_and_propagate(assign, e_c0, e_c1, e_unas):
        """
        Check for contradictions and do unit propagation.
        Returns False on contradiction.
        Mutates assign, e_c0, e_c1, e_unas in place.
        """
        # First pass: detect contradictions
        for i in range(m):
            if e_c0[i] == 5 or e_c1[i] == 5:
                return False

        # Unit propagation loop
        changed = True
        while changed:
            changed = False
            for i in range(m):
                if e_c0[i] == 5 or e_c1[i] == 5:
                    return False
                if len(e_unas[i]) == 1:
                    v = e_unas[i][0]
                    if e_c0[i] == 4:
                        # v must be color 1
                        if assign[v] == 0:
                            return False
                        if assign[v] == -1:
                            assign[v] = 1
                            e_c1[i] += 1
                            e_unas[i] = []
                            changed = True
                        # Update other edges containing v
                        for j in vert_in_edges[v]:
                            if j != i and assign[j] != -1:  # This check is wrong
                                pass
                            if j != i:
                                if e_unas[j] and v in e_unas[j]:
                                    e_unas[j].remove(v)
                                    e_c1[j] += 1
                    elif e_c1[i] == 4:
                        # v must be color 0
                        if assign[v] == 1:
                            return False
                        if assign[v] == -1:
                            assign[v] = 0
                            e_c0[i] += 1
                            e_unas[i] = []
                            changed = True
                        for j in vert_in_edges[v]:
                            if j != i:
                                if e_unas[j] and v in e_unas[j]:
                                    e_unas[j].remove(v)
                                    e_c0[j] += 1
        return True

    def _init_state():
        """Initialize state from current assignment."""
        assign = [-1] * n
        e_c0 = [0] * m
        e_c1 = [0] * m
        e_unas = [[] for _ in range(m)]
        for i, ev in enumerate(edge_verts):
            for v in ev:
                e_unas[i].append(v)
        return assign, e_c0, e_c1, e_unas

    def _backtrack(assign, e_c0, e_c1, e_unas, depth):
        """
        Returns True if a valid 2-coloring exists (IS 2-colorable).
        """
        if time.time() > deadline:
            return True  # Conservative: assume colorable

        # Find unassigned vertex with most constraints
        unassigned = [v for v in range(n) if assign[v] == -1]
        if not unassigned:
            # Check all edges
            for i in range(m):
                if e_c0[i] == 5 or e_c1[i] == 5:
                    return False
            return True

        # MRV + degree heuristic
        v = max(unassigned, key=lambda x: len(vert_in_edges[x]))

        for color in (0, 1):
            # Save state
            old_assign = assign[:]
            old_e_c0 = e_c0[:]
            old_e_c1 = e_c1[:]
            old_e_unas = [lst[:] for lst in e_unas]

            assign[v] = color
            # Update state for all edges containing v
            for j in vert_in_edges[v]:
                if color == 0:
                    e_c0[j] += 1
                else:
                    e_c1[j] += 1
                if v in e_unas[j]:
                    e_unas[j].remove(v)

            if _check_and_propagate(assign, e_c0, e_c1, e_unas):
                if _backtrack(assign, e_c0, e_c1, e_unas, depth + 1):
                    return True

            # Restore
            assign[:] = old_assign
            e_c0[:] = old_e_c0
            e_c1[:] = old_e_c1
            e_unas[:] = old_e_unas

        return False

    # Initialize
    assign, e_c0, e_c1, e_unas = _init_state()
    if not _check_and_propagate(assign, e_c0, e_c1, e_unas):
        return True  # Contradiction: NOT 2-colorable

    if _backtrack(assign, e_c0, e_c1, e_unas, 0):
        return False  # IS 2-colorable
    else:
        return True  # NOT 2-colorable


def _greedy_grow(n, deadline):
    """
    Grow a hypergraph by adding edges that "kill" the most remaining colorings.
    Uses the bitmask approach for small n.
    """
    if n > 14:
        return None  # Too large for bitmask approach

    all_edges = list(combinations(range(n), 5))
    total = 1 << n
    full_mask = (1 << total) - 1

    # Precompute kill masks
    kill_masks = []
    for e in all_edges:
        e_mask = 0
        for v in e:
            e_mask |= (1 << v)
        km = 0
        for c in range(total):
            if (e_mask & c) == e_mask or (e_mask & c) == 0:
                km |= (1 << c)
        kill_masks.append(km)

    uncovered = full_mask
    chosen = []
    used = [False] * len(all_edges)

    while uncovered:
        best_idx = -1
        best_count = 0
        for i in range(len(all_edges)):
            if used[i]:
                continue
            new_cov = (kill_masks[i] & uncovered).bit_count()
            if new_cov > best_count:
                best_count = new_cov
                best_idx = i
        if best_idx == -1:
            break
        chosen.append(best_idx)
        used[best_idx] = True
        uncovered &= ~kill_masks[best_idx]
        if time.time() > deadline - 5:
            break

    if not chosen:
        return None

    # Greedy removal
    result = chosen[:]
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for i in range(len(result) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            test = result[:i] + result[i+1:]
            union = 0
            for j in test:
                union |= kill_masks[j]
            if union == full_mask:
                result = test
                improved = True
                break

    return [list(all_edges[i]) for i in result]


def _local_search(n, start_edges, deadline):
    """
    Local search: try to remove edges, and if stuck, try swapping.
    """
    current = list(start_edges)
    best = list(current)

    def is_good(edges):
        return _is_non_2colorable(n, edges, deadline)

    # Phase 1: Remove edges
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for i in range(len(current)):
            if time.time() > deadline - 3:
                break
            test = current[:i] + current[i+1:]
            if len(test) < 15:
                continue
            if is_good(test):
                current = test
                improved = True
                if len(current) < len(best):
                    best = list(current)
                break

    # Phase 2: Try swaps (remove one, add another)
    all_edges_set = set(tuple(e) for e in current)
    all_possible = set(combinations(range(n), 5))
    candidates = [e for e in all_possible if e not in all_edges_set]
    random.shuffle(candidates)

    for _ in range(3):
        if time.time() > deadline - 2:
            break
        for i in range(len(current)):
            if time.time() > deadline - 2:
                break
            for add_edge in candidates[:20]:
                if time.time() > deadline - 2:
                    break
                test = current[:i] + current[i+1:] + [add_edge]
                if len(test) >= len(current):
                    continue
                if is_good(test):
                    current = test
                    if len(current) < len(best):
                        best = list(current)
                    break

    return best


def construct():
    start_time = time.time()
    deadline = start_time + 145

    best_edges = None
    best_n = 0

    # Strategy 1: Greedy bitmask for small n
    for n in range(9, 13):
        if time.time() > deadline - 20:
            break
        try:
            edges = _greedy_grow(n, deadline - 15)
            if edges and (best_edges is None or len(edges) < len(best_edges)):
                best_edges = edges
                best_n = n
        except Exception:
            continue

    # Strategy 2: Local search from greedy result or random start
    if best_edges is not None:
        try:
            improved = _local_search(best_n, best_edges, deadline - 5)
            if improved and len(improved) < len(best_edges):
                best_edges = improved
        except Exception:
            pass
    else:
        # Fallback: random search
        for n in range(10, 14):
            if time.time() > deadline - 10:
                break
            for attempt in range(3):
                if time.time() > deadline - 5:
                    break
                all_e = list(combinations(range(n), 5))
                random.shuffle(all_e)
                start = all_e[:40]
                if _is_non_2colorable(n, start, deadline - 5):
                    improved = _local_search(n, start, deadline - 3)
                    if improved and (best_edges is None or len(improved) < len(best_edges)):
                        best_edges = improved
                        best_n = n

    # Final fallback
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n
    else:
        best_edges = [list(e) for e in best_edges]

    return {'n': best_n, 'edges': best_edges}
