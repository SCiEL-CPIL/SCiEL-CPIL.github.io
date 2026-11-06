"""Few edges in 5-uniform hypergraphs without Property B (m(5))."""
import time
from itertools import combinations

_DEADLINE = 120.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START) - 10.0  # 10s safety margin


def _sat_count(edges_bitmasks, n):
    """Count 2-colorings of n vertices where no edge is monochromatic.
    Returns 0 if UNSAT. Uses backtracking with unit propagation.
    edges_bitmasks: list of int (bitmask of 5 vertices per edge).
    """
    if not edges_bitmasks:
        return 1 << n

    # Precompute for each edge: the 5 vertex indices
    edge_vertices = []
    for em in edges_bitmasks:
        vs = []
        for i in range(n):
            if em & (1 << i):
                vs.append(i)
        edge_vertices.append(vs)

    # Unit propagation + backtracking
    # State: assignment array (-1=unassigned, 0/1=assigned), edge status
    # For each edge, track count of 0s and 1s among assigned vertices.
    # If count0 == 5 -> edge all-0 -> invalid. If count1 == 5 -> edge all-1 -> invalid.
    # If count0 >= 4 and count1 >= 1 -> edge already mixed -> satisfied, ignore.
    # Similarly count1 >= 4 and count0 >= 1.
    # Unit propagation: if an edge has 4 vertices assigned to 0 and 1 unassigned,
    #   the unassigned must be 1 (otherwise all-0). Similarly for 1s.

    assignment = [-1] * n
    # edge_state[i] = (count0, count1, unassigned_list)
    edge_state = []
    for vs in edge_vertices:
        edge_state.append([0, 0, list(vs)])

    # Active edges: those not yet satisfied (mixed)
    # We'll use a simple recursive backtracking with propagation.

    def propagate(assignment, edge_state):
        """Apply unit propagation. Returns (ok, assignment, edge_state, changed)."""
        changed = True
        while changed:
            changed = False
            for ei, (c0, c1, unassigned) in enumerate(edge_state):
                if len(unassigned) == 0:
                    # Edge is fully assigned. Check if monochromatic.
                    if c0 == 5 or c1 == 5:
                        return False, assignment, edge_state, True
                    # Otherwise mixed, satisfied
                    continue
                # Check unit propagation
                if c0 == 4 and c1 == 0:
                    # Must assign remaining to 1
                    for v in unassigned:
                        if assignment[v] == -1:
                            assignment[v] = 1
                            edge_state[ei][1] += 1
                            edge_state[ei][2].remove(v)
                            changed = True
                    continue
                if c1 == 4 and c0 == 0:
                    for v in unassigned:
                        if assignment[v] == -1:
                            assignment[v] = 0
                            edge_state[ei][0] += 1
                            edge_state[ei][2].remove(v)
                            changed = True
                    continue
                if c0 == 5 or c1 == 5:
                    return False, assignment, edge_state, True
        return True, assignment, edge_state, changed

    def backtrack(assignment, edge_state):
        # Find unassigned vertex with smallest domain (all are {0,1})
        # Just pick the first unassigned
        unassigned = [i for i in range(n) if assignment[i] == -1]
        if not unassigned:
            # All assigned, check all edges
            for c0, c1, un in edge_state:
                if un == []:
                    if c0 == 5 or c1 == 5:
                        return 0
            return 1

        v = unassigned[0]
        count = 0
        for val in (0, 1):
            # Try assigning v = val
            new_assign = assignment[:]
            new_assign[v] = val
            new_es = []
            ok = True
            for ei, (c0, c1, un) in enumerate(edge_state):
                new_c0, new_c1, new_un = c0, c1, un[:]
                if v in new_un:
                    new_un.remove(v)
                    if val == 0:
                        new_c0 += 1
                    else:
                        new_c1 += 1
                if new_c0 == 5 or new_c1 == 5:
                    ok = False
                    break
                new_es.append([new_c0, new_c1, new_un])
            if not ok:
                continue
            # Propagate
            ok, new_assign, new_es, _ = propagate(new_assign, new_es)
            if not ok:
                continue
            count += backtrack(new_assign, new_es)
        return count

    ok, assignment, edge_state, _ = propagate(assignment, edge_state)
    if not ok:
        return 0
    return backtrack(assignment, edge_state)


def _is_unsat(edges_bitmasks, n):
    """Check if the hypergraph has no proper 2-coloring (UNSAT)."""
    return _sat_count(edges_bitmasks, n) == 0


def _greedy_build(n, max_edges=200):
    """Greedily add edges to make the hypergraph non-2-colorable.
    Returns list of edge bitmasks or None if not found within max_edges."""
    all_edges = list(combinations(range(n), 5))
    edge_bitmasks = []
    used = set()

    # Initial: check if already unsat (unlikely for small n)
    if _is_unsat(edge_bitmasks, n):
        return edge_bitmasks

    for _ in range(max_edges):
        if _remaining() < 5:
            return None
        best_edge = None
        best_score = -1
        # For each candidate edge, compute how many colorings it eliminates
        # Approximation: count colorings before and after
        current_count = _sat_count(edge_bitmasks, n)
        if current_count == 0:
            return edge_bitmasks

        for e in all_edges:
            if e in used:
                continue
            em = 0
            for v in e:
                em |= (1 << v)
            new_count = _sat_count(edge_bitmasks + [em], n)
            reduction = current_count - new_count
            if reduction > best_score:
                best_score = reduction
                best_edge = em
                best_tuple = e

        if best_edge is None or best_score <= 0:
            return None

        edge_bitmasks.append(best_edge)
        used.add(best_tuple)

        if _is_unsat(edge_bitmasks, n):
            return edge_bitmasks

    return None


def _local_search(edges_bitmasks, n, max_iters=50):
    """Try to reduce edges by swapping or removing."""
    current = list(edges_bitmasks)
    for _ in range(max_iters):
        if _remaining() < 5:
            break
        improved = False
        # Try removing each edge
        for i in range(len(current)):
            candidate = current[:i] + current[i+1:]
            if _is_unsat(candidate, n):
                current = candidate
                improved = True
                break
        if improved:
            continue
        # Try swapping one edge for another
        all_edges = list(combinations(range(n), 5))
        for i in range(len(current)):
            old_em = current[i]
            old_e = tuple(v for v in range(n) if old_em & (1 << v))
            for e in all_edges:
                if e == old_e:
                    continue
                if any(tuple(v for v in range(n) if em & (1 << v)) == e for em in current):
                    continue
                em = 0
                for v in e:
                    em |= (1 << v)
                candidate = current[:i] + [em] + current[i+1:]
                if _is_unsat(candidate, n):
                    current = candidate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return current


def construct():
    start = time.time()
    deadline = start + 110.0  # 10s margin

    best_edges = None
    best_n = None

    for n in range(5, 27):
        if time.time() > deadline:
            break
        if n < 5 or n > 26:
            continue

        result = _greedy_build(n)
        if result is not None:
            # Try local search to reduce
            result = _local_search(result, n)
            if best_edges is None or len(result) < len(best_edges):
                best_edges = result
                best_n = n
                print(f"n={n}, edges={len(result)}")

        # If we found something with <= 51 edges, we can stop
        if best_edges is not None and len(best_edges) <= 51:
            break

    if best_edges is None:
        # Fallback: complete hypergraph on 9 vertices
        from itertools import combinations as comb
        edges = [list(e) for e in comb(range(9), 5)]
        return {'n': 9, 'edges': edges}

    # Convert bitmasks to vertex lists
    edges = []
    for em in best_edges:
        vs = [v for v in range(best_n) if em & (1 << v)]
        edges.append(vs)

    return {'n': best_n, 'edges': edges}


if __name__ == '__main__':
    result = construct()
    print(f"n={result['n']}, edges={len(result['edges'])}")
