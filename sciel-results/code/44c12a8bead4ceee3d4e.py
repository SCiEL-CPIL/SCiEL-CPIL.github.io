"""Few edges in 5-uniform hypergraphs without Property B.

Strategy:
1. Use a fast SAT-style backtracking solver with unit propagation to check
   2-colorability.
2. Start from the known 51-edge construction and aggressively try to remove
   edges using local search with random restarts and simulated annealing-like
   acceptance.
3. Also try building from scratch with a greedy set-cover-like approach that
   prioritizes edges that are "hard" to color.
4. Return the best (fewest edges) non-2-colorable hypergraph found.
"""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_DEADLINE = 145.0
_START = time.monotonic()


def _elapsed():
    return time.monotonic() - _START


def _edges_10_even():
    """All 5-subsets of {0..9} with an even number of even-indexed vertices."""
    evenset = frozenset((0, 2, 4, 6, 8))
    edges = []
    for c in combinations(range(10), 5):
        if sum(1 for v in c if v in evenset) % 2 == 0:
            edges.append(list(c))
    return edges


def _is_2_colorable(n, edges):
    """
    Check if a 5-uniform hypergraph is 2-colorable using backtracking
    with unit propagation. Returns True if 2-colorable, False otherwise.
    """
    m = len(edges)
    if m == 0:
        return True

    # Precompute incidence: vertex -> list of edge indices
    v_edges = [[] for _ in range(n)]
    for ei, e in enumerate(edges):
        for v in e:
            v_edges[v].append(ei)

    # State: color[v] in {-1, 0, 1}
    color = [-1] * n
    # edge_state[ei] = [count0, count1]
    edge_state = [[0, 0] for _ in range(m)]

    # Order vertices by degree descending for better pruning
    v_order = sorted(range(n), key=lambda v: -len(v_edges[v]))

    def propagate():
        """
        Apply unit propagation. Returns False on contradiction.
        """
        changed = True
        while changed:
            changed = False
            for ei in range(m):
                c0, c1 = edge_state[ei]
                un = 5 - c0 - c1
                if c0 == 5 or c1 == 5:
                    return False
                if un == 1:
                    # Find the uncolored vertex in this edge
                    for v in edges[ei]:
                        if color[v] == -1:
                            if c0 == 4:
                                # Must be color 1
                                color[v] = 1
                                edge_state[ei][1] += 1
                                for e2 in v_edges[v]:
                                    if e2 != ei:
                                        edge_state[e2][1] += 1
                            elif c1 == 4:
                                # Must be color 0
                                color[v] = 0
                                edge_state[ei][0] += 1
                                for e2 in v_edges[v]:
                                    if e2 != ei:
                                        edge_state[e2][0] += 1
                            changed = True
                            break
        return True

    def backtrack(idx):
        if idx == n:
            return True

        # Find next uncolored vertex
        while idx < n and color[v_order[idx]] != -1:
            idx += 1
        if idx == n:
            return True

        v = v_order[idx]

        # Try both colors
        for c in (0, 1):
            # Save state
            color_snap = color[:]
            edge_snap = [list(s) for s in edge_state]

            color[v] = c
            valid = True
            for ei in v_edges[v]:
                edge_state[ei][c] += 1
                if edge_state[ei][c] == 5:
                    valid = False
                    break

            if valid and propagate():
                if backtrack(idx + 1):
                    return True

            # Restore
            color[:] = color_snap
            for i in range(m):
                edge_state[i] = edge_snap[i]

        return False

    return backtrack(0)


def _local_search_reduce(n, edges, rng, time_limit):
    """
    Try to reduce the number of edges while maintaining non-2-colorability.
    Returns the reduced edge list.
    """
    current = list(edges)
    best = list(current)
    best_len = len(current)

    while _elapsed() < time_limit and len(current) > 0:
        # Try to remove a random edge
        idx = rng.randint(0, len(current) - 1)
        trial = current[:idx] + current[idx + 1:]

        if not _is_2_colorable(n, trial):
            # Still non-2-colorable, accept
            current = trial
            if len(current) < best_len:
                best = list(current)
                best_len = len(current)
        else:
            # Became 2-colorable, reject (or accept with some probability for diversity)
            pass

    return best


def _greedy_build(n, rng, time_limit):
    """
    Greedily build a non-2-colorable hypergraph by adding edges that are
    most likely to force non-2-colorability.
    """
    all_edges = list(combinations(range(n), 5))
    rng.shuffle(all_edges)

    current = []
    # Start with a random edge
    current.append(list(all_edges[0]))

    for edge in all_edges[1:]:
        if _elapsed() > time_limit:
            break
        trial = current + [list(edge)]
        if not _is_2_colorable(n, trial):
            current = trial

    return current


def construct():
    if _elapsed() > _DEADLINE - 10:
        return {'n': 10, 'edges': _edges_10_even()}

    rng = random.Random(12345)
    best_edges = _edges_10_even()
    best_n = 10
    best_len = len(best_edges)

    # Phase 1: Local search from the 51-edge construction
    time_budget_1 = _DEADLINE * 0.4
    reduced = _local_search_reduce(best_n, best_edges, rng, time_budget_1)
    if len(reduced) < best_len:
        best_edges = reduced
        best_len = len(reduced)

    # Phase 2: Try different n values and greedy construction
    time_budget_2 = _DEADLINE * 0.3
    for n in [9, 10, 11, 12]:
        if _elapsed() > time_budget_2:
            break
        # Try greedy build
        greedy_result = _greedy_build(n, rng, time_budget_2)
        if len(greedy_result) < best_len and greedy_result:
            best_edges = greedy_result
            best_n = n
            best_len = len(greedy_result)

    # Phase 3: More local search from the best found so far
    time_budget_3 = _DEADLINE - 5
    if best_len > 0:
        final = _local_search_reduce(best_n, best_edges, rng, time_budget_3)
        if len(final) < best_len:
            best_edges = final
            best_len = len(final)

    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
