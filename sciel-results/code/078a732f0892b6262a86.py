"""Few edges in 5-uniform hypergraphs without Property B (m(5)).

Strategy: for each n, start from the complete 5-uniform hypergraph K^5_n
(which is trivially non-2-colorable when n >= 9) and greedily delete edges
while the hypergraph remains non-2-colorable. This yields an edge-deletion-
minimal non-2-colorable hypergraph. We use a fast bitmask backtracking
2-colorability checker with unit propagation, and keep the smallest result
found before the deadline.
"""

import time
from itertools import combinations

_DEADLINE = 150.0
_START = time.time()
_MARGIN = 5.0  # safety margin (seconds)


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _is_2_colorable(n, edges_mask, edge_list):
    """Return True if the hypergraph (vertices 0..n-1, edges as 5-bit-ish
    vertex masks in edges_mask) admits a 2-colouring with every edge having
    both colours. Uses backtracking with unit propagation on vertex colours.

    edge_list: list of frozenset of vertex indices (length 5) for iteration.
    edges_mask: list of integer bitmasks of the 5 vertices in each edge.
    """
    # colour: -1 unknown, 0/1 assigned. Use dict for sparse assignment.
    # We'll implement backtracking with propagation.
    n = n
    m = len(edge_list)
    if m == 0:
        # no edges -> trivially 2-colourable
        return True

    # Precompute for each vertex the list of edges it belongs to.
    v_edges = [[] for _ in range(n)]
    for i, es in enumerate(edge_list):
        for v in es:
            v_edges[v].append(i)

    # We'll do DFS. State: assignment array of length n with -1/0/1.
    # Unit propagation: if an edge has 4 vertices of one colour, the 5th is
    # forced to the other colour. If an edge has all 5 the same colour -> conflict.
    # We'll implement propagation as a worklist.

    def propagate(assign, queue):
        # assign: list of ints (-1/0/1). queue: list of vertex indices whose
        # colour was just set (to re-examine their edges).
        while queue:
            v = queue.pop()
            cv = assign[v]
            for ei in v_edges[v]:
                es = edge_list[ei]
                # count colours in this edge
                cnt = [0, 0]
                unknown = None
                for u in es:
                    cu = assign[u]
                    if cu == -1:
                        unknown = u
                    else:
                        cnt[cu] += 1
                if cnt[0] == 5 or cnt[1] == 5:
                    return False  # monochromatic edge -> conflict
                # If 4 of one colour and 1 unknown -> force the unknown.
                if unknown is not None:
                    if cnt[0] == 4:
                        # force unknown to 1
                        if assign[unknown] == 1:
                            continue
                        if assign[unknown] != -1:
                            return False
                        assign[unknown] = 1
                        queue.append(unknown)
                    elif cnt[1] == 4:
                        if assign[unknown] == 0:
                            continue
                        if assign[unknown] != -1:
                            return False
                        assign[unknown] = 0
                        queue.append(unknown)
        return True

    # Branching heuristic: pick an unassigned vertex with the most edges
    # (or a vertex in an edge with the most determined vertices).
    def branch(assign, queue):
        # Find an unassigned vertex; prefer one that appears in an edge with
        # the fewest remaining unknowns (to trigger propagation quickly).
        best_v = -1
        best_score = -1
        # Try to find an edge with exactly 1 unknown; that unknown gets forced
        # by propagation already. So look for an edge with 2 unknowns; branch
        # on one of them.
        for ei in range(m):
            es = edge_list[ei]
            unk = 0
            first_un = -1
            for u in es:
                if assign[u] == -1:
                    unk += 1
                    if first_un == -1:
                        first_un = u
            if unk == 2:
                # branching on first_un is good
                return first_un
            if unk == 1:
                # should have been resolved by propagation; force it here
                # (defensive)
                for u in es:
                    if assign[u] == -1:
                        # count colours
                        c0 = sum(1 for u2 in es if assign[u2] == 0)
                        c1 = sum(1 for u2 in es if assign[u2] == 1)
                        if c0 == 4:
                            assign[u] = 1
                            queue.append(u)
                        elif c1 == 4:
                            assign[u] = 0
                            queue.append(u)
                        else:
                            # shouldn't happen
                            assign[u] = 0
                            queue.append(u)
                        break
                return -2  # signal: propagation happened, re-check
        # No edge with 2 unknowns; pick any unassigned vertex.
        for v in range(n):
            if assign[v] == -1:
                return v
        # All assigned; check all edges have both colours.
        for ei in range(m):
            es = edge_list[ei]
            c0 = 0
            c1 = 0
            for u in es:
                if assign[u] == 0:
                    c0 += 1
                else:
                    c1 += 1
            if c0 == 0 or c1 == 0:
                return -3  # conflict
        return -4  # success

    def dfs(assign):
        queue = []
        # initial propagation: none (nothing assigned)
        while True:
            r = branch(assign, queue)
            if r == -2:
                # propagation changed something; ensure consistency
                if not propagate(assign, queue):
                    return False
                continue
            if r == -3:
                return False
            if r == -4:
                return True
            if r == -1:
                # all assigned but not verified? treat as success path check
                for ei in range(m):
                    es = edge_list[ei]
                    c0 = sum(1 for u in es if assign[u] == 0)
                    c1 = sum(1 for u in es if assign[u] == 1)
                    if c0 == 0 or c1 == 0:
                        return False
                return True
            # branch on vertex r
            # Try colour 0 first
            assign[r] = 0
            q2 = [r]
            if propagate(assign, q2):
                if dfs(assign):
                    return True
            assign[r] = 1
            q3 = [r]
            if propagate(assign, q3):
                if dfs(assign):
                    return True
            assign[r] = -1
            return False

    assign = [-1] * n
    return dfs(assign)


def construct():
    best = None  # (edge_count, n, edges)
    # n must be at least 9 for K^5_n to be non-2-colourable (pigeonhole).
    # For n in 9..26, build K^5_n and minimize.
    # Start with larger n first? Larger n has more edges initially but may
    # minimize to fewer. We'll try n from 9 to 26, but order by a heuristic.
    # Actually K^5_n has C(n,5) edges; minimizing may leave a small core.
    # We'll try n in increasing order of C(n,5) but cap by deadline.
    ns = list(range(9, 27))
    # Try smaller n first (fewer initial edges, faster to minimize).
    for n in ns:
        if _time_left() < 0.5:
            break
        all_edges = list(combinations(range(n), 5))
        # Convert to frozensets and masks.
        edge_list = [frozenset(e) for e in all_edges]
        # Start with all edges present; check non-2-colourable (should be true for n>=9).
        # Greedy deletion: try to remove each edge; if still non-2-colourable, remove it.
        present = [True] * len(all_edges)
        # First verify the full hypergraph is non-2-colourable.
        if _is_2_colorable(n, None, edge_list):
            # Should not happen for n>=9, but skip.
            continue
        # Greedy deletion. To speed up, iterate edges in a fixed order and
        # attempt removal; re-check 2-colourability of the remaining set.
        # Re-checking the whole hypergraph each time is expensive; but n is
        # small and edge counts moderate. We'll do it.
        # To reduce checks, we can try removals in random-ish order but keep
        # deterministic for reproducibility. Use a simple order.
        idx = list(range(len(all_edges)))
        # We'll attempt to remove each edge once in this pass.
        for i in idx:
            if _time_left() < 0.2:
                break
            if not present[i]:
                continue
            # Try removing edge i
            present[i] = False
            # Build the current edge list (only present edges)
            cur_edges = [edge_list[j] for j in range(len(all_edges)) if present[j]]
            if _is_2_colorable(n, None, cur_edges):
                # 2-colourable now, so we must keep this edge
                present[i] = True
            else:
                # Still non-2-colourable; keep it removed
                pass
        # After one pass, do additional passes to catch more removals
        # (removing one edge can enable removing others).
        improved = True
        while improved and _time_left() > 0.2:
            improved = False
            for i in idx:
                if _time_left() < 0.1:
                    break
                if not present[i]:
                    continue
                present[i] = False
                cur_edges = [edge_list[j] for j in range(len(all_edges)) if present[j]]
                if _is_2_colorable(n, None, cur_edges):
                    present[i] = True
                else:
                    improved = True
        # Build final edge list
        final_edges = [list(edge_list[j]) for j in range(len(all_edges)) if present[j]]
        ec = len(final_edges)
        if best is None or ec < best[0]:
            best = (ec, n, final_edges)
    if best is None:
        # Fallback: complete hypergraph on 9 vertices
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    ec, n, edges = best
    return {'n': n, 'edges': edges}


# EVOLVE-BLOCK-START
def construct():
    # Use the optimized construction defined above.
    return _construct_impl()


def _construct_impl():
    best = None
    ns = list(range(9, 27))
    for n in ns:
        if _time_left() < 0.5:
            break
        all_edges = list(combinations(range(n), 5))
        edge_list = [frozenset(e) for e in all_edges]
        if _is_2_colorable(n, None, edge_list):
            continue
        present = [True] * len(all_edges)
        idx = list(range(len(all_edges)))
        for i in idx:
            if _time_left() < 0.2:
                break
            if not present[i]:
                continue
            present[i] = False
            cur_edges = [edge_list[j] for j in range(len(all_edges)) if present[j]]
            if _is_2_colorable(n, None, cur_edges):
                present[i] = True
        improved = True
        while improved and _time_left() > 0.2:
            improved = False
            for i in idx:
                if _time_left() < 0.1:
                    break
                if not present[i]:
                    continue
                present[i] = False
                cur_edges = [edge_list[j] for j in range(len(all_edges)) if present[j]]
                if _is_2_colorable(n, None, cur_edges):
                    present[i] = True
                else:
                    improved = True
        final_edges = [list(edge_list[j]) for j in range(len(all_edges)) if present[j]]
        ec = len(final_edges)
        if best is None or ec < best[0]:
            best = (ec, n, final_edges)
    if best is None:
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    ec, n, edges = best
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
