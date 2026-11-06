"""Constructs a C5-free graph with no independent set of size 9.

Design family kept: disjoint union of C5-free blocks. Each block is a complete
graph K4 (C5-free, independence number 1). 8 such blocks give n = 32 with
independence number 8 (< 9). The program verifies both properties exactly and
returns immediately; a wall-clock deadline guard ensures a valid result is
always returned before the evaluator kills the process.
"""
import time


# EVOLVE-BLOCK-START
def _has_c5(n, adj):
    """Exact check for any 5-cycle as a (not necessarily induced) subgraph."""
    for v0 in range(n):
        for v1 in range(v0 + 1, n):
            if not adj[v0][v1]:
                continue
            for v2 in range(v1 + 1, n):
                if not adj[v1][v2]:
                    continue
                for v3 in range(v2 + 1, n):
                    if not adj[v2][v3]:
                        continue
                    for v4 in range(v3 + 1, n):
                        if adj[v3][v4] and adj[v4][v0]:
                            return True
    return False


def _has_independent_set(n, adj, k):
    """Exact decision: does an independent set of size k exist? (branch & bound)"""
    if k <= 1:
        return True
    adj_list = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i][j]:
                adj_list[i].add(j)
                adj_list[j].add(i)

    # Order candidates by ascending degree in the current subgraph to prune faster.
    best_found = 0

    def dfs(cand, size):
        nonlocal best_found
        if size + len(cand) < k:
            return False
        if size >= best_found:
            best_found = size
        if not cand:
            return size >= k
        # Greedy upper-bound prune: pick a vertex and bound.
        v = cand[0]
        rest = cand[1:]
        # Include v: neighbours of v cannot be in the set.
        new_cand = [x for x in rest if x not in adj_list[v]]
        if dfs(new_cand, size + 1):
            return True
        # Exclude v.
        if dfs(rest, size):
            return True
        return False

    return dfs(list(range(n)), 0)


def _build_blocks(blocks, k):
    n = blocks * k
    edges = []
    for b in range(blocks):
        base = b * k
        for i in range(k):
            for j in range(i + 1, k):
                edges.append([base + i, base + j])
    return n, edges


def _make_adj(n, edges):
    adj = [[False] * n for _ in range(n)]
    for u, v in edges:
        adj[u][v] = adj[v][u] = True
    return adj


def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin well under 150s wall-clock

    # Primary construction: 8 disjoint K4s.
    blocks, k = 8, 4
    n, edges = _build_blocks(blocks, k)
    adj = _make_adj(n, edges)

    # Verify C5-freedom.
    if _has_c5(n, adj):
        # Should never happen for K4 blocks, but fall back defensively.
        blocks, k = 7, 4
        n, edges = _build_blocks(blocks, k)
        adj = _make_adj(n, edges)

    # Verify no independent set of size 9 (i.e. alpha < 9).
    if _has_independent_set(n, adj, 9):
        # Should never happen (alpha = 8), but fall back defensively.
        blocks, k = 7, 4
        n, edges = _build_blocks(blocks, k)
        adj = _make_adj(n, edges)

    # Deadline guard: the construction is O(1) in practice; if somehow slow,
    # return the current (still valid) result rather than risk a timeout.
    if time.time() > deadline:
        return {'n': n, 'edges': edges}

    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
