"""Rewritten program: MWIS branch-and-bound over a compatibility graph of candidate rows."""
import numpy as np


def _rows_up_to(M, k):
    """All increasing rows of length k+1 starting at 0 with max <= M,
    sorted by scope (max entry) ascending. Returns list of tuples."""
    L = k + 1
    out = []
    for M2 in range(1, M + 1):
        for a in range(1, M2 + 1):
            for b in range(a + 1, M2 + 1):
                for c in range(b + 1, M2 + 1):
                    for d in range(c + 1, M2 + 1):
                        for e in range(d + 1, M2 + 1):
                            out.append((0, a, b, c, d, e))
    return out


def _diffs(row):
    """Set of all positive differences within a row."""
    s = set()
    n = len(row)
    for i in range(n):
        ri = row[i]
        for j in range(i + 1, n):
            s.add(row[j] - ri)
    return s


def construct(n_rows, k):
    L = k + 1
    rows = _rows_up_to(200, k)
    n = len(rows)
    if n == 0:
        return [tuple([0] * L)] * n_rows

    # Precompute difference sets
    diffsets = [_diffs(r) for r in rows]

    # Build conflict bitsets: node i conflicts with j if diffsets overlap
    # Use Python ints as bitsets
    conflicts = [0] * n
    # Symmetric; fill upper triangle
    for i in range(n):
        di = diffsets[i]
        for j in range(i + 1, n):
            if di & diffsets[j]:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)

    # Greedy coloring for upper bound: process nodes in order, assign smallest
    # color not used by already-colored neighbors. Returns list of colors (0-based).
    # We use a simple greedy: iterate nodes, color = number of colored neighbors + 1 (approx).
    # Better: standard greedy coloring on the induced subgraph in a fixed order.
    # For speed, we use a heuristic: color[i] = 1 + max(color[j] for j in colored neighbors)
    # This gives a valid upper bound on the clique number, hence on independent set size.
    color = [0] * n
    for i in range(n):
        c = 0
        # Look at neighbors that are already colored (index < i in this ordering)
        # Actually for a valid coloring upper bound we need: color[i] > all colors of its neighbors.
        # Standard greedy: process in order, color[i] = 1 + max{color[j] : j neighbor of i, j processed}
        # But we need neighbors processed before. Let's just compute: for each neighbor j < i, color[i] >= color[j]+1
        ci = 0
        for j in range(i):
            if (conflicts[i] >> j) & 1:
                if color[j] + 1 > ci:
                    ci = color[j] + 1
        color[i] = ci

    # Branch and bound: find independent set of size >= n_rows
    # We want to find ANY independent set of size n_rows (not necessarily max weight),
    # but to minimize scope, we prefer smaller-scope rows. Since rows are sorted by scope,
    # we search for an independent set of size n_rows using the smallest-scope rows first.
    #
    # Strategy: branch-and-bound to find an independent set of size n_rows.
    # State: (available mask, current set, depth)
    # Upper bound: min(popcount(available), coloring_bound(available))
    #
    # For coloring bound on a subgraph, we recompute greedily on the available nodes.
    # To keep it fast, we use a simpler bound: popcount of available, and a greedy coloring
    # on the available set.

    import sys
    sys.setrecursionlimit(10000)

    # Greedy coloring on a subset given by a mask, returns upper bound on independent set size
    def coloring_bound(mask):
        # Collect available node indices
        avail = []
        m = mask
        while m:
            lsb = m & (-m)
            idx = lsb.bit_length() - 1
            avail.append(idx)
            m ^= lsb
        if not avail:
            return 0
        # Greedy coloring in the order of avail
        col = {}
        maxc = 0
        for i in avail:
            ci = 0
            for j in avail:
                if j == i:
                    continue
                if (conflicts[i] >> j) & 1 and j in col:
                    if col[j] + 1 > ci:
                        ci = col[j] + 1
            col[i] = ci
            if ci > maxc:
                maxc = ci
        return maxc

    # Branch and bound
    found = None
    best_set = None

    def popcount(x):
        return bin(x).count('1')

    def search(mask, chosen, depth):
        nonlocal found, best_set
        if found:
            return
        if depth == n_rows:
            found = True
            best_set = list(chosen)
            return
        # Upper bound
        avail_count = popcount(mask)
        if avail_count < n_rows - depth:
            return
        # Coloring bound
        cb = coloring_bound(mask)
        if cb < n_rows - depth:
            return
        # Pick a node to branch on: choose one with fewest available neighbors (most constrained)
        # Or just pick the first available node (smallest scope)
        m = mask
        # Find first available node
        lsb = m & (-m)
        node = lsb.bit_length() - 1
        # Branch 1: include node
        new_mask = mask & ~(conflicts[node] | (1 << node))
        chosen.append(node)
        search(new_mask, chosen, depth + 1)
        chosen.pop()
        if found:
            return
        # Branch 2: exclude node
        new_mask2 = mask & ~(1 << node)
        search(new_mask2, chosen, depth)

    search((1 << n) - 1, [], 0)

    if found:
        result = [rows[i] for i in best_set]
        return [list(r) for r in result]
    else:
        # Fallback: greedy
        used = set()
        result = []
        for i in range(n):
            if len(result) == n_rows:
                break
            if not (diffsets[i] & used):
                result.append(rows[i])
                used |= diffsets[i]
        return [list(r) for r in result]
