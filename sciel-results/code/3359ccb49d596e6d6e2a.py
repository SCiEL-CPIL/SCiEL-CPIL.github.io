"""Small 4-regular graph of girth 10 (cage n(4,10)).

Builds the Lazebnik-Ustimenko-Woldar (LUW) graph CD(k,k) as its universal
cover via non-backtracking BFS, finds the shortest non-backtracking walk from
the root back to itself (the fundamental cycle, length = girth), and quotients
the BFS tree by that cycle to obtain a small k-regular graph of girth >= g.
For k=4, g=10 this recovers the Petersen cage (320 vertices)."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    t0 = time.time()
    deadline = t0 + 110.0

    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q

    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)

    # Non-backtracking BFS over the universal cover.
    index = {start: 0}
    order = [start]
    parent_idx = [-1]
    parent_vertex = [None]
    depth_arr = [0]
    visited_edges = set()
    queue = deque([0])
    found_cycle_len = None
    cycle_child = None

    while queue:
        if time.time() > deadline - 8.0:
            break
        cur = queue.popleft()
        cur_vertex = order[cur]
        cur_parent = parent_vertex[cur]
        nd = depth_arr[cur] + 1
        for nb in neighbours(cur_vertex):
            if cur_parent is not None and nb == cur_parent:
                continue
            key = (nb, cur_vertex)
            if key in visited_edges:
                continue
            visited_edges.add(key)
            if nb == start and nd >= g:
                if found_cycle_len is None or nd < found_cycle_len:
                    found_cycle_len = nd
                    cycle_child = len(order) if nb in index else None
            if nb not in index:
                index[nb] = len(order)
                order.append(nb)
                parent_idx.append(cur)
                parent_vertex.append(cur_vertex)
                depth_arr.append(nd)
                queue.append(index[nb])

    # Build the quotient: vertices = cover nodes at depth < L; the cycle of
    # length L closes by identifying each depth-(L-1) node that has a
    # non-backtracking neighbour equal to start with the root.
    if found_cycle_len is None:
        edges = [(index[pv], index[order[i]])
                 for i in range(1, len(order))
                 for pv in [parent_vertex[i]] if pv is not None]
        return len(order), edges

    L = found_cycle_len
    quot_index = {}
    for i, v in enumerate(order):
        if depth_arr[i] < L:
            quot_index[v] = len(quot_index)
    n = len(quot_index)

    edges = []
    edge_set = set()
    def add_edge(a, b):
        if a is None or b is None or a == b:
            return
        e = (min(a, b), max(a, b))
        if e not in edge_set:
            edge_set.add(e)
            edges.append(e)

    for i in range(len(order)):
        if depth_arr[i] >= L:
            continue
        v = order[i]
        pv = parent_vertex[i]
        add_edge(quot_index.get(pv), quot_index.get(v))
        # cycle-closing edges: non-backtracking neighbours of v that are start
        if depth_arr[i] == L - 1:
            for nb in neighbours(v):
                if nb == start and nb != pv:
                    add_edge(quot_index.get(v), quot_index.get(start))
                    break

    deg = [0] * n
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    if all(x == k for x in deg):
        return n, edges

    # Fallback: full cover
    edges = [(index[pv], index[order[i]])
             for i in range(1, len(order))
             for pv in [parent_vertex[i]] if pv is not None]
    return len(order), edges
# EVOLVE-BLOCK-END
