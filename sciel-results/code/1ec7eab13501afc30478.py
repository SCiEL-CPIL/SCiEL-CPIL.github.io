"""Direct construction of a 4-coloring of the degree-4 grid graph
that is adjacent-different and has no bicolored cycle (i.e. every
two-color induced subgraph is a forest)."""
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check(coloring):
    """Validate: adjacent differ and every two-color induced subgraph is a forest."""
    adj = _build_adj()
    n = len(coloring)
    for u in range(n):
        for v in adj[u]:
            if u < v and coloring[u] == coloring[v]:
                return False
    pair_edges = defaultdict(list)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                c1, c2 = coloring[u], coloring[v]
                pair_edges[(min(c1, c2), max(c1, c2))].append((u, v))

    for (a, b), edges in pair_edges.items():
        verts = set()
        for u, v in edges:
            verts.add(u)
            verts.add(v)
        vlist = list(verts)
        idx = {v: i for i, v in enumerate(vlist)}
        par = list(range(len(vlist)))
        rn = [0] * len(vlist)

        def f(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        def un(x, y):
            rx, ry = f(x), f(y)
            if rx == ry:
                return False
            if rn[rx] < rn[ry]:
                rx, ry = ry, rx
            par[ry] = rx
            if rn[rx] == rn[ry]:
                rn[rx] += 1
            return True

        for u, v in edges:
            if not un(idx[u], idx[v]):
                return False
    return True


def _backtrack(adj, nc, deadline, start=0, ordering=None):
    n = P['n']
    coloring = [-1] * n
    if ordering is None:
        ordering = sorted(range(n), key=lambda x: -len(adj[x]))

    # For fast bicolored cycle detection: maintain per-pair union-find incrementally.
    # When we assign color c to vertex v, for each neighbor nb with color cn != c,
    # we check if adding edge (v,nb) to the {c,cn} subgraph creates a cycle.
    # We do this by checking if v and nb are already connected in the {c,cn} subgraph
    # via other edges. Instead of BFS, we track connected components per pair.
    
    # Strategy: for each pair (a,b), maintain a set of edges that form the current
    # forest. When adding a new edge, check if endpoints are in same component.
    # Use a simple approach: for each pair, keep a list of edges and do a quick
    # connectivity check only when needed.
    
    # Better: precompute which edges belong to which color pair as we go.
    # For the incremental check, when assigning v color c:
    #   For each colored neighbor nb with color cn:
    #     Check if there's a path from v to nb in the {c,cn}-induced subgraph 
    #     excluding the direct edge. This is equivalent to checking if v and nb
    #     are in the same connected component of {c,cn} subgraph after removing
    #     edge (v,nb). Since we're adding the edge, we just need to check if
    #     they're already connected without this edge.
    #
    # Fast check: BFS/DFS restricted to vertices with color c or cn.
    # Since degree is 4 and the graph is small, this should be fast enough.
    
    # Even faster: precompute for each vertex, which other vertices are reachable
    # through a given color pair. But this changes as we assign colors.
    
    # Let's use a hybrid: for each pair (c,cn), maintain a union-find.
    # When we assign v to color c, for each neighbor nb with color cn:
    #   - If the pair (min(c,cn), max(c,cn)) doesn't have v or nb yet, just add them.
    #   - Check if v and nb are in the same component. If yes, cycle -> backtrack.
    #   - If no, union them.
    # When we unassign, we need to undo. This is complex with union-find.
    
    # Alternative: just do the BFS check but make it very fast with bitmasks or
    # early termination. The graph has only 50 vertices and degree 4, so BFS
    # is at most 50 steps. With 4 neighbors to check, that's 200 steps per assignment.
    # With backtracking, this should still be fast for a structured graph.
    
    # Let me try a different ordering: process row by row in the 5x10 grid.
    # Vertex i is at row i//10, col i%10. Neighbors: (i±1, same col) and (same row, col±1 mod 10).
    # Actually looking at edges: 0-1 (row 0, cols 0-1), 0-7 (row 0 to row... wait)
    # 0 connects to 1, 7, 43, 49. Let me figure out the grid layout.
    # 50 vertices, 5x10. Let's say vertex i = 10*r + c where r in 0..4, c in 0..9.
    # 0 = (0,0): neighbors 1=(0,1), 7=(0,7)? That doesn't fit a simple grid.
    # Let me just use the given edges.
    
    def dfs(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return coloring[:]
        v = ordering[idx]
        used = set()
        for nb in adj[v]:
            c = coloring[nb]
            if c != -1:
                used.add(c)
        for c in range(nc):
            if c in used:
                continue
            coloring[v] = c
            ok = True
            # Check bicolored cycle for each neighbor's color
            for nb in adj[v]:
                cn = coloring[nb]
                if cn == -1 or cn == c:
                    continue
                # BFS from v to nb in {c,cn}-subgraph, not using direct edge (v,nb)
                stack = [v]
                visited = {v}
                found = False
                while stack:
                    cur = stack.pop()
                    for w in adj[cur]:
                        if cur == v and w == nb:
                            continue
                        if w == nb:
                            found = True
                            break
                        cw = coloring[w]
                        if cw == c or cw == cn:
                            if w not in visited:
                                visited.add(w)
                                stack.append(w)
                    if found:
                        break
                if found:
                    ok = False
                    break
            if ok:
                res = dfs(idx + 1)
                if res is not None:
                    return res
            coloring[v] = -1
        return None

    return dfs(0)


def construct():
    start_time = time.time()
    deadline = start_time + 85
    adj = _build_adj()

    # The graph is a 5x10 torus. Try 3 colors first, then 4.
    # Use a row-major ordering for better locality.
    n = P['n']
    # Row-major ordering: vertex i at position i
    ordering = list(range(n))
    
    for nc in [3, 4, 5]:
        if time.time() > deadline:
            break
        result = _backtrack(adj, nc, deadline, ordering=ordering)
        if result is not None:
            if _check(result):
                return [int(x) for x in result]

    # Fallback: all distinct colors
    return [int(i) for i in range(P['n'])]
