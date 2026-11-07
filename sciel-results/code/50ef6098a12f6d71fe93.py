"""Solver for few-color graph coloring with no bicolored cycles."""
import time

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _check_bicolored_cycles(coloring, adj):
    """Check if any pair of colors induces a cycle. Returns True if valid (no bicolored cycles)."""
    n = len(coloring)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                if coloring[u] == coloring[v]:
                    return False
    
    # Union-find per color pair
    from collections import defaultdict
    pair_edges = defaultdict(list)
    for u in range(n):
        for v in adj[u]:
            if u < v:
                cu, cv = coloring[u], coloring[v]
                pair_edges[(min(cu, cv), max(cu, cv))].append((u, v))
    
    for (c1, c2), edges in pair_edges.items():
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for a, b in edges:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
    return True

def _backtrack_coloring(adj, num_colors, deadline):
    """Backtracking with incremental union-find per color pair for bicolored cycle detection."""
    n = P['n']
    coloring = [-1] * n
    
    # Order vertices: prefer high degree, then prefer vertices that are in many cycles
    # Use a simple heuristic: degree descending
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    # For each color pair, maintain a union-find structure
    # Key: (c1, c2) with c1 < c2
    # Value: list of [parent, rank] for vertices, but we only track vertices with those colors
    # To keep it simple and fast, we use a dict mapping (pair, vertex) -> parent
    # Actually, let's use a simpler approach: for each pair, maintain a set of edges and
    # a union-find. When we add an edge, check if it creates a cycle.
    
    # Union-find state per color pair
    # uf_state[pair] = {'parent': {vertex: parent}, 'rank': {vertex: rank}}
    uf_state = {}
    
    def _uf_init(pair):
        uf_state[pair] = {'parent': {}, 'rank': {}}
    
    def _uf_find(pair, x):
        st = uf_state[pair]
        p = st['parent']
        if x not in p:
            p[x] = x
            st['rank'][x] = 0
            return x
        # Path compression
        root = x
        while p[root] != root:
            root = p[root]
        # Compress path
        while x != root:
            nxt = p[x]
            p[x] = root
            x = nxt
        return root
    
    def _uf_union(pair, a, b):
        st = uf_state[pair]
        p = st['parent']
        r = st['rank']
        if a not in p:
            p[a] = a
            r[a] = 0
        if b not in p:
            p[b] = b
            r[b] = 0
        ra = _uf_find(pair, a)
        rb = _uf_find(pair, b)
        if ra == rb:
            return False  # cycle
        if r[ra] < r[rb]:
            ra, rb = rb, ra
        p[rb] = ra
        if r[ra] == r[rb]:
            r[ra] += 1
        return True
    
    def _uf_rollback(pair, a, b, ra, rb, rank_ra_before):
        """Rollback a union operation. ra was merged into rb or vice versa."""
        st = uf_state[pair]
        p = st['parent']
        r = st['rank']
        # We need to undo: p[rb] = ra (assuming ra had higher/equal rank)
        # This is tricky with path compression. Let's use a different approach:
        # Instead of rolling back, we'll track the operations and redo.
        # For simplicity, let's just not use rollback but instead use a copy-on-write approach.
        # Actually, let's use a simpler strategy: for each assignment, we check all affected
        # pairs by doing a fresh connectivity check on just the affected subgraph.
        pass
    
    # Better approach: for each vertex assignment, check each affected color pair by
    # doing a BFS/DFS on the subgraph induced by those two colors, restricted to
    # vertices that are already colored. This is O(degree * subgraph_size) per check.
    # For degree-4 graph, this is manageable.
    
    def _has_path_in_pair(c1, c2, v1, v2, adj, coloring):
        """Check if there's a path from v1 to v2 in the subgraph induced by colors {c1, c2},
        using only already-colored vertices (excluding the direct edge v1-v2)."""
        if v1 == v2:
            return False
        # BFS from v1
        visited = set()
        visited.add(v1)
        stack = [v1]
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if nb == v1 and cur == v1:
                    continue  # skip direct edge back to start (handled by not adding v1)
                if nb == v2:
                    # Check if this is the direct edge v1-v2
                    if cur == v1:
                        continue  # skip direct edge
                    return True
                if coloring[nb] in (c1, c2) and nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        return False
    
    # Even better: use the incremental union-find but with proper rollback via
    # operation log per pair.
    
    # Let's use a hybrid: maintain union-find per pair, but when backtracking,
    # we pop the last operation. We'll log operations.
    
    # op_log[pair] = list of (a, b, ra_before, rb_before, rank_ra_before)
    op_log = {}
    
    def _log_op(pair, a, b, ra, rb, rank_ra_before):
        if pair not in op_log:
            op_log[pair] = []
        op_log[pair].append((a, b, ra, rb, rank_ra_before))
    
    def _pop_op(pair):
        if pair in op_log and op_log[pair]:
            a, b, ra, rb, rank_ra_before = op_log[pair].pop()
            st = uf_state[pair]
            p = st['parent']
            r = st['rank']
            # Undo: p[rb] was set to ra, r[ra] may have been incremented
            p[rb] = rb
            r[ra] = rank_ra_before
            # Note: path compression may have modified other entries, but since we
            # only care about connectivity (not exact tree structure), this is fine
            # as long as we don't query nodes that were compressed to rb.
            # This is a known issue with rollback + path compression.
            # To be safe, let's NOT use path compression in the incremental version.
        pass
    
    # Given the complexity of proper rollback with path compression, let's use
    # a cleaner approach: no path compression, just union by rank.
    # This makes find O(log n) which is fine for n=50.
    
    # Reset uf_state without path compression
    def _uf_find_no_compress(pair, x):
        st = uf_state[pair]
        p = st['parent']
        if x not in p:
            p[x] = x
            st['rank'][x] = 0
            return x
        while p[x] != x:
            x = p[x]
        return x
    
    def _uf_union_no_compress(pair, a, b):
        st = uf_state[pair]
        p = st['parent']
        r = st['rank']
        if a not in p:
            p[a] = a
            r[a] = 0
        if b not in p:
            p[b] = b
            r[b] = 0
        ra = _uf_find_no_compress(pair, a)
        rb = _uf_find_no_compress(pair, b)
        if ra == rb:
            return False, None, None, None
        if r[ra] < r[rb]:
            ra, rb = rb, ra
        p[rb] = ra
        if r[ra] == r[rb]:
            r[ra] += 1
        return True, ra, rb, r[ra]
    
    def _uf_rollback_no_compress(pair, ra, rb, rank_ra_after):
        st = uf_state[pair]
        p = st['parent']
        r = st['rank']
        p[rb] = rb
        r[ra] = rank_ra_after - 1 if r[ra] > rank_ra_after else rank_ra_after
        # Actually, we need to restore rank_ra_before
        pass
    
    # Let me simplify: use the BFS-based check for each affected pair.
    # For a degree-4 graph with 50 vertices, the BFS on a 2-color subgraph is at most
    # 50 vertices, so it's fast.
    
    def dfs(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return coloring[:]
        v = order[idx]
        # Try colors in order
        used_by_neighbors = set()
        for nb in adj[v]:
            if coloring[nb] != -1:
                used_by_neighbors.add(coloring[nb])
        
        # Collect affected color pairs for rollback
        for c in range(num_colors):
            if c in used_by_neighbors:
                continue
            coloring[v] = c
            # Check bicolored cycles: for each neighbor with a different color,
            # check if adding edge (v, nb) creates a cycle in the {c, cn} subgraph
            valid = True
            for nb in adj[v]:
                cn = coloring[nb]
                if cn == -1:
                    continue
                # Check if v and nb are already connected in {c, cn} subgraph
                # (excluding the direct edge v-nb)
                if _has_path_in_pair(c, cn, v, nb, adj, coloring):
                    valid = False
                    break
            if valid:
                result = dfs(idx + 1)
                if result is not None:
                    return result
            coloring[v] = -1
        return None
    
    return dfs(0)

def construct():
    deadline = time.time() + 85  # safety margin (evaluator kills at 90s)
    adj = _build_adj()
    
    # Try 3 colors first (minimum possible since graph has cycles)
    for nc in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        result = _backtrack_coloring(adj, nc, deadline)
        if result is not None:
            # Verify
            assert _check_bicolored_cycles(result, adj)
            return [int(x) for x in result]
    
    # Fallback: use all 50 colors (always valid since no two adjacent share a color,
    # and with all distinct colors, no color pair has more than one edge)
    return list(range(P['n']))
