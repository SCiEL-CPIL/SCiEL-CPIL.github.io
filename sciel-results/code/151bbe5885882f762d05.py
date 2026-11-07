"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    deadline = time.time() + 80
    n = P['n']
    edges = P['edges']
    
    # Build adjacency
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    # For each pair of colors, maintain union-find to detect bicolored cycles
    # color_pair_edges[(c1, c2)] = list of edges between vertices of color c1 and c2
    
    best_result = None
    
    for k in range(3, 20):
        if time.time() > deadline:
            break
        result = _backtrack(k, n, edges, adj, deadline)
        if result is not None:
            return [int(x) for x in result]
    
    # Fallback: use many colors (always valid since each color class is independent set or small)
    # With 50 colors, each vertex has unique color, no bicolored subgraph has >1 edge
    if best_result is not None:
        return [int(x) for x in best_result]
    return [int(i) for i in range(n)]

def _backtrack(k, n, edges, adj, deadline):
    """Try to color n vertices with k colors such that:
    1. Adjacent vertices have different colors
    2. For any pair of colors, the induced subgraph is a forest
    """
    color = [-1] * n
    # For each color pair (c1,c2) with c1<c2, track edges to detect cycles
    # Use union-find per color pair
    # We'll track: for each (c1,c2), a union-find structure over vertices of those colors
    
    # Simpler: maintain for each color pair, a set of edges, and check acyclicity
    # When adding a vertex v with color c, for each neighbor u already colored with color d:
    #   add edge (u,v) to pair (min(c,d), max(c,d))
    #   check if this creates a cycle in that pair's subgraph
    
    # For efficiency, use union-find per color pair
    # uf[(c1,c2)] = union-find dict for vertices of colors c1,c2
    
    uf = {}  # key: (c1,c2), value: dict vertex->parent
    uf_rank = {}  # key: (c1,c2), value: dict vertex->rank
    
    def uf_find(ufd, x):
        if x not in ufd:
            ufd[x] = x
        while ufd[x] != x:
            ufd[x] = ufd[ufd[x]]
            x = ufd[x]
        return x
    
    def uf_union(ufd, uf_rank_d, x, y):
        rx = uf_find(ufd, x)
        ry = uf_find(ufd, y)
        if rx == ry:
            return False  # cycle detected
        if uf_rank_d.get(rx, 0) < uf_rank_d.get(ry, 0):
            rx, ry = ry, rx
        ufd[ry] = rx
        if uf_rank_d.get(rx, 0) == uf_rank_d.get(ry, 0):
            uf_rank_d[rx] = uf_rank_d.get(rx, 0) + 1
        return True
    
    def add_vertex(v, c):
        """Add vertex v with color c. Return True if no cycle created."""
        for u in adj[v]:
            d = color[u]
            if d == -1 or d == c:
                continue
            c1, c2 = (c, d) if c < d else (d, c)
            key = (c1, c2)
            if key not in uf:
                uf[key] = {}
                uf_rank[key] = {}
            if not uf_union(uf[key], uf_rank[key], v, u):
                return False
        return True
    
    def remove_vertex(v, c):
        """Remove vertex v with color c. (Hard to undo union-find, so we'll use a different approach)"""
        pass
    
    # Since undoing union-find is hard, let's use a different strategy:
    # Process vertices in a fixed order, and use backtracking with full state restoration
    
    # Alternative: use a simpler check - for each color pair, maintain the edge set
    # and check acyclicity via DFS when needed (expensive but correct)
    
    # Let's try a different approach: greedy + local search
    # Or: use the fact that the graph is small (50 vertices) and do full backtracking
    # with careful state management
    
    # Actually, let's just do backtracking with explicit state copying for small k
    # For k up to ~8, the state space is manageable with good ordering
    
    # Vertex ordering: try to order by degree (descending)
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    # Map: position in order -> vertex
    pos_to_v = order
    v_to_pos = [0] * n
    for i, v in enumerate(order):
        v_to_pos[v] = i
    
    # Backtracking over positions in order
    color = [-1] * n
    # For cycle detection, maintain for each color pair a set of edges
    # and use union-find that we can rebuild
    
    # Simpler: when assigning color to vertex v at position p, check all previously
    # assigned neighbors. For each pair of colors involved, verify no cycle.
    # To check if adding edge (u,v) to color pair (c1,c2) creates a cycle:
    #   In the subgraph induced by colors c1,c2, is there already a path from u to v?
    #   We can do BFS/DFS on the subgraph of vertices colored c1 or c2.
    
    def check_no_cycle(c1, c2, u, v):
        """Check if adding edge (u,v) to subgraph of colors c1,c2 creates a cycle.
        i.e., is there already a path from u to v in the c1,c2 subgraph (excluding this edge)?"""
        # BFS from u to v in subgraph induced by vertices with color c1 or c2
        # Only traverse edges between vertices both colored c1 or c2
        if u == v:
            return False
        visited = {u}
        queue = [u]
        while queue:
            cur = queue.pop(0)
            for w in adj[cur]:
                if w == v:
                    return True  # found path, adding edge creates cycle
                cw = color[w]
                if cw in (c1, c2) and w not in visited:
                    visited.add(w)
                    queue.append(w)
        return False
    
    def assign(v, c):
        """Try to assign color c to vertex v. Return True if valid."""
        for u in adj[v]:
            d = color[u]
            if d == -1:
                continue
            if d == c:
                return False
            c1, c2 = (c, d) if c < d else (d, c)
            if check_no_cycle(c1, c2, v, u):
                return False
        color[v] = c
        return True
    
    def unassign(v):
        color[v] = -1
    
    def backtrack(pos):
        if time.time() > deadline:
            return None
        if pos == n:
            return color[:]
        v = pos_to_v[pos]
        for c in range(k):
            if assign(v, c):
                result = backtrack(pos + 1)
                if result is not None:
                    return result
                unassign(v)
        return None
    
    result = backtrack(0)
    return result
# EVOLVE-BLOCK-END
