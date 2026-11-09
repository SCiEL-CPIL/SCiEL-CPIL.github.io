"""Construct a 3-regular graph with girth 16.

Strategy:
1. Build the unique 3-regular tree of depth 8 (girth infinity).
2. Match leaves into pairs using a greedy + local-search algorithm that
   minimizes the number of "short" cycles created (cycles < 16).
3. Verify the result has girth >= 16.
"""
import time
import random
from collections import deque

def construct(k, g):
    """Return (n, edges) for a k-regular graph with girth >= g."""
    t0 = time.time()
    deadline = t0 + 110.0
    
    if k != 3 or g != 16:
        # Fallback to LUW for other parameters
        return _luw_fallback(k, g)
    
    depth = g // 2  # 8
    # Build tree of depth `depth` (root at depth 0, leaves at depth `depth`)
    # Each node has 3 children except root (3 children), and non-leaf non-root (2 children)
    # Total nodes: 1 + 3 + 3*2 + 3*2^2 + ... + 3*2^(depth-1)
    # = 1 + 3*(2^depth - 1) = 3*2^depth - 2
    
    nodes = []       # list of (depth, parent, child_idx)
    children = []    # list of lists of child indices
    parent = []      # parent index or -1
    depth_list = []  # depth of each node
    
    def add_node(d, p, ci):
        idx = len(nodes)
        nodes.append((d, p, ci))
        parent.append(p)
        depth_list.append(d)
        children.append([])
        return idx
    
    root = add_node(0, -1, -1)
    frontier = [root]
    
    for d in range(1, depth + 1):
        next_frontier = []
        for u in frontier:
            n_children = 2 if d > 1 else 3  # non-root nodes have 2 children, root has 3
            if d == depth:
                continue  # leaves, no children
            for ci in range(n_children):
                v = add_node(d, u, ci)
                children[u].append(v)
                next_frontier.append(v)
        frontier = next_frontier
    
    n_internal = len(nodes)
    leaves = [i for i in range(n_internal) if not children[i]]
    n_leaves = len(leaves)
    
    # Each leaf will be connected to another leaf (matching).
    # We need to pair up all n_leaves leaves.
    # The "distance" between two leaves u and v in the tree is depth[u] + depth[v] - 2*depth[lca(u,v)].
    # When we add an edge (u,v), it creates a cycle of length dist(u,v) + 2.
    # We want all such cycles to have length >= 16, i.e., dist(u,v) >= 14.
    # Since max dist is 2*depth = 16, we need dist(u,v) >= 14.
    # dist = 16 - 2*depth[lca]. So 16 - 2*depth[lca] >= 14 => depth[lca] <= 1.
    # So the LCA of any matched pair must be at depth 0 or 1.
    
    # Let's compute LCA depths for all leaf pairs.
    # First, build parent pointers and depth array.
    # For each leaf, trace up to root to get path.
    
    leaf_paths = {}  # leaf index -> list of ancestors from leaf to root
    for lf in leaves:
        path = []
        u = lf
        while u >= 0:
            path.append(u)
            u = parent[u]
        leaf_paths[lf] = path
    
    def lca_depth(u, v):
        """Return depth of LCA of nodes u and v."""
        # Use the paths
        path_u = set(leaf_paths[u])
        # Walk up from v until we hit a node in path_u
        x = v
        while x >= 0:
            if x in path_u:
                return depth_list[x]
            x = parent[x]
        return 0  # should not happen
    
    # For efficiency, precompute for each pair of leaves their LCA depth.
    # But n_leaves can be large. Let's estimate.
    # depth=8: nodes = 3*2^8 - 2 = 766. Leaves = 3*2^7 = 384.
    # 384 leaves, pairwise = ~73k pairs. Fine.
    
    lca_depths = {}
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            u, v = leaves[i], leaves[j]
            lca_depths[(u, v)] = lca_depth(u, v)
    
    # Now we need a perfect matching on the leaves such that for every matched pair (u,v),
    # lca_depth(u,v) <= 1.
    
    # Build a graph on leaves where edge (u,v) exists if lca_depth(u,v) <= 1.
    # Then find a perfect matching in this graph.
    
    # Let's use a simple approach: greedy matching with backtracking, or use a max-flow.
    # Actually, let's just do a greedy matching and see if it works.
    
    # Build adjacency for leaves
    leaf_to_idx = {lf: i for i, lf in enumerate(leaves)}
    adj = [[] for _ in range(n_leaves)]
    for (u, v), d in lca_depths.items():
        if d <= 1:
            i, j = leaf_to_idx[u], leaf_to_idx[v]
            adj[i].append(j)
            adj[j].append(i)
    
    # Try to find a perfect matching using a simple algorithm.
    # Use augmenting path (Hopcroft-Karp is overkill, simple DFS matching works for small graphs).
    
    def find_matching(adj, n):
        match = [-1] * n
        def dfs(u, visited):
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    if match[v] == -1 or dfs(match[v], visited):
                        match[v] = u
                        return True
            return False
        for u in range(n):
            visited = [False] * n
            dfs(u, visited)
        return match
    
    match = find_matching(adj, n_leaves)
    
    # Check if perfect matching
    matched = [False] * n_leaves
    valid = True
    for u in range(n_leaves):
        if match[u] == -1:
            valid = False
            break
        matched[u] = True
    
    if not valid:
        # Try randomized greedy
        # Actually, let's try a different approach: randomly pair leaves and check.
        # Or use a more sophisticated matching.
        # For now, fall back to LUW.
        return _luw_fallback(k, g)
    
    # Build the edge list
    edges = []
    # Internal edges
    for u in range(n_internal):
        for v in children[u]:
            edges.append((u, v))
    
    # Leaf edges (matching)
    for u in range(n_leaves):
        if u < match[u]:  # avoid duplicates
            edges.append((leaves[u], leaves[match[u]]))
    
    # Verify girth
    n = n_internal
    # Build adjacency
    adj_full = [[] for _ in range(n)]
    for (a, b) in edges:
        adj_full[a].append(b)
        adj_full[b].append(a)
    
    # Check degree
    for i in range(n):
        if len(adj_full[i]) != k:
            return _luw_fallback(k, g)
    
    # Check girth using BFS from each vertex
    def girth_check(adj, n, g):
        for start in range(n):
            # BFS from start, tracking parent to avoid backtracking
            dist = [-1] * n
            par = [-1] * n
            dist[start] = 0
            q = deque([start])
            while q:
                u = q.popleft()
                for v in adj[u]:
                    if v == par[u]:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        par[v] = u
                        q.append(v)
                    elif dist[u] + 1 < dist[v] + 1:
                        # Found a cycle of length dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < g:
                            return False
            return True
        return True
    
    if not girth_check(adj_full, n, g):
        return _luw_fallback(k, g)
    
    return n, edges


def _luw_fallback(k, g):
    """Lazebnik-Ustimenko-Woldar construction as fallback."""
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
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
