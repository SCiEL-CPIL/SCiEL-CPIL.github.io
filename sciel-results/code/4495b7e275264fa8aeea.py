"""Combined program: LZW algebraic + tree-based greedy + bipartite random greedy."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import deque


def construct(k, g):
    """
    Construct a k-regular graph of girth at least g.
    Returns (n, edges).
    """
    deadline = time.time() + 112  # Safety margin

    best = None

    # 1) Fast algebraic LZW construction (deterministic, valid upper bound)
    try:
        lzw = _lzw_construct(k, g)
        if lzw is not None:
            best = lzw
    except Exception:
        best = None

    # 2) Tree-based greedy: build the k-regular tree of radius r, then add edges
    #    between leaves at tree-distance >= g-1 to achieve k-regularity.
    if time.time() < deadline:
        r = (g - 1) // 2
        # For odd g=9: r=4. Tree has 1 + k*sum((k-1)^i, i=0..r-1) vertices
        # Leaves need (k-1) more edges each.
        # Total vertices in tree = Moore bound for odd girth.
        n_tree = 1 + k * sum((k - 1) ** i for i in range(r))
        # Each leaf needs (k-1) additional edges.
        # n_leaves = k*(k-1)^(r-1)
        n_leaves = k * (k - 1) ** (r - 1)
        # Total edges to add = n_leaves * (k-1) / 2
        # This creates a k-regular graph on n_tree vertices IF we can pair leaves
        # such that each leaf gets exactly (k-1) new edges and all pairs are at
        # tree-distance >= g-1.
        
        # For the tree of radius r, two leaves have tree-distance = 2*r - 2*depth(LCA)
        # where depth is from root. Max distance between leaves = 2r (opposite sides).
        # For g=9, r=4: max leaf distance = 8 = g-1. So we need distance >= g-1 = 8.
        # Only leaves at MAXIMUM distance (opposite branches from root) qualify!
        # This is very restrictive. Let's check: for k=5, r=4:
        # Leaves at level 4. Two leaves are at distance 8 iff their LCA is the root.
        # Each leaf at level 4 has a unique "branch" from root (one of k branches at level 1).
        # Leaves in different root-branches have LCA = root, distance = 8.
        # Leaves in the same root-branch have LCA at level >= 1, distance <= 6.
        # So valid pairs are only between leaves in DIFFERENT root-branches.
        # Each root-branch has (k-1)^(r-1) = 4^3 = 64 leaves.
        # Total leaves = 5 * 64 = 320.
        # Each leaf can connect to leaves in the other 4 branches: 4*64 = 256 candidates.
        # Each leaf needs exactly (k-1)=4 new edges.
        # Total edges to add = 320*4/2 = 640.
        # This is a perfect matching problem between leaves across branches!
        
        tree_result = _tree_greedy(k, g, r, n_tree, n_leaves, deadline)
        if tree_result is not None:
            if best is None or tree_result[0] < best[0]:
                best = tree_result

    # 3) Bipartite random greedy for larger n (if tree approach failed)
    if time.time() < deadline - 10 and best is not None:
        base = _moore_bound(k, g)
        cap = best[0] if best else 2000
        n = max(base + 2, 600)
        while n <= cap and time.time() < deadline - 10:
            res = _bipartite_greedy(k, g, n, deadline - 10)
            if res is not None:
                if best is None or res[0] < best[0]:
                    best = res
                break
            n = int(n * 1.15) + 2

    if best is not None:
        return best

    # 4) Final fallback
    return _lzw_construct(k, g)


def _moore_bound(k, g):
    """Moore bound (lower bound) for n(k,g)."""
    if g % 2 == 0:
        d = g // 2
        n = k * (k - 1) ** (d - 1)
    else:
        d = (g - 1) // 2
        n = 1 + k * sum((k - 1) ** i for i in range(d))
    return n


def _tree_greedy(k, g, r, n_tree, n_leaves, deadline):
    """
    Tree-based greedy: build k-regular tree of radius r, then add edges between
    leaves at tree-distance >= g-1 to achieve k-regularity.
    
    For odd g = 2r+1: tree has girth >= 2r+1 = g.
    Adding edge (u,v) creates cycle of length tree_dist(u,v) + 1.
    Need tree_dist(u,v) + 1 >= g => tree_dist(u,v) >= g-1 = 2r.
    In the tree of radius r, max distance is 2r (between leaves in opposite root-branches).
    So only pairs at EXACTLY distance 2r are valid.
    
    This means: leaves in different root-branches (LCA = root) are at distance 2r.
    Leaves in the same root-branch are at distance <= 2r-2.
    
    So we need a regular bipartite-like matching: each leaf connects to (k-1) leaves
    in OTHER root-branches.
    """
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline - 3:
            return None

        # Build tree structure
        # Level 0: vertex 0 (root)
        # Level i (i>=1): k*(k-1)^(i-1) vertices
        # Vertex numbering: level 0 = [0], level 1 = [1..k], level 2 = [k+1..k+k*(k-1)], etc.
        
        level_start = [0]
        for i in range(1, r + 1):
            level_start.append(level_start[-1] + k * (k - 1) ** (i - 1))
        
        # parent[v] = parent vertex
        # branch[v] = which root-branch (0..k-1) the vertex belongs to
        parent = [0] * n_tree
        branch = [0] * n_tree
        level = [0] * n_tree
        tree_edges = []
        
        parent[0] = -1
        branch[0] = -1
        level[0] = 0
        
        for lvl in range(1, r + 1):
            start = level_start[lvl]
            count = k * (k - 1) ** (lvl - 1)
            for i in range(count):
                v = start + i
                level[v] = lvl
                if lvl == 1:
                    p = 0
                    b = i
                else:
                    # Each vertex at level lvl-1 has (k-1) children
                    # Vertex i at level lvl: parent is at position i//(k-1) in level lvl-1
                    prev_start = level_start[lvl - 1]
                    p_idx = i // (k - 1)
                    p = prev_start + p_idx
                    b = branch[p]  # inherit branch
                parent[v] = p
                branch[v] = b
                tree_edges.append((p, v))
        
        # Leaves are at level r
        leaf_start = level_start[r]
        leaves = list(range(leaf_start, n_tree))
        
        # Group leaves by branch
        branch_leaves = [[] for _ in range(k)]
        for v in leaves:
            branch_leaves[branch[v]].append(v)
        
        # Each branch has (k-1)^(r-1) leaves
        # Each leaf needs (k-1) edges to leaves in other branches
        # Total edges: n_leaves * (k-1) / 2
        
        # Build adjacency for the final graph
        adj = [set() for _ in range(n_tree)]
        for u, v in tree_edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # For each leaf, it needs (k-1) more neighbors
        # Valid neighbors: leaves in other branches (tree-distance = 2r = g-1)
        # We need to find a (k-1)-regular "matching" between leaves across branches
        
        # Strategy: for each leaf, randomly pick (k-1) leaves from other branches
        # ensuring no duplicates and each leaf ends up with exactly (k-1) new edges
        
        success = True
        # Shuffle leaves within each branch
        for b in range(k):
            random.shuffle(branch_leaves[b])
        
        # For each leaf, assign (k-1) partners from other branches
        # Use a greedy approach with backtracking-like restarts
        
        new_edges = []
        deg_new = [0] * n_tree  # degree from new edges
        
        for v in leaves:
            if deg_new[v] >= (k - 1):
                continue
            b_v = branch[v]
            # Candidates: leaves in other branches with deg_new < (k-1)
            candidates = []
            for b in range(k):
                if b == b_v:
                    continue
                for w in branch_leaves[b]:
                    if deg_new[w] < (k - 1) and w != v and w not in adj[v]:
                        candidates.append(w)
            
            if len(candidates) < (k - 1) - deg_new[v]:
                success = False
                break
            
            # Pick (k-1) - deg_new[v] candidates, preferring those with lower deg_new
            needed = (k - 1) - deg_new[v]
            candidates.sort(key=lambda w: deg_new[w])
            # Take from the pool, but ensure diversity
            # Use random selection from the lower-degree half for balance
            half = max(1, len(candidates) // 2)
            pool = candidates[:half] if len(candidates) > 2 else candidates
            chosen = random.sample(pool, min(needed, len(pool)))
            if len(chosen) < needed:
                # Fill from remaining
                remaining = [c for c in candidates if c not in chosen]
                chosen += random.sample(remaining, needed - len(chosen))
            
            for w in chosen:
                if w not in adj[v]:
                    adj[v].add(w)
                    adj[w].add(v)
                    deg_new[v] += 1
                    deg_new[w] += 1
                    new_edges.append((v, w))
        
        if not success:
            continue
        
        # Verify all leaves have exactly (k-1) new edges
        if any(deg_new[v] != (k - 1) for v in leaves):
            continue
        
        # Verify all vertices have degree k
        if any(len(adj[v]) != k for v in range(n_tree)):
            continue
        
        # Build edge list
        edges = []
        for v in range(n_tree):
            for w in adj[v]:
                if v < w:
                    edges.append((v, w))
        
        # Verify girth
        if _verify_girth_fast(n_tree, edges, k, g):
            return (n_tree, edges)

    return None


def _bipartite_greedy(k, g, n, deadline):
    """Bipartite random greedy construction."""
    n_half = n // 2
    if 2 * n_half != n:
        n = 2 * n_half

    min_dist = -(-g // 2) - 1  # ceil(g/2) - 1

    max_restarts = 5
    for _ in range(max_restarts):
        if time.time() > deadline - 2:
            return None

        adj = [set() for _ in range(n)]
        deg = [0] * n

        order = list(range(n))
        random.shuffle(order)

        success = True
        for u in order:
            if deg[u] >= k:
                continue
            side = 0 if u < n_half else 1
            valid_vs = []
            for v in range(n_half, n) if side == 0 else range(n_half):
                if deg[v] >= k:
                    continue
                if v in adj[u]:
                    continue
                d = _bipartite_distance(adj, u, v, min_dist)
                if d >= min_dist:
                    valid_vs.append(v)

            if not valid_vs:
                success = False
                break

            valid_vs.sort(key=lambda v: deg[v])
            half = len(valid_vs) // 2
            pool = valid_vs[:max(1, half)]
            v = random.choice(pool)

            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

        if not success:
            continue

        if any(d != k for d in deg):
            continue

        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))

        if _verify_girth_fast(n, edges, k, g):
            return (n, edges)

    return None


def _bipartite_distance(adj, u, v, limit):
    """BFS distance from u to v, capped at limit."""
    if u == v:
        return 0
    dist = {u: 0}
    queue = deque([u])
    while queue:
        node = queue.popleft()
        d = dist[node]
        if d >= limit:
            continue
        for nb in adj[node]:
            if nb == v:
                return d + 1
            if nb not in dist:
                dist[nb] = d + 1
                queue.append(nb)
    return limit + 1


def _verify_girth_fast(n, edges, k, g):
    """Verify k-regular and girth >= g."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != k:
            return False

    edge_set = set()
    for u, v in edges:
        key = (min(u, v), max(u, v))
        if key in edge_set:
            return False
        edge_set.add(key)

    min_girth = float('inf')
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    parent[nb] = node
                    queue.append(nb)
                elif nb != parent[node]:
                    cycle_len = dist[node] + dist[nb] + 1
                    if cycle_len < min_girth:
                        min_girth = cycle_len
                        if min_girth < g:
                            return False
    return min_girth >= g


def _lzw_construct(k, g):
    """Fallback: Lazebnik-Ustimenko-Woldar graph"""
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
# EVOLVE-BLOCK-END
