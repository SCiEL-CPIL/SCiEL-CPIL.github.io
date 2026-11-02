"""Cage graph construction: build k-regular graph, iteratively remove short-cycle edges."""
import sys
from collections import defaultdict, deque

# EVOLVE-BLOCK-START

def construct(k, g):
    """Construct a k-regular graph with girth >= g on as few vertices as possible."""
    
    def moore_lower_bound(k, g):
        if g % 2 == 1:
            # odd girth: 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
            h = (g - 3) // 2
            s = 0
            for i in range(h + 1):
                s += (k - 1) ** i
            return 1 + k * s
        else:
            # even girth: 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
            h = g // 2 - 1
            s = 0
            for i in range(h + 1):
                s += (k - 1) ** i
            return 2 * s
    
    lb = moore_lower_bound(k, g)
    
    # Target: try to get close to known best (1116 for n(5,9))
    # Start with a reasonable n
    if k == 5 and g == 9:
        n_target = 1200  # slightly above known best of 1116
    else:
        n_target = max(lb * 3, 100)
    
    # Try different n values
    best_result = None
    best_n = float('inf')
    
    for n_try in [n_target, n_target - 50, n_target + 50, 1116, 1150]:
        if n_try < lb:
            continue
        result = _build_cage(k, g, n_try)
        if result is not None:
            n_val, edges = result
            if n_val < best_n:
                best_n = n_val
                best_result = result
    
    if best_result is not None:
        return best_result
    
    # Fallback: use a simpler construction
    return _fallback_cage(k, g)


def _build_cage(k, g, n):
    """Build a k-regular graph with girth >= g on n vertices."""
    
    # Step 1: Build initial k-regular graph
    # Use a circulant-like construction for even n, or a different approach
    edges = _build_initial_regular(k, n)
    if edges is None:
        return None
    
    # Build adjacency
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Verify k-regular
    for v in range(n):
        if len(adj[v]) != k:
            return None
    
    # Step 2: Iteratively remove edges that are in short cycles
    # An edge (u,v) is in a cycle of length < g iff there's a path from u to v 
    # of length < g-1 not using edge (u,v)
    
    max_iter = 200
    for iteration in range(max_iter):
        removed_any = False
        
        # Find all edges that are in cycles shorter than g
        bad_edges = set()
        
        for u in range(n):
            for v in adj[u]:
                if u > v:
                    continue
                # Check if edge (u,v) is in a cycle of length < g
                # i.e., is there a path from u to v of length < g-1 not using (u,v)?
                # BFS from u avoiding edge (u,v), see if we reach v in < g-1 steps
                if _in_short_cycle(u, v, adj, g, n):
                    bad_edges.add((u, v))
        
        if not bad_edges:
            break
        
        # Remove one bad edge at a time (or all?)
        # Remove all bad edges
        for (u, v) in bad_edges:
            if v in adj[u]:
                adj[u].remove(v)
                adj[v].remove(u)
                removed_any = True
        
        if not removed_any:
            break
        
        # Check degrees - some vertices may now have degree < k
        # We need to restore k-regularity
        under_degree = [v for v in range(n) if len(adj[v]) < k]
        if under_degree:
            # Try to add back edges to restore k-regularity
            if not _restore_regularity(adj, k, g, n):
                return None
        
        # Rebuild edge list
        edges = [(u, v) for u in range(n) for v in adj[u] if u < v]
    
    # Final verification
    if not _verify_cage(k, g, n, edges):
        return None
    
    return n, edges


def _in_short_cycle(u, v, adj, g, n):
    """Check if edge (u,v) lies in a cycle of length < g.
    This is true iff there's a path from u to v of length < g-1 that doesn't use edge (u,v)."""
    # BFS from u, avoiding edge (u,v), see if we can reach v in < g-1 steps
    if g <= 3:
        return False
    
    max_dist = g - 2  # path length must be < g-1, so <= g-2
    
    # BFS
    visited = {u: 0}
    parent = {u: None}
    queue = deque([u])
    
    while queue:
        curr = queue.popleft()
        dist = visited[curr]
        if dist >= max_dist:
            continue
        
        for nbr in adj[curr]:
            # Skip the edge (u,v) itself
            if (curr == u and nbr == v) or (curr == v and nbr == u):
                continue
            if nbr not in visited:
                visited[nbr] = dist + 1
                parent[nbr] = curr
                if nbr == v:
                    return True
                if dist + 1 < max_dist:
                    queue.append(nbr)
    
    return False


def _restore_regularity(adj, k, g, n):
    """Try to restore k-regularity by adding edges that don't create short cycles."""
    # Find vertices with degree < k
    under = [v for v in range(n) if len(adj[v]) < k]
    if not under:
        return True
    
    # Find candidate edges to add
    # For each pair (u,v) where u,v both have degree < k and u,v not adjacent,
    # check if adding (u,v) creates a cycle shorter than g
    candidates = []
    for i, u in enumerate(under):
        for v in under[i+1:]:
            if v not in adj[u]:
                candidates.append((u, v))
    
    # Also consider edges from under-degree vertices to full-degree vertices? No, that would make those > k
    # Actually we can only add edges between two under-degree vertices, or we need to do a swap
    
    # Simple approach: try to add candidate edges that don't create short cycles
    for (u, v) in candidates:
        if len(adj[u]) >= k or len(adj[v]) >= k:
            continue
        # Check if adding (u,v) creates a cycle shorter than g
        # This happens iff there's already a path from u to v of length < g-1
        if _path_exists(u, v, adj, g - 1, n):
            continue
        # Safe to add
        adj[u].add(v)
        adj[v].add(u)
        if len(adj[u]) >= k and len(adj[v]) >= k:
            continue
    
    # Check if all vertices are now k-regular
    for v in range(n):
        if len(adj[v]) != k:
            return False
    return True


def _path_exists(u, v, adj, max_len, n):
    """Check if there's a path from u to v of length < max_len (i.e., <= max_len - 1)."""
    if u == v:
        return False
    visited = {u: 0}
    queue = deque([u])
    while queue:
        curr = queue.popleft()
        dist = visited[curr]
        if dist >= max_len - 1:
            continue
        for nbr in adj[curr]:
            if nbr not in visited:
                visited[nbr] = dist + 1
                if nbr == v:
                    return True
                if dist + 1 < max_len - 1:
                    queue.append(nbr)
    return False


def _build_initial_regular(k, n):
    """Build an initial k-regular graph on n vertices."""
    if n < k + 1:
        return None
    
    # Try a circulant-like construction
    # For k-regular, we can use a circulant graph C(n; 1, 2, ..., k/2) if k is even
    # or C(n; 1, 2, ..., (k-1)/2, n/2) if k is odd and n is even
    
    edges = set()
    
    if k % 2 == 0:
        # Even k: use offsets 1, 2, ..., k/2
        for i in range(1, k // 2 + 1):
            for j in range(n):
                u = j
                v = (j + i) % n
                if u != v:
                    edge = (min(u, v), max(u, v))
                    edges.add(edge)
    else:
        # Odd k: need n even
        if n % 2 != 0:
            return None
        half = n // 2
        # Use offsets 1, 2, ..., (k-1)/2 and the antipodal offset n/2
        for i in range(1, (k - 1) // 2 + 1):
            for j in range(n):
                u = j
                v = (j + i) % n
                if u != v:
                    edge = (min(u, v), max(u, v))
                    edges.add(edge)
        # Add antipodal edges
        for j in range(half):
            edge = (j, j + half)
            edges.add(edge)
    
    # Check if it's k-regular
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    for v in range(n):
        if len(adj[v]) != k:
            return None
    
    return list(edges)


def _verify_cage(k, g, n, edges):
    """Verify the graph is k-regular and has girth >= g."""
    adj = defaultdict(set)
    for u, v in edges:
        if u == v:
            return False
        adj[u].add(v)
        adj[v].add(u)
    
    # Check k-regular
    for v in range(n):
        if len(adj[v]) != k:
            return False
    
    # Check no cycles shorter than g using non-backtracking BFS from each vertex
    for start in range(n):
        # Non-backtracking BFS: state is (node, prev_node)
        # We want to find any cycle of length < g
        # A cycle of length L corresponds to a closed non-backtracking walk of length L
        visited_nb = set()
        queue = deque()
        
        for nbr in adj[start]:
            queue.append((nbr, start))
            visited_nb.add((nbr, start))
        
        while queue:
            curr, prev = queue.popleft()
            # Count distance from start
            # Actually, we need to track distance
            pass
        
        # Better: BFS tracking (node, prev, distance)
        # A cycle of length < g exists iff we can get back to start in < g steps
        # via non-backtracking walk
        
        # Let's do it properly: for each start, do non-backtracking BFS
        # State: (current, previous, steps)
        # We found a cycle if current == start and steps < g (and steps > 0)
        
        nb_visited = set()
        q = deque()
        for nbr in adj[start]:
            q.append((nbr, start, 1))
            nb_visited.add((nbr, start))
        
        while q:
            curr, prev, dist = q.popleft()
            if curr == start and 0 < dist < g:
                return False
            if dist >= g - 1:
                continue
            for nbr in adj[curr]:
                if nbr == prev:
                    continue
                if (nbr, curr) not in nb_visited:
                    nb_visited.add((nbr, curr))
                    q.append((nbr, curr, dist + 1))
    
    return True


def _fallback_cage(k, g):
    """Fallback: use the Lazebnik-Ustimenko-Woldar construction."""
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
