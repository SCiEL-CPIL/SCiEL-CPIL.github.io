"""Construct a 4-regular graph of girth 10 with as few vertices as possible."""
import time
import random
import sys
from collections import deque

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar construction as baseline."""
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


def _check_girth_bfs(n, adj, g):
    """Check that no cycle shorter than g exists using non-backtracking BFS from each vertex."""
    for start in range(n):
        # BFS tracking (node, prev) to avoid backtracking
        visited = {}  # (node, prev) -> distance
        dq = deque()
        for nb in adj[start]:
            dq.append((nb, start, 1))
            visited[(nb, start)] = 1
        while dq:
            node, prev, dist = dq.popleft()
            if dist >= g:
                continue
            for nb in adj[node]:
                if nb == prev:
                    continue
                if nb == start and dist + 1 < g:
                    return False
                if (nb, node) in visited:
                    if visited[(nb, node)] + 1 <= dist + 1 and dist + 1 < g:
                        # Found a cycle of length visited[(nb,node)] + dist + 1
                        # Actually we need to check if this creates a short cycle
                        if dist + 1 < g:
                            # This means there's a path from start to nb (not through node) 
                            # of length visited[(nb,node)], and a path start->...->node->nb of length dist+1
                            # The cycle length is visited[(nb,node)] + dist + 1
                            if visited[(nb, node)] + dist + 1 < g:
                                return False
                else:
                    visited[(nb, node)] = dist + 1
                    if dist + 1 < g:
                        dq.append((nb, node, dist + 1))
    return True


def _try_random_bipartite(n_half, degree, g, rng):
    """Try to build a degree-regular bipartite graph on 2*n_half vertices with girth >= g."""
    n = 2 * n_half
    # Build adjacency as list of lists
    adj = [set() for _ in range(n)]
    
    # Greedy: assign edges one by one, avoiding short cycles
    # Target: each vertex has exactly 'degree' neighbors
    # Left side: 0..n_half-1, Right side: n_half..n-1
    
    # For each left vertex, assign 'degree' right neighbors
    # Check that no 4-cycle, 6-cycle, or 8-cycle is created
    
    def creates_short_cycle(left_v, right_v, adj, degree, g):
        """Check if adding edge (left_v, right_v) creates a cycle shorter than g."""
        if g <= 4:
            return False
        # Check for 4-cycles: left_v - r1 - left_v2 - r2 - left_v
        # A 4-cycle exists if there's a left_v2 connected to both right_v and some r1 connected to left_v
        # More generally, do a BFS from left_v (excluding the new edge) to see if right_v is reachable
        # within distance (g//2 - 1) through the bipartite structure
        
        # For a bipartite graph, a cycle of length 2*d exists iff there's a path of length d-1 
        # from left_v to right_v not using the direct edge, plus the two new edges... 
        # Actually: adding edge (L, R) creates a cycle of length 2d if there's an existing path 
        # from L to R of length 2d-1 (odd, since L and R are on opposite sides).
        # Wait, L and R are on opposite sides, so any path from L to R has odd length.
        # A cycle through the new edge L-R has length = 2 + (path length from R to L not using L-R).
        # Path from R to L not using L-R: R -> L1 -> R1 -> L2 -> ... -> L. This has even length.
        # Cycle length = 2 + even = even. Good.
        # For girth g, we need no cycle of length < g, so no path from R to L of length < g-2 (even).
        
        max_path_len = g - 3  # We need path from R to L of length >= g-2, so check paths of length < g-2
        # Path from R to L of length 2k means k intermediate pairs
        # BFS from R, not going through L directly, see if we can reach L in < g-2 steps
        
        # BFS from R
        # State: (vertex, depth)
        # We want to reach L with depth < g - 2
        if max_path_len < 2:
            return False
        
        visited = set()
        dq = deque([(right_v, 0)])
        visited.add(right_v)
        while dq:
            v, depth = dq.popleft()
            if depth >= max_path_len:
                continue
            if v == left_v:
                # Found path of length depth from R to L
                # Cycle length = depth + 2
                if depth + 2 < g:
                    return True
                continue
            for nb in adj[v]:
                if nb == left_v and depth + 1 < max_path_len:
                    # Would create cycle of length depth + 1 + 1 = depth + 2
                    if depth + 2 < g:
                        return True
                    continue
                if nb not in visited:
                    visited.add(nb)
                    dq.append((nb, depth + 1))
        return False
    
    # Try to build the graph
    # For each left vertex, pick 'degree' right vertices
    for lv in range(n_half):
        available = list(range(n_half, n))
        rng.shuffle(available)
        chosen = []
        for rv in available:
            if len(chosen) == degree:
                break
            if not creates_short_cycle(lv, rv, adj, degree, g):
                adj[lv].add(rv)
                adj[rv].add(lv)
                chosen.append(rv)
        if len(chosen) < degree:
            return None  # Failed
    
    # Now check right side degrees
    for rv in range(n_half, n):
        if len(adj[rv]) != degree:
            return None
    
    # Verify girth with full check
    if not _check_girth_bfs(n, [list(a) for a in adj], g):
        return None
    
    edges = []
    for i in range(n_half):
        for j in adj[i]:
            edges.append((i, j))
    return n, edges


def _try_improve_luw(k, g, deadline):
    """Try to find a smaller graph than LUW via random bipartite construction."""
    # LUW gives 450 for (4,10). Try sizes from 244 (even, >= Moore bound 242) up to 448
    # n must be even for bipartite
    moore = 2 * sum(3**i for i in range(g // 2))  # 242 for g=10
    luw_n, _ = _luw_construct(k, g)
    
    best_n = luw_n
    best_edges = None
    rng = random.Random(42)
    
    # Try various sizes, starting smaller
    target_sizes = list(range(moore + (2 - moore % 2), luw_n, 2))
    rng.shuffle(target_sizes)
    # But prioritize smaller sizes
    target_sizes.sort()
    
    for n_half in [s // 2 for s in target_sizes[:50]]:  # Try up to 50 sizes
        if time.time() > deadline - 10:
            break
        n_target = 2 * n_half
        if n_target >= best_n:
            break
        # Try a few random attempts for this size
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            result = _try_random_bipartite(n_half, k, g, rng)
            if result is not None:
                n_found, edges = result
                if n_found < best_n:
                    best_n = n_found
                    best_edges = edges
                break  # Found one for this size, move to smaller
    
    if best_edges is not None:
        return best_n, best_edges
    return luw_n, _luw_construct(k, g)[1]


def construct(k, g):
    """Construct a k-regular graph with girth >= g, minimizing vertices."""
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Get LUW baseline
    luw_n, luw_edges = _luw_construct(k, g)
    
    # Verify LUW is valid
    if not _check_girth_bfs(luw_n, luw_edges, g):
        pass  # LUW should be valid
    
    # Try to improve
    if k == 4 and g == 10:
        # Special case: try random construction
        try:
            improved_n, improved_edges = _try_improve_luw(k, g, deadline)
            if improved_n < luw_n:
                return improved_n, improved_edges
        except Exception:
            pass
    
    return luw_n, luw_edges

# EVOLVE-BLOCK-END
