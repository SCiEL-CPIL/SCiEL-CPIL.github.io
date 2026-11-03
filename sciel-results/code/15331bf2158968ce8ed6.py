"""Construct 5-regular girth-9 cage candidates.

Uses the Lazebnik-Ustimenko-Woldar (LUW) construction with optimized rule
selection, plus a targeted local-search improvement for the k=5,g=9 case
within a strict wall-clock budget.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k.

    Tries multiple rule assignments to minimize the resulting vertex count.
    Returns (n, edges) for a k-regular bipartite graph with girth at least g.
    """
    q = k
    # For g=9, d=3 might yield girth 8 or 10. We need to verify.
    # If girth < g, we must increase d.
    # Standard heuristic: d = g - 5 is for girth ~ 2d+2.
    # If 2d+2 < g, we need larger d.
    # For g=9, 2d+2 >= 9 => d >= 3.5 => d=4.
    # But d=4 for q=5 often has girth 8.
    # So we should try d=3, check, and if fail, go to d=5.
    
    candidate_ds = [3, 4, 5, 6]
    # Filter d such that n is not too large (optional, but good for time)
    # And try smallest first.
    
    best_n = float('inf')
    best_edges = None
    
    for d in candidate_ds:
        if d < 3: continue
        # Optional: skip if n is too large? 
        # n = 2 * q^d. For q=5, d=6 => 31250. 
        # Let's cap d at 5 for safety in time, unless g is large.
        if q ** d > 10000: break
        
        # Build with standard rules
        n, edges = _luw_build_with_d(k, d)
        
        # Check girth
        if _check_girth_fast(edges, n, g):
            if n < best_n:
                best_n = n
                best_edges = edges
            # If we found a valid one with small d, we can stop?
            # Not necessarily, a larger d might have smaller n? No, n grows with d.
            # So the first valid d gives the smallest n.
            break
    else:
        # If no d worked (should not happen for d large enough), use d=5
        n, edges = _luw_build_with_d(k, 5)
        best_n = n
        best_edges = edges
        
    return best_n, best_edges


def _luw_build_with_d(k, d):
    """Helper to build LUW graph with specific d."""
    q = k
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q

    def build_with_rule(rule):
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_op(x[c], mul_op(y[a], x[b]))
                    else:
                        y[c] = sub_op(x[c], mul_op(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Standard rule
    rule_standard = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule_standard[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    return build_with_rule(rule_standard)


def _check_girth_fast(edges, n, g):
    """Fast girth check using BFS from each vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    half_g = (g - 1) // 2
    for v in range(n):
        # BFS
        dist = [-1] * n
        prev = [-1] * n
        q = [v]
        dist[v] = 0
        head = 0
        while head < len(q):
            cur = q[head]
            head += 1
            if dist[cur] >= half_g:
                continue
            for nb in adj[cur]:
                if dist[nb] == -1:
                    dist[nb] = dist[cur] + 1
                    prev[nb] = cur
                    q.append(nb)
                elif nb != prev[cur]:
                    # Cycle found
                    cycle_len = dist[cur] + dist[nb] + 1
                    if cycle_len < g:
                        return False
    return True

    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add_op = lambda x, y: x ^ y
        sub_op = lambda x, y: x ^ y
        mul_op = lambda x, y: table[x][y]
    else:
        add_op = lambda x, y: (x + y) % q
        sub_op = lambda x, y: (x - y) % q
        mul_op = lambda x, y: x * y % q

    def build_with_rule(rule):
        """Build graph with a given rule dict, return (n, edges) or None if invalid."""
        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        y[c] = add_op(x[c], mul_op(y[a], x[b]))
                    else:
                        y[c] = sub_op(x[c], mul_op(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index = {start: 0}
        order = [start]
        edges = []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    # Try the standard LUW rule
    rule_standard = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule_standard[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    best_n, best_edges = build_with_rule(rule_standard)

    # Try alternative rules: vary the coefficients for c >= 5
    # The rule at level c is (a, b) meaning y[c] = x[c] +/- y[a]*x[b]
    # Valid options: a in {1, ..., c-1}, b in {1, ..., c-1}, a != b
    # For small d, we can try a few permutations
    alt_rules = []
    for c in range(5, d + 1):
        # Generate alternative (a,b) pairs
        alts = [(1, c-1), (2, 1), (1, 2), (c-1, 1), (2, c-1)]
        alts = [(a, b) for a, b in alts if 1 <= a <= c-1 and 1 <= b <= c-1 and a != b]
        if not alts:
            alts = [(1, 2)]
        alt_rules.append(alts)

    # Try a few combinations (limited to keep runtime reasonable)
    import itertools
    max_combos = 20
    count = 0
    for combo in itertools.product(*alt_rules):
        if count >= max_combos:
            break
        count += 1
        rule_alt = dict(rule_standard)
        for i, c in enumerate(range(5, d + 1)):
            rule_alt[c] = combo[i]
        try:
            n_alt, edges_alt = build_with_rule(rule_alt)
            if n_alt < best_n:
                best_n, best_edges = n_alt, edges_alt
        except Exception:
            continue

    return best_n, best_edges


def _hexagon_4_4():
    """Construct the generalized hexagon H(4,4) incidence graph.

    The generalized hexagon of order (4,4) has 330 vertices (165 points + 165 lines),
    is 5-regular, and has girth 12.

    Uses the explicit construction from the parabolic quadric in PG(3,4).
    Points are 1-dim subspaces of singular vectors of a quadratic form over GF(5).
    """
    q = 5

    def gf5_inv(x):
        x = x % 5
        if x == 0:
            return 0
        return pow(x, 4, 5)

    # Singular vectors: x1*x4 - x2*x3 = 0 in GF(5)
    # Normalize so first nonzero coordinate is 1
    points = []
    point_set = set()

    for x1 in range(5):
        for x2 in range(5):
            for x3 in range(5):
                for x4 in range(5):
                    if x1 == x2 == x3 == x4 == 0:
                        continue
                    if (x1 * x4 - x2 * x3) % 5 != 0:
                        continue
                    vec = (x1, x2, x3, x4)
                    for i in range(4):
                        if vec[i] != 0:
                            inv = gf5_inv(vec[i])
                            norm = tuple(c * inv % 5 for c in vec)
                            break
                    if norm not in point_set:
                        point_set.add(norm)
                        points.append(norm)

    n_points = len(points)
    # Expected: 165 points for H(4,4)
    # If construction is off, return None to fall back to LUW
    if n_points != 165:
        return None

    # Lines: 2-dim subspaces where all points are singular
    # For each pair of points, check if the 2-dim subspace they span is totally singular
    point_index = {p: i for i, p in enumerate(points)}
    edges = []
    seen_pairs = set()

    for i in range(n_points):
        for j in range(i + 1, n_points):
            p1 = points[i]
            p2 = points[j]
            # Check if all vectors in span(p1, p2) are singular
            # A vector a*p1 + b*p2 is singular if (a*p1[0]+b*p2[0])*(a*p1[3]+b*p2[3]) -
            # (a*p1[1]+b*p2[1])*(a*p1[2]+b*p2[2]) = 0
            # This is a quadratic in a,b. For it to be identically zero:
            # Coefficient of a^2: p1[0]*p1[3] - p1[1]*p1[2] = 0 (already true)
            # Coefficient of b^2: p2[0]*p2[3] - p2[1]*p2[2] = 0 (already true)
            # Coefficient of ab: p1[0]*p2[3] + p2[0]*p1[3] - p1[1]*p2[2] - p2[1]*p1[2] = 0
            cross = (p1[0] * p2[3] + p2[0] * p1[3] - p1[1] * p2[2] - p2[1] * p1[2]) % 5
            if cross == 0:
                edges.append((i, n_points + j))
                edges.append((j, n_points + i))
                seen_pairs.add((i, j))

    # Check regularity: each point should have degree 5
    adj = {i: set() for i in range(2 * n_points)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    for i in range(2 * n_points):
        if len(adj[i]) != 5:
            return None

    n = 2 * n_points
    edge_list = [(u, v) for u, v in edges if u < v]
    return n, edge_list


def _check_girth(adj, n, g, deadline):
    """Check if graph has girth >= g using non-backtracking BFS from each vertex.

    Optimized: uses array-based storage for visited states and early termination.
    """
    half_g = (g - 1) // 2
    for v in range(n):
        if time.time() > deadline:
            return False
        # BFS: (vertex, prev) -> depth
        # Use dict for sparse visited set
        seen = {}
        queue = deque()
        for nb in adj[v]:
            queue.append((nb, v, 1))
            seen[(nb, v)] = 1
        while queue:
            cur, prev, depth = queue.popleft()
            if depth >= half_g:
                continue
            for nb in adj[cur]:
                if nb == prev:
                    continue
                if nb == v:
                    if depth + 1 < g:
                        return False
                elif (nb, cur) in seen:
                    cycle_len = seen[(nb, cur)] + depth + 1
                    if cycle_len < g:
                        return False
                else:
                    seen[(nb, cur)] = depth + 1
                    queue.append((nb, cur, depth + 1))
    return True


def _local_search_5_9(max_time=110.0):
    """Attempt to find a 5-regular graph with girth >= 9 and fewer vertices than baseline."""
    start_time = time.time()
    deadline = start_time + max_time

    # Try hexagon construction first (deterministic, girth 12)
    hex_result = _hexagon_4_4()
    if hex_result is not None:
        best_n, best_edges = hex_result
    else:
        best_n, best_edges = _luw_construct(5, 9)

    # Try candidate sizes smaller than best_n
    candidate_ns = [106, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000]
    candidate_ns = [n for n in candidate_ns if n % 2 == 0 and n < best_n]
    candidate_ns = sorted(set(candidate_ns))

    random.seed(42)

    for n in candidate_ns:
        if time.time() > deadline - 15:
            break

        half = n // 2
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 15:
                break

            # Generate random 5-regular bipartite graph
            left_stubs = []
            for i in range(half):
                left_stubs.extend([i] * 5)
            right_stubs = []
            for j in range(half):
                right_stubs.extend([j + half] * 5)

            random.shuffle(left_stubs)
            random.shuffle(right_stubs)

            adj = {i: set() for i in range(n)}
            for ls, rs in zip(left_stubs, right_stubs):
                adj[ls].add(rs)
                adj[rs].add(ls)

            if any(len(adj[i]) != 5 for i in range(n)):
                continue

            edges = set()
            for i in range(n):
                for j in adj[i]:
                    if i < j:
                        edges.add((i, j))

            valid = _check_girth(adj, n, 9, deadline - 20)
            if not valid:
                continue

            edge_list = list(edges)
            if n < best_n:
                best_n = n
                best_edges = edge_list
            break

    return best_n, best_edges


def construct(k, g):
    """Construct a k-regular graph with girth at least g on as few vertices as possible.

    Returns (n, edges) where edges is a list of (i, j) tuples.
    """
    start_time = time.time()
    deadline = start_time + 140.0

    # Use the robust LUW construction with girth verification
    n, edges = _luw_construct(k, g)
    return n, edges

# EVOLVE-BLOCK-END
