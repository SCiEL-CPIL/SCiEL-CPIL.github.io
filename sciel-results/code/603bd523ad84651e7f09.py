"""Construct small 4-regular graphs of high girth (cage problem n(4,10))."""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START

def _build_adj(n, edges):
    """Build adjacency list from edge list, verifying simplicity."""
    adj = [[] for _ in range(n)]
    seen = set()
    for a, b in edges:
        if a == b or a < 0 or a >= n or b < 0 or b >= n:
            return None
        key = (min(a, b), max(a, b))
        if key in seen:
            return None
        seen.add(key)
        adj[a].append(b)
        adj[b].append(a)
    return adj


def _is_valid(k, g, n, edges):
    """Check degree, simplicity, and girth via non-backtracking BFS from every vertex."""
    if n <= 0:
        return False
    adj = _build_adj(n, edges)
    if adj is None:
        return False
    for v in range(n):
        if len(adj[v]) != k:
            return False
    # Girth check: for each vertex s, BFS in non-backtracking state space
    for s in range(n):
        visited = set()
        q = deque()
        for v in adj[s]:
            visited.add((v, s))
            q.append((v, s, 1))
        while q:
            cur, prev, dist = q.popleft()
            if dist >= g - 1:
                continue
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                new_state = (nxt, cur)
                if new_state in visited:
                    continue
                if nxt == s:
                    if dist + 1 < g:
                        return False
                visited.add(new_state)
                q.append((nxt, cur, dist + 1))
    return True


def _luw_graph(k, g):
    """Lazebnik-Ustimenko-Woldar construction (baseline)."""
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


def _random_k_regular(n, k, seed=0):
    """Generate a random k-regular graph on n vertices using configuration model + dedup."""
    if n * k % 2 != 0:
        return None
    rng = random.Random(seed)
    stubs = []
    for v in range(n):
        for _ in range(k):
            stubs.append(v)
    max_attempts = 200
    for _ in range(max_attempts):
        rng.shuffle(stubs)
        edges = set()
        valid = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v:
                valid = False
                break
            key = (min(u, v), max(u, v))
            if key in edges:
                valid = False
                break
            edges.add(key)
        if valid and len(edges) == n * k // 2:
            return list(edges)
    return None


def _verify_girth_only(k, g, n, edges):
    """Faster girth-only check assuming degree and simplicity already verified."""
    adj = _build_adj(n, edges)
    if adj is None:
        return False
    for v in range(n):
        if len(adj[v]) != k:
            return False
    for s in range(n):
        visited = set()
        q = deque()
        for v in adj[s]:
            visited.add((v, s))
            q.append((v, s, 1))
        while q:
            cur, prev, dist = q.popleft()
            if dist >= g - 1:
                continue
            for nxt in adj[cur]:
                if nxt == prev:
                    continue
                new_state = (nxt, cur)
                if new_state in visited:
                    continue
                if nxt == s:
                    if dist + 1 < g:
                        return False
                visited.add(new_state)
                q.append((nxt, cur, dist + 1))
    return True


def _construct_moore_candidate(k, g, seed=0):
    """
    Try to construct a graph near the Moore bound using a structured approach.
    For even girth g=2r, the Moore bound is 2*(k-1)^(r-1).
    We try to build a graph with exactly that many vertices using a layered structure.
    """
    r = g // 2
    moore_n = 2 * (k - 1) ** (r - 1)
    # For k=4, g=10: r=5, moore_n = 2*3^4 = 162
    # Try random regular graphs of size moore_n
    rng = random.Random(seed)
    for attempt in range(50):
        edges = _random_k_regular(moore_n, k, seed=seed + attempt)
        if edges is None:
            continue
        if _verify_girth_only(k, g, moore_n, edges):
            return moore_n, edges
    return None


def _try_improve(n, k, g, edges, deadline, seed=0):
    """Try to reduce n by finding a smaller valid graph."""
    best_n, best_edges = n, edges
    moore = 2 * (k - 1) ** (g // 2 - 1) if g % 2 == 0 else k * (k - 1) ** ((g - 2) // 2)
    
    # First, try the Moore bound size
    if moore < n:
        result = _construct_moore_candidate(k, g, seed=seed)
        if result is not None:
            return result
        if time.time() > deadline:
            return best_n, best_edges
    
    # Try sizes between Moore bound and current n
    for n_try in range(moore + 1, min(best_n, moore + 50)):
        if time.time() > deadline:
            break
        for attempt in range(30):
            if time.time() > deadline:
                break
            cand_edges = _random_k_regular(n_try, k, seed=seed + n_try * 100 + attempt)
            if cand_edges is None:
                continue
            if _verify_girth_only(k, g, n_try, cand_edges):
                return n_try, cand_edges
    return best_n, best_edges


def construct(k, g):
    deadline = time.time() + 110  # Safety margin before 150s limit
    # Start with LUW construction
    n, edges = _luw_graph(k, g)
    # Verify it's valid
    if not _is_valid(k, g, n, edges):
        # Fallback: try random
        for attempt in range(50):
            if time.time() > deadline:
                break
            cand = _random_k_regular(n, k, seed=attempt)
            if cand and _is_valid(k, g, n, cand):
                edges = cand
                break
    
    # Try to improve
    n, edges = _try_improve(n, k, g, edges, deadline, seed=42)
    return n, edges

# EVOLVE-BLOCK-END
