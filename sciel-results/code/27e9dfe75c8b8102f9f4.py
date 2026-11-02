import time
import collections

_DEADLINE = None

def _time_left():
    if _DEADLINE is None:
        return 1e9
    return _DEADLINE - time.monotonic()

def _has_short_cycle_nb(n, edges, min_girth):
    """
    Returns True if the graph has a cycle shorter than min_girth.
    Uses non-backtracking walks from every directed edge.
    A cycle of length L exists iff there is a non-backtracking closed walk of length L.
    For girth >= min_girth, no non-backtracking closed walk of length < min_girth exists.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    # For each directed edge (u, v), do BFS on non-backtracking walks.
    # State: (prev, curr). Start: (u, v). Next: (curr, w) where w != prev.
    # A cycle is found if we reach (v, u) — meaning we came back to the start edge reversed.
    # Actually, a closed non-backtracking walk of length L starting from (u,v) means
    # after L steps we're at (v, u). The cycle length is L.
    # We need to check if any such closed walk has length < min_girth.

    max_len = min_girth - 1

    for u, v in edges:
        # BFS from (u, v)
        # dist[(prev, curr)] = length of shortest non-backtracking walk from (u,v) to (prev,curr)
        dist = {}
        dist[(u, v)] = 1
        queue = collections.deque([(u, v)])

        while queue:
            prev, curr = queue.popleft()
            d = dist[(prev, curr)]
            if d >= max_len:
                continue
            for nxt in adj[curr]:
                if nxt == prev:
                    continue
                state = (curr, nxt)
                if state in dist:
                    continue
                dist[state] = d + 1
                # Check if we've closed a cycle: state (curr, nxt) where nxt == u and curr == v
                # That means we're back at the start directed edge reversed.
                # The walk length is d+1. If d+1 < min_girth, we have a short cycle.
                if nxt == u and curr == v:
                    if d + 1 < min_girth:
                        return True
                    break
                queue.append(state)

    return False

def construct(k, g):
    global _DEADLINE
    _DEADLINE = time.monotonic() + 115.0

    # Moore bound for k=3, g=16: 2 + k * sum_{i=0}^{g/2-2} (k-1)^i = 2 + 3*(1+2+4+8+16+32) = 2+3*63 = 191?
    # Actually for even girth g=2m, Moore bound is 2 + k * sum_{i=0}^{m-2} (k-1)^i
    # g=16, m=8: 2 + 3*(1+2+4+8+16+32+64) = 2 + 3*127 = 2 + 381 = 383
    # Wait, let me recalculate. For girth g (even), the Moore bound is:
    # n >= 2 + k * sum_{i=0}^{g/2 - 2} (k-1)^i
    # For k=3, g=16: g/2-2 = 6, sum = 1+2+4+8+16+32 = 63, n >= 2 + 3*63 = 191
    # Hmm, but the task says lower bound is 512. Let me use a safe lower bound.
    
    # The task says best lower bound is 512. Let's start from a reasonable range.
    # Generalized Petersen Graphs have 2n vertices.
    
    min_n = 256  # 2*256 = 512 vertices
    max_n = 700  # 2*700 = 1400 vertices

    for n in range(min_n, max_n + 1):
        if _time_left() < 20.0:
            break

        for k_gp in range(1, n // 2 + 1):
            if _time_left() < 8.0:
                break

            # Skip k_gp that create multi-edges or loops
            if k_gp == n - k_gp:
                continue
            if k_gp == 0 or k_gp == n:
                continue

            # Quick modular check for girth
            # For GP(n, k_gp), short cycles come from:
            # 1. Outer cycle: always length n (>= min_n >= 256 > 16, fine)
            # 2. Inner cycle: always length n (fine)
            # 3. Mixed cycles involving spokes
            # The condition for no short cycles: k_gp^j not in {1, -1} mod n for j=1..g/2-1
            # More precisely, for girth g, we need k_gp^j not congruent to 1 or -1 mod n
            # for j = 1, 2, ..., g/2 - 1, and also k_gp^j not congruent to 0 mod n.

            valid = True
            val = k_gp % n
            for j in range(1, g // 2):
                if val == 0:
                    valid = False
                    break
                val = (val * k_gp) % n
                if val == 1 or val == n - 1:
                    valid = False
                    break

            if not valid:
                continue

            edges = []
            # Outer cycle
            for i in range(n):
                edges.append((i, (i + 1) % n))
            # Spokes
            for i in range(n):
                edges.append((i, n + i))
            # Inner cycle
            for i in range(n):
                edges.append((n + i, n + (i + k_gp) % n))

            total_vertices = 2 * n

            if not _has_short_cycle_nb(total_vertices, edges, g):
                return total_vertices, edges

    # Fallback to LUW
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
        if _time_left() < 5.0:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
