"""
Construct a k-regular graph with girth at least g.
Strategy:
  1. Random covering-code (random-lift) construction -- a structurally
     different design family (lifts of a small base graph).  For k=3 we lift
     the Petersen graph; the girth grows ~log(N) and we can land well below
     the LUW size.  Girth is certified exactly with non-backtracking walks.
  2. Circulant search C_n(1,a,b) (fast, vertex-transitive check from 0).
  3. Lazebnik-Ustimenko-Woldar CD(d,q) as a guaranteed fallback.
"""
import time
import random
from collections import deque

# EVOLVE-BLOCK-START


def construct(k, g):
    """
    Construct a k-regular graph with girth at least g.
    Returns (n, edges) where n is the number of vertices and edges is a list of (i, j) pairs.
    """
    start_time = time.time()
    deadline = start_time + 112  # safety margin vs 150s wall-clock limit

    def check_deadline():
        return time.time() > deadline

    # ---- Primary: random covering-code / lift construction (k=3, g=16) ----
    if k == 3 and g == 16:
        try:
            res = _construct_covering_code(3, 16, deadline)
            if res is not None:
                n, edges = res
                # Double-check (cheap for the sizes we reach)
                if _verify_girth(n, edges, 16):
                    return n, edges
        except Exception:
            pass

    # ---- Secondary: circulant search C_n(1,a,b) ----
    if k == 3 and g == 16:
        try:
            res = _find_circulant_cage(3, 16, deadline)
            if res is not None:
                n, edges = res
                return n, edges
        except Exception:
            pass

    # ---- Fallback: Lazebnik-Ustimenko-Woldar ----
    return _luw_construct(k, g)


# ---------------------------------------------------------------------------
# Random covering-code (lift) construction
# ---------------------------------------------------------------------------

# Petersen graph edges (0..9), 3-regular, girth 5
_PETERSEN_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
]


def _lift_graph(base_edges, m):
    """
    Randomly lift base_edges (on 0..b-1) to a 3-regular graph on b*m vertices.
    Each base vertex u becomes a fibre {u*m .. (u+1)*m - 1}.  For each base edge
    (u,v), pair the two fibres by a random bijection (random permutation).
    Returns (n, edges).
    """
    b = max(u for u, v in base_edges) + 1
    n = b * m
    edges = []
    for (u, v) in base_edges:
        perm = list(range(m))
        random.shuffle(perm)
        for i in range(m):
            a = u * m + i
            c = v * m + perm[i]
            if a < c:
                edges.append((a, c))
            else:
                edges.append((c, a))
    return n, edges


def _verify_girth(n, edges, g):
    """
    Exact girth certification via non-backtracking walks from every vertex.
    Returns True iff no cycle shorter than g exists.
    """
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if len(adj[i]) != 3:
            return False

    for src in range(n):
        if _short_cycle_from(src, adj, g):
            return False
    return True


def _short_cycle_from(src, adj, g):
    """
    Non-backtracking BFS from src.  Returns True iff a cycle of length < g is
    found.  A cycle of length L>=3 through src exists iff there is a
    non-backtracking walk of length L from src back to src.
    """
    max_len = g - 1
    dist = {}
    queue = deque()
    start = (src, -1)
    dist[start] = 0
    queue.append(start)

    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]
        if d >= max_len:
            continue
        for v in adj[u]:
            if v == prev:
                continue
            nd = d + 1
            if nd > max_len:
                continue
            state = (v, u)
            if state not in dist or nd < dist[state]:
                dist[state] = nd
                if v == src and u != -1:
                    return True  # closed a non-backtracking walk -> cycle
                queue.append(state)
    return False


def _construct_covering_code(k, g, deadline):
    """
    Random-lift (covering-code) search.  Tries increasing fibre size m; for each
    m, repeatedly builds a random lift of the Petersen graph and verifies the
    exact girth.  Returns (n, edges) on the first valid graph, else None.
    """
    base = _PETERSEN_EDGES
    rng = random.Random(12345)
    # Use module-level shuffle but seed our own for determinism
    best = None

    # m is the number of vertices per base vertex; total n = 10*m
    # girth grows ~log_m; target n ~ 2000-8000 is realistic for g=16.
    m_values = [80, 120, 160, 200, 240, 280, 320, 360, 400, 480, 560, 640, 800, 1000]

    for m in m_values:
        if time.time() > deadline - 4:
            break
        n = 10 * m
        # Budget per m: more attempts for smaller m
        if m <= 200:
            max_attempts = 250
        elif m <= 400:
            max_attempts = 160
        else:
            max_attempts = 80

        for attempt in range(max_attempts):
            if time.time() > deadline - 4:
                break
            # deterministic-ish but varied random lift
            random.seed(100000 + m * 1000 + attempt)
            _, edges = _lift_graph(base, m)
            if _verify_girth(n, edges, g):
                return n, edges
            if best is None:
                best = (n, edges)

    return best


# ---------------------------------------------------------------------------
# Circulant search C_n(1, a, b)
# ---------------------------------------------------------------------------

def _find_circulant_cage(k, g, deadline):
    """
    Search for a circulant graph C_n(1, a, b) with girth at least g.
    """
    min_n = 512
    max_n = 900

    for n in range(min_n, max_n + 1, 2):
        if time.time() > deadline - 5:
            return None

        candidates = _generate_candidates(n)
        for a, b in candidates:
            if time.time() > deadline - 5:
                return None
            if _has_short_cycle_fast(n, a, b, g):
                continue
            edges = []
            for i in range(n):
                for shift in [1, a, b]:
                    j = (i + shift) % n
                    if i < j:
                        edges.append((i, j))
            return n, edges
    return None


def _generate_candidates(n):
    """Generate candidate (a, b) pairs for C_n(1, a, b), most promising first."""
    candidates = []
    half = n // 2
    for a in range(2, half):
        for b in range(a + 1, half):
            if _quick_reject(n, a, b):
                continue
            candidates.append((a, b))
    return candidates


def _quick_reject(n, a, b):
    """Reject pairs that obviously create a 4-cycle."""
    G = [1, -1, a, -a, b, -b]
    G_mod = [x % n for x in G]
    for i in range(6):
        for j in range(6):
            if i == j:
                continue
            if G_mod[i] == (-G_mod[j]) % n:
                continue
            s = (G_mod[i] + G_mod[j]) % n
            for kk in range(6):
                if kk == i or kk == j:
                    continue
                if s == G_mod[kk]:
                    return True
    return False


def _has_short_cycle_fast(n, a, b, g):
    """
    Fast girth check for C_n(1, a, b) using non-backtracking walks from vertex 0.
    Returns True if a cycle shorter than g is found.
    """
    def get_neighbors(u):
        return [(u + 1) % n, (u - 1) % n, (u + a) % n, (u - a) % n, (u + b) % n, (u - b) % n]

    dist = {}
    queue = deque()
    start_state = (0, -1)
    dist[start_state] = 0
    queue.append(start_state)
    max_len = g - 1

    while queue:
        u, prev = queue.popleft()
        d = dist[(u, prev)]
        if d >= max_len:
            continue
        for v in get_neighbors(u):
            if v == prev:
                continue
            new_state = (v, u)
            new_d = d + 1
            if new_d > max_len:
                continue
            if new_state not in dist or new_d < dist[new_state]:
                dist[new_state] = new_d
                if v == 0 and u != -1:
                    return True
                queue.append(new_state)
    return False


# ---------------------------------------------------------------------------
# Lazebnik-Ustimenko-Woldar fallback
# ---------------------------------------------------------------------------

def _luw_construct(k, g):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with q = k."""
    q = k
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1
    return _luw_construct_with_d(k, g, d)


def _luw_construct_with_d(k, g, d):
    """Lazebnik-Ustimenko-Woldar graph CD(d, q) with specified d."""
    q = k

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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
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


# EVOLVE-BLOCK-END
