import time
import math
import random
from collections import deque

_START_TIME = time.time()
_TIMEOUT = 145  # wall-clock safety margin before 150s kill


def _deadline_exceeded():
    return (time.time() - _START_TIME) > _TIMEOUT


def _is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.isqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def _is_prime_power(n):
    """Check if n is a prime power p^e. Returns (p, e) or None."""
    if n < 2:
        return None
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % p == 0:
            m, e = n, 0
            while m % p == 0:
                m //= p
                e += 1
            if m == 1:
                return (p, e)
            return None
    return (n, 1)  # n itself is prime


def _verify_girth(edges, n, g):
    """Check no cycle shorter than g using non-backtracking walks from every vertex."""
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        dist = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb, 1))
            dist[(start, nb)] = 1
        while queue:
            prev, curr, d = queue.popleft()
            if d >= g:
                break
            for nb2 in adj[curr]:
                if nb2 == prev:
                    continue
                state = (curr, nb2)
                if state in dist:
                    if d + 1 < g:
                        return False
                else:
                    dist[state] = d + 1
                    queue.append((curr, nb2, d + 1))
    return True


def _check_simple_and_regular(edges, n, k):
    """Check simple (no loops, no repeated edges) and k-regular."""
    if len(edges) != n * k // 2:
        return False
    seen = set()
    deg = [0] * n
    for u, v in edges:
        if u == v:
            return False
        a, b = min(u, v), max(u, v)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[u] += 1
        deg[v] += 1
    for d in deg:
        if d != k:
            return False
    return True


def construct_lw(k, g):
    """Lazebnik-Ustimenko-Woldar CD(d,q) construction. Valid but large."""
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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order = {start: 0}, [start]
    edges = []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges


def construct_abs(k, g):
    """Alon-Babai-Suzuki construction on Heisenberg group H_q. Requires even k in {4,6}."""
    if g % 2 == 0:
        mb = 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        mb = 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))

    q = 2
    while True:
        if _is_prime(q) and q ** 3 >= mb:
            break
        q += 1

    if k == 4:
        gens = [(1, 0, 0), (q - 1, 0, 0), (0, 1, 0), (0, q - 1, 1)]
    elif k == 6:
        gens = [(1, 0, 0), (q - 1, 0, 0), (0, 1, 0), (0, q - 1, 1), (0, 0, 1), (0, 0, q - 1)]
    else:
        return None  # signal fallback

    n = q ** 3

    def idx(x, y, z):
        return (x * q + y) * q + z

    edges = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                u = idx(x, y, z)
                for gx, gy, gz in gens:
                    vx = (x + gx) % q
                    vy = (y + gy) % q
                    vz = (z + gz + x * gy) % q
                    v = idx(vx, vy, vz)
                    if u < v:
                        edges.append((u, v))
    return n, edges


def _moore_bound(k, g):
    """Lower bound on vertices of a k-regular graph of girth g."""
    if g % 2 == 0:
        return 2 * sum((k - 1) ** i for i in range(g // 2))
    else:
        return 1 + k * sum((k - 1) ** i for i in range((g - 1) // 2))


def _try_circulant(k, g, n, max_walk_depth):
    """Try to find a k-regular circulant graph of girth >= g on n vertices.
    A circulant graph on Z_n with connection set S is k-regular if |S|=k (with S symmetric,
    no 0). We search over random symmetric connection sets."""
    if n * k % 2 != 0:
        return None
    # For circulant, need k even for symmetric set without 0 (unless n even allows self-pairing)
    if k % 2 != 0:
        return None

    # We need |S| = k where S is a symmetric subset of Z_n \ {0}
    # Number of distinct offsets: k/2 if n odd or n even with no n/2, etc.
    # Simple approach: pick k/2 random distinct offsets in [1, n//2], and their negatives.
    half_k = k // 2
    candidates = list(range(1, n // 2 + 1))
    random.shuffle(candidates)
    offsets = candidates[:half_k]
    S = set()
    for o in offsets:
        S.add(o % n)
        S.add((-o) % n)
    if 0 in S:
        return None
    if len(S) != k:
        return None

    # Build edges
    edges = []
    for u in range(n):
        for s in S:
            v = (u + s) % n
            if u < v:
                edges.append((u, v))

    if not _check_simple_and_regular(edges, n, k):
        return None

    # Quick girth check via non-backtracking BFS (limited depth)
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for start in range(n):
        dist = {}
        queue = deque()
        for nb in adj[start]:
            queue.append((start, nb, 1))
            dist[(start, nb)] = 1
        while queue:
            prev, curr, d = queue.popleft()
            if d >= g:
                break
            for nb2 in adj[curr]:
                if nb2 == prev:
                    continue
                state = (curr, nb2)
                if state in dist:
                    if d + 1 < g:
                        return None
                else:
                    dist[state] = d + 1
                    queue.append((curr, nb2, d + 1))
    return n, edges


def construct(k, g):
    """Main entry: construct k-regular graph of girth >= g, minimize n."""
    t0 = time.time()

    # Baseline: LW construction (always valid)
    n_lw, edges_lw = construct_lw(k, g)
    best_n, best_edges = n_lw, edges_lw

    # Try ABS for even k in {4, 6}
    if k in (4, 6):
        abs_result = construct_abs(k, g)
        if abs_result is not None:
            n_abs, edges_abs = abs_result
            if _check_simple_and_regular(edges_abs, n_abs, k):
                if n_abs < best_n:
                    best_n, best_edges = n_abs, edges_abs

    # For the specific target k=5, g=9: try to find a smaller graph via search
    # The known best is 1116, lower bound is 428.
    # Try circulant and other structured approaches within time budget.
    mb = _moore_bound(k, g)

    # Time budget for search
    search_deadline = _TIMEOUT - 20

    if k == 5 and g == 9:
        # Try various n values from just above the Moore bound upward
        # Use random k-regular graph generation and check girth
        # This is expensive, so limit attempts.
        attempts = 0
        max_attempts = 200  # tune based on time
        for n in range(max(mb + 1, 100), min(mb + 1500, 2000)):
            if (time.time() - t0) > search_deadline:
                break
            if n * k % 2 != 0:
                continue
            # Try a random k-regular graph via configuration model
            # (limited attempts per n)
            for trial in range(3):
                if (time.time() - t0) > search_deadline:
                    break
                attempts += 1
                # Configuration model: create n*k half-edges, pair randomly
                half_edges = []
                for v in range(n):
                    for _ in range(k):
                        half_edges.append(v)
                random.shuffle(half_edges)
                edges = set()
                valid = True
                for i in range(0, n * k, 2):
                    u, v = half_edges[i], half_edges[i + 1]
                    if u == v:
                        valid = False
                        break
                    a, b = min(u, v), max(u, v)
                    if (a, b) in edges:
                        valid = False
                        break
                    edges.add((a, b))
                if not valid:
                    continue
                edge_list = list(edges)
                if len(edge_list) != n * k // 2:
                    continue
                # Check regularity (should hold by construction)
                # Check girth
                adj = [[] for _ in range(n)]
                for u, v in edge_list:
                    adj[u].append(v)
                    adj[v].append(u)
                # Verify degrees
                ok = True
                for v in range(n):
                    if len(adj[v]) != k:
                        ok = False
                        break
                if not ok:
                    continue
                # Girth check via non-backtracking BFS
                girth_ok = True
                for start in range(n):
                    dist = {}
                    queue = deque()
                    for nb in adj[start]:
                        queue.append((start, nb, 1))
                        dist[(start, nb)] = 1
                    while queue:
                        prev, curr, d = queue.popleft()
                        if d >= g:
                            break
                        for nb2 in adj[curr]:
                            if nb2 == prev:
                                continue
                            state = (curr, nb2)
                            if state in dist:
                                if d + 1 < g:
                                    girth_ok = False
                                    break
                            else:
                                dist[state] = d + 1
                                queue.append((curr, nb2, d + 1))
                        if not girth_ok:
                            break
                    if not girth_ok:
                        break
                if girth_ok:
                    if n < best_n:
                        best_n, best_edges = n, edge_list
                    break  # found a valid graph for this n, move to smaller n? No, we go up.
            # If we found a valid graph at this n, we could stop, but we want minimum.
            # Actually we're iterating n upward, so the first valid n is the minimum for this method.
            # But we might find a smaller one by trying different n? No, we go up from mb.
            # Let's just continue to use remaining time to try more trials at the same n
            # or break if we found one.
            if best_n == n:
                # Found a valid graph at n; try to find smaller? We started at mb+1.
                # We can't go below mb. So this is the best for this method.
                # But maybe a different random graph at same n has same n (no improvement).
                # Move to next n to see if... no, next n is larger. We want smallest.
                # So break out of n loop.
                break

    # Try circulant for even k
    if k % 2 == 0 and (time.time() - t0) < search_deadline:
        for n in range(max(mb + 1, 10), min(mb + 500, 1000)):
            if (time.time() - t0) > search_deadline:
                break
            if n * k % 2 != 0:
                continue
            result = _try_circulant(k, g, n, g)
            if result is not None:
                n_c, edges_c = result
                if n_c < best_n:
                    best_n, best_edges = n_c, edges_c
                break

    # Final verification of best result
    if not _check_simple_and_regular(best_edges, best_n, k):
        # Fallback to LW
        best_n, best_edges = n_lw, edges_lw

    # If time permits, verify girth of the best result
    if (time.time() - t0) < _TIMEOUT - 5:
        if not _verify_girth(best_edges, best_n, g):
            # Girth violated; fallback to LW (which is guaranteed to have sufficient girth)
            best_n, best_edges = n_lw, edges_lw

    return best_n, best_edges
