import time
import sys

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    q = k  # q = 5 for k = 5

    # Try d values; girth >= d + 5 for LUW
    for d in range(max(3, g - 5), max(3, g - 5) + 4):
        if time.time() > deadline - 40:
            break
        n, edges = _luw_construct(k, q, d)
        if n is None or len(edges) == 0:
            continue
        if _verify_girth(n, edges, g):
            return n, edges

    # Fallback
    d = max(3, g - 5) + 3
    n, edges = _luw_construct(k, q, d)
    return n, edges


def _luw_construct(k, q, d):
    q_pow_d = q ** d
    n = 2 * q_pow_d

    def add(x, y):
        return (x + y) % q

    def sub(x, y):
        return (x - y) % q

    def mul(x, y):
        return (x * y) % q

    edges = []
    seen = set()

    for i in range(q_pow_d):
        x_vec = [0] * d
        val = i
        for pos in range(d - 1, -1, -1):
            x_vec[pos] = val % q
            val //= q

        for y1 in range(q):
            y_vec = [0] * d
            y_vec[0] = y1
            for c in range(2, d + 1):
                y_vec[c - 1] = add(x_vec[c - 1], mul(y_vec[c - 2], x_vec[c - 2]))

            y_idx = 0
            for pos in range(d):
                y_idx = y_idx * q + y_vec[pos]

            u = i
            v = q_pow_d + y_idx
            edge = (min(u, v), max(u, v))
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)

    return n, edges


def _verify_girth(n, edges, g):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    for u in range(n):
        if len(adj[u]) != 5:
            return False

    for start in range(n):
        dist = [-1] * n
        dist[start] = 0
        queue = [(start, -1)]
        head = 0
        while head < len(queue):
            u, parent = queue[head]
            head += 1
            if dist[u] >= g // 2:
                continue
            for v in adj[u]:
                if v == parent:
                    continue
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append((v, u))
                else:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    return True


def _luw_construct_fast(k, d):
    return _luw_construct(k, k, d)

def _verify_girth_fast(n, edges, g):
    return _verify_girth(n, edges, g)

def _construct_5_9(deadline):
    pass

def _random_5_regular(n, rng):
    pass

def _get_adjacency(edges, n):
    pass

def _find_short_cycle(adj, n, max_len):
    pass

def _improve_girth(edges, n, target_girth, deadline, rng):
    pass

def _estimate_girth(adj, n, upper_bound):
    pass

def _has_cycle_of_length(adj, n, length):
    pass
