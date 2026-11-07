"""Find m in [0,1e6] maximizing f(12,m): least L with distinct b_i in (m,m+L], i|b_i."""
import math
import time

P = 27720  # lcm(1..12)
MAX_M = 1000000
DEADLINE = 80.0  # safety margin under 90s wall-clock

def _hopcroft_karp(adj):
    """adj: list of lists (0-indexed left -> right). Returns max matching size."""
    n = len(adj)
    m = max((len(a) for a in adj), default=0)
    pair_u = [-1] * n
    pair_v = [-1] * m
    dist = [0] * n
    from collections import deque

    def bfs():
        dq = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                dq.append(u)
            else:
                dist[u] = -1
        found = False
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    dq.append(pu)
        return found

    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1 and dfs(u):
                matching += 1
    return matching

def _has_match(m, L):
    """Does (m, m+L] contain 12 distinct b_i with i|b_i?"""
    adj = []
    for i in range(1, 13):
        # multiples of i in (m, m+L]
        lo = m + 1
        hi = m + L
        # first multiple >= lo
        first = ((lo + i - 1) // i) * i
        if first > hi:
            adj.append([])
            continue
        # count multiples in [first, hi]
        cnt = (hi - first) // i + 1
        if cnt > 60:  # cap for speed; more than enough
            cnt = 60
        adj.append(list(range(first, first + cnt * i, i)))
    # quick check: total right nodes
    total = sum(len(a) for a in adj)
    if total < 12:
        return False
    return _hopcroft_karp(adj) == 12

def _min_L(m):
    """Least positive L such that (m, m+L] has a valid matching."""
    lo, hi = 1, 200
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    t0 = time.time()
    best_m = 12
    best_L = _min_L(12)

    # Search over one period [0, P-1]; the value f(12,m) depends only on m mod P
    # We scan in chunks, respecting the deadline.
    # First pass: coarse scan every 1 to find a good candidate quickly
    for m in range(0, P):
        if time.time() - t0 > DEADLINE - 5:
            break
        L = _min_L(m)
        if L > best_L:
            best_L = L
            best_m = m

    # If we have time, refine: the best in [0,P-1] is the global best since
    # f(12, m) = f(12, m mod P). So best_m in [0, P-1] is optimal.
    # But we need m in [0, 1e6]. best_m is already in that range.
    return int(best_m)
