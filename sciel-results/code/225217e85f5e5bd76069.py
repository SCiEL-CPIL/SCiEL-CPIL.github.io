import time
import random

N = 12
MAX_M = 1000000
DEADLINE = 85.0  # seconds, safety margin below 90s

def _can_match(m, L, n=N):
    """Exact: does (m, m+L] contain distinct b_1..b_n with i | b_i?"""
    adj = []
    for i in range(1, n + 1):
        first = ((m // i) + 1) * i
        mask = 0
        v = first
        end = m + L
        while v <= end:
            idx = v - (m + 1)
            if 0 <= idx < L:
                mask |= (1 << idx)
            v += i
        adj.append(mask)

    pair_u = [-1] * n
    pair_v = [-1] * L
    dist = [0] * n

    def bfs():
        from collections import deque
        q = deque()
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = -1
        found = False
        while q:
            u = q.popleft()
            msk = adj[u]
            while msk:
                lb = msk & (-msk)
                idx = lb.bit_length() - 1
                msk ^= lb
                pu = pair_v[idx]
                if pu == -1:
                    found = True
                elif dist[pu] == -1:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found

    def dfs(u):
        msk = adj[u]
        while msk:
            lb = msk & (-msk)
            idx = lb.bit_length() - 1
            msk ^= lb
            pu = pair_v[idx]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_v[idx] = u
                pair_u[u] = idx
                return True
        dist[u] = -1
        return False

    matching = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1 and dfs(u):
                matching += 1
    return matching == n

def _f_val(m, n=N):
    """Least positive L with a valid matching."""
    lo, hi = 1, 144
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best if best is not None else 144

def construct():
    start = time.time()
    random.seed(42)
    
    def timed():
        return time.time() - start > DEADLINE

    best_m = 0
    best_f = _f_val(0)
    seen = {0}

    # Check some specific candidates first
    candidates = [0, 1, 11, 12, 13, 27720, 27721, 27719]
    for m in candidates:
        if 0 <= m <= MAX_M and m not in seen:
            seen.add(m)
            if timed():
                break
            f = _f_val(m)
            if f > best_f:
                best_f = f
                best_m = m

    # Random search
    while not timed():
        m = random.randint(0, MAX_M)
        if m in seen:
            continue
        seen.add(m)
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m
            
    return int(best_m)
