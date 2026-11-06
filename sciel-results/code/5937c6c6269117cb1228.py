import time
import random

DEADLINE = 140.0

def _has_triangle_fast(adj, n):
    """Fast triangle check using bitsets."""
    for i in range(n):
        ai = adj[i]
        j = ai & ~((1 << (i + 1)) - 1)
        while j:
            lb = j & (-j)
            jj = lb.bit_length() - 1
            common = ai & adj[jj] & ~((1 << (jj + 1)) - 1)
            if common:
                return True
            j ^= lb
    return False

def _greedy_alpha(adj, n):
    """Greedy independent set size."""
    candidates = (1 << n) - 1
    count = 0
    while candidates:
        min_deg = n
        min_v = -1
        c = candidates
        while c:
            lb = c & (-c)
            v = lb.bit_length() - 1
            deg = (adj[v] & candidates).bit_count()
            if deg < min_deg:
                min_deg = deg
                min_v = v
            c ^= lb
        if min_v == -1:
            break
        count += 1
        candidates &= ~adj[min_v]
        candidates &= ~(1 << min_v)
    return count

def _alpha_exact_bnb(adj, n, limit=10):
    """Exact max independent set size via branch and bound."""
    best = [0]
    def bnb(chosen_count, candidates):
        if chosen_count + candidates.bit_count() <= best[0]:
            return
        if chosen_count >= limit:
            best[0] = chosen_count
            return
        if candidates == 0:
            if chosen_count > best[0]:
                best[0] = chosen_count
            return
        lb = candidates & (-candidates)
        v = lb.bit_length() - 1
        rest = candidates ^ lb
        bnb(chosen_count, rest)
        new_cands = rest & ~adj[v]
        bnb(chosen_count + 1, new_cands)
    bnb(0, (1 << n) - 1)
    return best[0]

def _is_valid(adj, n):
    """Check triangle-free and alpha < 10."""
    if _has_triangle_fast(adj, n):
        return False
    if _greedy_alpha(adj, n) >= 10:
        return False
    if _alpha_exact_bnb(adj, n, limit=10) >= 10:
        return False
    return True

def _random_maximal_triangle_free(n):
    """Generate a random maximal triangle-free graph on n vertices."""
    adj = [0] * n
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    random.shuffle(pairs)
    for u, v in pairs:
        common = adj[u] & adj[v]
        if common == 0:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    return adj

def _build_circulant(n, steps):
    """Build circulant graph C(n, steps) where steps are symmetric."""
    adj = [0] * n
    for i in range(n):
        for s in steps:
            j = (i + s) % n
            if j > i:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    start_time = time.time()
    
    # Strategy 1: Circulant graphs for n=40, 39
    for n in [40, 39]:
        if time.time() > start_time + DEADLINE - 30:
            break
        for attempt in range(80):
            if time.time() > start_time + DEADLINE - 30:
                break
            max_step = n // 2
            steps = set()
            for s in range(1, max_step + 1):
                if random.random() < 0.35:
                    steps.add(s)
            if not steps:
                continue
            adj = _build_circulant(n, list(steps))
            if _is_valid(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
    
    # Strategy 2: Random maximal triangle-free for n=40, 39, 38
    for n in [40, 39, 38, 37, 36]:
        if time.time() > start_time + DEADLINE - 15:
            break
        for attempt in range(150):
            if time.time() > start_time + DEADLINE - 15:
                break
            adj = _random_maximal_triangle_free(n)
            if _is_valid(adj, n):
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
    
    # Fallback: smaller n
    for n_try in range(35, 20, -1):
        if time.time() > start_time + DEADLINE - 5:
            break
        for attempt in range(80):
            if time.time() > start_time + DEADLINE - 5:
                break
            adj = _random_maximal_triangle_free(n_try)
            if _is_valid(adj, n_try):
                edges = []
                for i in range(n_try):
                    for j in range(i + 1, n_try):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n_try, 'edges': edges}
    
    return {'n': 0, 'edges': []}
