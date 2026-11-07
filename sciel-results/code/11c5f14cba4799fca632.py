import time
import random

N = 12
MAX_M = 1000000
DEADLINE = 85  # seconds, safety margin

def _can_match(m, L, n=N):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i."""
    # Build adjacency: for each i in 1..n, list of multiples of i in (m, m+L]
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        r = ((m // i) + 1) * i
        end = m + L
        v = r
        while v <= end:
            adj[i - 1].append(v)
            v += i
    
    # Bipartite matching via DFS augmenting paths
    # match_right maps value -> left index (0-based), or -1
    match_right = {}
    
    def dfs(u, visited):
        for v in adj[u]:
            if visited.get(v, False):
                continue
            visited[v] = True
            if v not in match_right or dfs(match_right[v], visited):
                match_right[v] = u
                return True
        return False
    
    for u in range(n):
        visited = {}
        if not dfs(u, visited):
            return False
    return True

def _f_val(m, n=N):
    """Find the least positive L such that (m, m+L] has a valid matching."""
    # Binary search on L
    lo, hi = 1, 200  # L is at most ~144 per task
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            best = mid
            hi = mid - 1
        else:
            lo = mid + 1
    return best if best is not None else 200

def construct():
    start_time = time.time()
    
    # First, evaluate some promising candidates
    # m=0 is a natural candidate
    candidates = [0, 1, 2, 3, 5, 6, 10, 11, 12, 20, 24, 30, 36, 40, 42, 50, 60, 70, 80, 90, 100]
    
    # Add random samples
    random.seed(42)
    for _ in range(50):
        candidates.append(random.randint(0, MAX_M))
    
    # Add candidates near multiples of small numbers (where patterns change)
    for base in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        for k in range(0, MAX_M // base + 1, max(1, MAX_M // (base * 20))):
            m_candidate = k * base
            if 0 <= m_candidate <= MAX_M:
                candidates.append(m_candidate - 1)
                candidates.append(m_candidate)
                candidates.append(m_candidate + 1)
    
    # Deduplicate
    candidates = list(set(candidates))
    
    best_m = 0
    best_f = _f_val(0)
    
    # Evaluate candidates within time budget
    for m in candidates:
        if time.time() - start_time > DEADLINE:
            break
        if m < 0 or m > MAX_M:
            continue
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    # If we have time, do a more systematic search around the best region
    if time.time() - start_time < DEADLINE - 10:
        # Search in a window around the best m found
        window = 100
        for m in range(max(0, best_m - window), min(MAX_M, best_m + window) + 1):
            if time.time() - start_time > DEADLINE:
                break
            f = _f_val(m)
            if f > best_f:
                best_f = f
                best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
