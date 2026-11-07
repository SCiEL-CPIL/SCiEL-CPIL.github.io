import time
import random

def _can_match(m, L, n=12):
    # Check if (m, m+L] contains distinct b_1..b_n with i | b_i
    # For n=12, use bitmask matching for speed
    n = 12
    # Build candidate lists for each i (1-indexed)
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        r = ((m // i) + 1) * i
        end = m + L
        v = r
        while v <= end:
            adj[i - 1].append(v)
            v += i

    # Try to find matching using backtracking with bitmasks
    # For each left node, try to assign a right node
    # Use bitsets for right nodes
    # Since right nodes are actual values, map them to indices
    # But values can be large; instead, use the fact that n is small
    
    # Simpler: use standard DFS matching but with array-based visited
    # Right side: collect all unique values, map to indices
    all_vals = set()
    for i in range(n):
        for v in adj[i]:
            all_vals.add(v)
    vals = sorted(all_vals)
    val_to_idx = {v: i for i, v in enumerate(vals)}
    rcount = len(vals)
    
    # Build adjacency as index lists
    adj_idx = []
    for i in range(n):
        adj_idx.append([val_to_idx[v] for v in adj[i]])
    
    match_r = [-1] * rcount  # right_idx -> left_idx
    
    def dfs(u, visited):
        for vi in adj_idx[u]:
            if visited[vi]:
                continue
            visited[vi] = True
            if match_r[vi] == -1 or dfs(match_r[vi], visited):
                match_r[vi] = u
                return True
        return False
    
    count = 0
    for u in range(n):
        visited = [False] * rcount
        if dfs(u, visited):
            count += 1
        else:
            return False
    return count == n

def _f(m, n=12):
    # binary search smallest L in [1, 200]
    lo, hi = 1, 200
    if not _can_match(m, hi, n):
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    deadline = time.time() + 75  # 75s budget, 15s safety
    best_m = 12
    best_f = _f(12)
    
    # Phase 1: Coarse scan with step 5
    step = 5
    m = 0
    while m <= 1000000:
        if time.time() > deadline:
            break
        f = _f(m)
        if f > best_f:
            best_f = f
            best_m = m
        m += step
    
    # Phase 2: Finer scan around best_m
    if time.time() < deadline:
        for dm in range(-100, 101):
            if time.time() > deadline:
                break
            mm = best_m + dm
            if mm < 0 or mm > 1000000:
                continue
            f = _f(mm)
            if f > best_f:
                best_f = f
                best_m = mm
    
    # Phase 3: If time remains, do another coarse pass with different step
    if time.time() < deadline:
        step2 = 11
        m = 3  # offset to avoid overlap
        while m <= 1000000:
            if time.time() > deadline:
                break
            f = _f(m)
            if f > best_f:
                best_f = f
                best_m = m
            m += step2
    
    # Final polish: check neighbors of best
    if time.time() < deadline:
        for dm in range(-50, 51):
            if time.time() > deadline:
                break
            mm = best_m + dm
            if mm < 0 or mm > 1000000:
                continue
            f = _f(mm)
            if f > best_f:
                best_f = f
                best_m = mm
    
    return int(best_m)
