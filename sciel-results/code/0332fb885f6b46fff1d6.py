import time
import math
from functools import reduce

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _can_match(m, L, n=12):
    # Check if (m, m+L] contains distinct b_1..b_n with i | b_i
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        r = ((m // i) + 1) * i
        v = r
        end = m + L
        while v <= end:
            adj[i - 1].append(v)
            v += i
        if not adj[i - 1]:
            return False

    # Map right side values to indices for efficient matching
    val_to_idx = {}
    right_vals = []
    for i in range(n):
        for v in adj[i]:
            if v not in val_to_idx:
                val_to_idx[v] = len(right_vals)
                right_vals.append(v)
    
    # Convert adj to index-based
    adj_idx = [[val_to_idx[v] for v in adj[i]] for i in range(n)]
    
    # Sort by degree (fewest options first) for better pruning
    order = sorted(range(n), key=lambda u: len(adj_idx[u]))
    
    match_r = [-1] * len(right_vals)
    
    def dfs(u, visited):
        for v_idx in adj_idx[u]:
            if visited[v_idx]:
                continue
            visited[v_idx] = True
            if match_r[v_idx] == -1 or dfs(match_r[v_idx], visited):
                match_r[v_idx] = u
                return True
        return False

    count = 0
    for u in order:
        visited = [False] * len(right_vals)
        if dfs(u, visited):
            count += 1
        else:
            return False
    return count == n

def _f(m, n=12):
    # Binary search smallest L in [1, 144]
    lo, hi = 1, 144
    if not _can_match(m, hi, n):
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid, n):
            hi = mid
        else:
            lo = mid + 1
    return lo

def _score_m(m, n=12):
    # Heuristic: sum of first multiple offsets
    score = 0
    for i in range(1, n + 1):
        r = ((m // i) + 1) * i
        offset = r - m
        score += offset
    return score

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    p = reduce(_lcm, range(1, 13))
    
    # Generate all m in [0, p-1] that are "interesting"
    # Focus on m where m+1 is divisible by at least one i in 2..12
    # (i=1 always works, so it's not a constraint)
    candidate_ms = set()
    for i in range(2, 13):
        for k in range(1, p // i + 1):
            m_val = k * i - 1
            if 0 <= m_val < p:
                candidate_ms.add(m_val)
    
    candidate_ms.add(0)
    
    # Sort by heuristic score (higher = potentially worse f(m))
    # We want to find the m with MAXIMUM f(m), so check high-score m first
    candidate_list = sorted(candidate_ms, key=lambda m: -_score_m(m))
    
    best_m = 0
    best_f = _f(0)
    
    for m in candidate_list:
        if time.time() > deadline:
            break
        f_val = _f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return int(best_m)
