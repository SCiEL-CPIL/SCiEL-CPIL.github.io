import time
import sys
import math
from functools import reduce

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _can_match(m, L, n=12):
    """
    Check if (m, m+L] contains distinct b_1..b_n with i | b_i.
    Uses bipartite matching.
    """
    adj = [[] for _ in range(n)]
    val_to_idx = {}
    right_vals = []
    
    end = m + L
    
    for i in range(1, n + 1):
        # First multiple of i in (m, m+L]
        # m < v <= m+L
        # v = k*i
        # k*i > m => k > m/i => k >= floor(m/i) + 1
        k_start = (m // i) + 1
        v = k_start * i
        if v > end:
            return False
        
        adj_i = []
        while v <= end:
            if v not in val_to_idx:
                val_to_idx[v] = len(right_vals)
                right_vals.append(v)
            adj_i.append(val_to_idx[v])
            v += i
        adj[i - 1] = adj_i

    # Map right side values to indices for efficient matching
    # adj already contains indices
    
    # Order left nodes by number of candidates (most constrained first)
    order = sorted(range(n), key=lambda u: len(adj[u]))
    
    match_r = [-1] * len(right_vals)
    sys.setrecursionlimit(10000)

    def dfs(u, visited):
        for v_idx in adj[u]:
            if visited[v_idx]:
                continue
            visited[v_idx] = True
            if match_r[v_idx] == -1 or dfs(match_r[v_idx], visited):
                match_r[v_idx] = u
                return True
        return False

    for u in order:
        visited = [False] * len(right_vals)
        if not dfs(u, visited):
            return False
    return True

def _f(m, n=12):
    """Find smallest L in [1, 144] such that (m, m+L] has valid matching."""
    lo, hi = 1, 144
    # Quick check: if L=144 doesn't work, return 144 (or higher, but max is 144 per task)
    # Actually, task says f is at most 144.
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
    deadline = time.time() + 85  # 5s safety margin
    
    period = reduce(_lcm, range(1, 13))
    
    best_f = 0
    best_r = 0
    
    # We iterate through all residues in the period
    # To speed up, we can check residues in an order likely to yield high f
    # Or just iterate all. 27720 * 8 * cost(_can_match)
    
    for r in range(period):
        if time.time() > deadline:
            break
            
        f_val = _f(r)
        if f_val > best_f:
            best_f = f_val
            best_r = r
            
    # Find the largest m <= 1000000 with m % period == best_r
    if best_r > 1000000:
        # This shouldn't happen since period < 1000000
        m = best_r % period
    else:
        k_max = (1000000 - best_r) // period
        m = best_r + k_max * period
        
    # Verify and potentially refine with neighbors (just in case of edge cases or if f is not strictly periodic in a way that matters for the "smallest L" definition? 
    # The definition is exact. f(m) depends only on m mod P.
    # So m is the optimal answer.
    
    # However, the task asks for m in 0..1000000.
    # If there are multiple r with the same max f, we should pick the one that allows the largest m?
    # No, the score is the raw value f(12, m). All r with max f give the same score.
    # But we must return a valid m.
    # The largest m is a good choice.
    
    # Let's double check if there are other residues with the same best_f that might have a "larger" m?
    # The m is determined by r. m = r + k*P.
    # For a fixed r, the largest m is unique.
    # Different r's give different m's.
    # The score is the same.
    # So any r with max f is fine.
    
    return int(m)
