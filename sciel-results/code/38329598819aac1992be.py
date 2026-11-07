import time
import sys
import math
from functools import reduce

def _lcm(a, b):
    return a * b // math.gcd(a, b)

def _can_match(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1..b_n with i | b_i using bipartite matching."""
    adj = [[] for _ in range(n)]
    for i in range(1, n + 1):
        r = ((m // i) + 1) * i
        end = m + L
        if r > end:
            return False
        v = r
        while v <= end:
            adj[i - 1].append(v)
            v += i

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

    match_r = [-1] * len(right_vals)

    sys.setrecursionlimit(10000)

    def dfs(u, visited):
        for v_idx in adj_idx[u]:
            if visited[v_idx]:
                continue
            visited[v_idx] = True
            if match_r[v_idx] == -1 or dfs(match_r[v_idx], visited):
                match_r[v_idx] = u
                return True
        return False

    for u in range(n):
        visited = [False] * len(right_vals)
        if not dfs(u, visited):
            return False
    return True

def _f(m, n=12):
    """Find smallest L in [1, 144] such that (m, m+L] has valid matching."""
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

def construct():
    deadline = time.time() + 85  # 85s budget, 5s safety margin

    # LCM of 1..12 = 27720; f(m) is periodic with this period
    period = reduce(_lcm, range(1, 13))

    # Generate critical m values in [0, period-1] where f(m) could change:
    # m is critical if m+1 is a multiple of some i in 1..12
    critical_ms = set()
    for i in range(1, 13):
        for k in range(1, period // i + 1):
            m_val = k * i - 1
            if 0 <= m_val < period:
                critical_ms.add(m_val)
    critical_ms.add(0)

    critical_list = sorted(critical_ms)

    best_residue = 0
    best_f = 0

    # Evaluate f at each critical point
    for r in critical_list:
        if time.time() > deadline - 5:
            break
        f_val = _f(r)
        if f_val > best_f:
            best_f = f_val
            best_residue = r

    # Find the largest m <= 1000000 with m % period == best_residue
    if best_residue <= 1000000:
        k_max = (1000000 - best_residue) // period
        best_m = best_residue + k_max * period
    else:
        best_m = 0

    # Local refinement: check neighbors of best_m and other high-scoring residues
    if time.time() < deadline - 5:
        candidates = set()
        candidates.add(best_m)

        # Check all critical residues that achieved the same best_f
        for r in critical_list:
            if time.time() > deadline - 5:
                break
            if _f(r) == best_f:
                if r <= 1000000:
                    k_max = (1000000 - r) // period
                    mm = r + k_max * period
                    candidates.add(mm)
                    for delta in range(-3, 4):
                        if 0 <= mm + delta <= 1000000:
                            candidates.add(mm + delta)

        # Also check immediate neighbors of best_m
        for delta in range(-5, 6):
            mm = best_m + delta
            if 0 <= mm <= 1000000:
                candidates.add(mm)

        for mm in sorted(candidates):
            if time.time() > deadline - 3:
                break
            f_val = _f(mm)
            if f_val > best_f:
                best_f = f_val
                best_m = mm

    return int(best_m)
