"""Find m in [0, 1000000] maximizing f(12, m)."""
import time
import sys
from math import gcd
from functools import lru_cache

N = 12
MAX_M = 1000000
DEADLINE = 85.0  # safety margin before 90s kill


def _has_perfect_matching(m, L):
    """Check if (m, m+L] contains distinct b_1..b_12 with i|b_i."""
    # Build candidate sets for each i
    # For divisor i, candidates in (m, m+L] divisible by i
    # We need a perfect matching in bipartite graph
    # Use DFS with backtracking (Hungarian would be overkill for 12)
    
    # Sort by number of candidates (most constrained first) for pruning
    candidates = []
    for i in range(1, N + 1):
        # smallest multiple of i in (m, m+L]
        # first multiple > m
        first = ((m // i) + 1) * i
        # last multiple <= m+L
        last = ((m + L) // i) * i
        cnt = 0
        if first <= last:
            cnt = (last - first) // i + 1
        candidates.append((cnt, i, first, last))
    
    # Quick check: if any i has 0 candidates, fail
    for cnt, i, first, last in candidates:
        if cnt == 0:
            return False
    
    # Sort by fewest candidates first
    candidates.sort(key=lambda x: x[0])
    
    # Backtracking with bitmasks
    # For each i, the candidates are first, first+i, first+2i, ...
    # We'll use DFS assigning i's in order of increasing candidate count
    
    used = [False] * (L + 1)  # used[offset] where offset = b - m, 1..L
    # Actually, let's track which specific values are used
    # Values range from m+1 to m+L
    
    # For efficiency, precompute candidate lists
    cand_lists = []
    for cnt, i, first, last in candidates:
        lst = []
        v = first
        while v <= last:
            lst.append(v)
            v += i
        cand_lists.append((i, lst))
    
    # DFS
    def dfs(idx, used_mask):
        if idx == N:
            return True
        i, lst = cand_lists[idx]
        for v in lst:
            bit = v - m  # 1..L
            if not (used_mask & (1 << bit)):
                if dfs(idx + 1, used_mask | (1 << bit)):
                    return True
        return False
    
    return dfs(0, 0)


def _compute_f(m):
    """Compute f(12, m) via binary search on L."""
    # f is at most 144 per problem statement
    # Binary search for smallest L with perfect matching
    lo, hi = 1, 144
    # First check if L=144 works (should always)
    if not _has_perfect_matching(m, 144):
        # Should not happen, but return 144 as fallback
        return 144
    
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def construct():
    start_time = time.time()
    
    best_m = 12
    best_f = _compute_f(12)
    
    # Strategy: 
    # 1. Scan strategic points: near multiples of LCM(1..12)=27720
    # 2. Also scan uniformly
    # 3. Local refinement around best found
    
    # LCM of 1..12
    lcm = 1
    for i in range(2, 13):
        lcm = lcm * i // gcd(lcm, i)
    # lcm = 27720
    
    # Generate candidate m values
    candidates = set()
    
    # Near multiples of lcm
    for k in range(0, MAX_M // lcm + 1):
        base = k * lcm
        for offset in range(-50, 51):
            m = base + offset
            if 0 <= m <= MAX_M:
                candidates.add(m)
    
    # Also near multiples of smaller lcms
    # LCM of 1..6 = 60, LCM of 1..8 = 840
    for k in range(0, MAX_M // 840 + 1):
        base = k * 840
        for offset in range(-20, 21):
            m = base + offset
            if 0 <= m <= MAX_M:
                candidates.add(m)
    
    # Uniform scan every 100
    for m in range(0, MAX_M + 1, 100):
        candidates.add(m)
    
    # Sort candidates for deterministic order
    candidates = sorted(candidates)
    
    # Evaluate candidates
    for m in candidates:
        if time.time() > DEADLINE - 5.0:
            break
        f = _compute_f(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    # Local refinement around best_m
    if time.time() < DEADLINE - 3.0:
        for dm in range(-50, 51):
            if time.time() > DEADLINE - 1.0:
                break
            m = best_m + dm
            if 0 <= m <= MAX_M:
                f = _compute_f(m)
                if f > best_f:
                    best_f = f
                    best_m = m
    
    # Second refinement pass if time allows
    if time.time() < DEADLINE - 2.0:
        for dm in range(-20, 21):
            if time.time() > DEADLINE - 0.5:
                break
            m = best_m + dm
            if 0 <= m <= MAX_M:
                f = _compute_f(m)
                if f > best_f:
                    best_f = f
                    best_m = m
    
    return int(best_m)


if __name__ == '__main__':
    # Quick test
    m = construct()
    print(f"Best m = {m}, f = {_compute_f(m)}")
