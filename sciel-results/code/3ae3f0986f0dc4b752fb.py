"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque

P = {'n': 12, 'maximum_start': 1000000}

# Precompute LCM of 1..12
_lcm_12 = 1
for _i in range(1, 13):
    _lcm_12 = _lcm_12 * _i // math.gcd(_lcm_12, _i)
# _lcm_12 = 27720

def _has_matching(m, L, n=12):
    """Check if (m, m+L] contains distinct b_1,...,b_n with i | b_i using Hopcroft-Karp."""
    # Build adjacency: for each divisor i (0-indexed: i-1), list of candidate numbers in (m, m+L]
    # Map numbers to indices for compact representation
    num_to_idx = {}
    cand_list = []
    adj = [[] for _ in range(n)]
    
    for i in range(1, n + 1):
        start = m + 1
        end = m + L
        first = ((start + i - 1) // i) * i
        last = (end // i) * i
        if first > last:
            return False
        for num in range(first, last + 1, i):
            if num not in num_to_idx:
                num_to_idx[num] = len(cand_list)
                cand_list.append(num)
            adj[i - 1].append(num_to_idx[num])
    
    num_right = len(cand_list)
    if num_right < n:
        return False
    
    # Hopcroft-Karp
    pair_u = [-1] * n
    pair_v = [-1] * num_right
    dist = [0] * n
    
    INF = float('inf')
    
    def bfs():
        q = deque()
        found = False
        for u in range(n):
            if pair_u[u] == -1:
                dist[u] = 0
                q.append(u)
            else:
                dist[u] = INF
        while q:
            u = q.popleft()
            for v in adj[u]:
                pu = pair_v[v]
                if pu == -1:
                    found = True
                elif dist[pu] == INF:
                    dist[pu] = dist[u] + 1
                    q.append(pu)
        return found
    
    def dfs(u):
        for v in adj[u]:
            pu = pair_v[v]
            if pu == -1 or (dist[pu] == dist[u] + 1 and dfs(pu)):
                pair_u[u] = v
                pair_v[v] = u
                return True
        dist[u] = INF
        return False
    
    matching = 0
    while bfs():
        for u in range(n):
            if pair_u[u] == -1:
                if dfs(u):
                    matching += 1
    
    return matching == n

def _compute_f(m, n=12, max_L=144):
    """Compute f(n, m): least positive L such that (m, m+L] contains distinct b_i with i|b_i."""
    for L in range(12, max_L + 1):  # L must be at least n=12
        if _has_matching(m, L, n):
            return L
    return max_L

def construct():
    """Find m in [0, 1000000] that maximizes f(12, m)."""
    deadline = time.time() + 85
    
    # Key insight: f(12, m) depends only on m mod 27720
    # Because the divisibility pattern of multiples of 1..12 repeats with period lcm(1..12)=27720
    # So we only need to check m in [0, 27719]
    
    period = _lcm_12  # 27720
    best_m = 0
    best_f = 0
    
    # Exhaustive search over one period
    for r in range(period):
        if time.time() > deadline - 5:
            break
        f_val = _compute_f(r)
        if f_val > best_f:
            best_f = f_val
            best_m = r
            if best_f >= 144:
                break
    
    # Verify: for any m in [0, 1000000], f(12, m) = f(12, m % period)
    # So the best m in [0, 1000000] is some multiple offset from best_m
    # We want the smallest m in [0, 1000000] with m % period == best_m
    # That's just best_m itself (since best_m < period <= 1000000)
    
    return best_m
# EVOLVE-BLOCK-END
