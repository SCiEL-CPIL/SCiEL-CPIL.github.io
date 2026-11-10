"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

N = 12
MAX_M = 1000000
LCM_12 = 27720  # lcm(1..12)

def _has_perfect_matching(m, L):
    """Check if (m, m+L] contains distinct b_1..b_12 with i|b_i."""
    adj = [[] for _ in range(N)]
    
    for i in range(1, N + 1):
        li = i - 1
        start = ((m // i) + 1) * i
        for v in range(start, m + L + 1, i):
            if v > m and v <= m + L:
                right_idx = v - (m + 1)
                if 0 <= right_idx < L:
                    adj[li].append(right_idx)
    
    # Order left nodes by fewest candidates first for faster failure
    order = sorted(range(N), key=lambda li: len(adj[li]))
    
    matchR = [-1] * L
    
    def dfs(left_idx, visited):
        for ri in adj[left_idx]:
            if visited[ri]:
                continue
            visited[ri] = True
            if matchR[ri] == -1 or dfs(matchR[ri], visited):
                matchR[ri] = left_idx
                return True
        return False
    
    for li in order:
        visited = [False] * L
        if not dfs(li, visited):
            return False
    return True


def _compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has 12 distinct b_i with i|b_i."""
    lo = 1
    hi = 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _has_perfect_matching(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def construct():
    """Find m in [0, 1000000] maximizing f(12, m)."""
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill
    
    best_m = 0
    best_f = 0
    
    # Key insight: f(12, m) is periodic with period lcm(1..12) = 27720
    # So we only need to search m in [0, 27720)
    # Do a fine-grained scan with step 1 over [0, 27720)
    
    for m in range(0, LCM_12):
        if time.time() - start_time > DEADLINE:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
            if best_f == 144:  # Can't do better
                break
    
    best_m = int(best_m)
    if best_m < 0:
        best_m = 0
    if best_m > MAX_M:
        best_m = MAX_M
    
    return best_m

# EVOLVE-BLOCK-END
