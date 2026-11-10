"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

N = 12
MAX_M = 1000000

def _has_perfect_matching(m, L):
    """Check if (m, m+L] contains 12 distinct b_i with i|b_i.
    Left side: indices 1..12. Right side: integers in (m, m+L].
    Edge i-j exists if j is in (m, m+L] and i divides j.
    """
    # Build adjacency: for each i in 1..12, which numbers in (m, m+L] are divisible by i
    # Use Hopcroft-Karp or simple DFS augmenting paths (N=12 is small)
    
    # Collect candidate numbers
    # For each i, the numbers in (m, m+L] divisible by i
    # We need to pick distinct numbers
    
    # Simple approach: build bipartite graph and do augmenting path
    # Left: 1..12 (indices 0..11)
    # Right: we'll use the actual numbers, but map to indices
    
    # For efficiency, let's just enumerate
    # For each i, list of candidates in (m, m+L] divisible by i
    # Then do bipartite matching
    
    # Build adjacency list for left nodes
    # left[i] = list of right-node indices
    # We need to enumerate right nodes: all numbers in (m, m+L]
    # But that's L numbers, up to 144. And 12 left nodes.
    
    # Actually, let's just do it directly
    # For matching, we need: assign each i a distinct number b_i in (m,m+L] with i|b_i
    
    # Use simple DFS-based bipartite matching
    # left: 0..11 (representing i=1..12)
    # right: the numbers in (m, m+L], indexed 0..L-1
    
    # adj[left_idx] = list of right_idx
    adj = [[] for _ in range(N)]
    # right_nodes: list of actual values
    right_vals = list(range(m + 1, m + L + 1))
    # val_to_idx: map value to right index
    # But values are consecutive, so idx = val - (m+1)
    
    for i in range(1, N + 1):
        li = i - 1
        # Find numbers in (m, m+L] divisible by i
        # First multiple of i greater than m:
        start = ((m // i) + 1) * i
        if start <= m:
            start += i
        # Actually: smallest x > m with x % i == 0
        # x = m + 1 + ((i - (m+1) % i) % i)
        # Let's just: start = ((m + 1 + i - 1) // i) * i  -- no
        # smallest multiple of i > m:
        # if (m+1) % i == 0: start = m+1
        # else: start = (m+1) + (i - (m+1)%i)
        # Simpler: start = ((m // i) + 1) * i
        # Check: m=10, i=3: (10//3+1)*3 = 4*3=12. 12>10, 12%3=0. Correct.
        # m=9, i=3: (9//3+1)*3 = 4*3=12. 12>9. Correct.
        # m=11, i=3: (11//3+1)*3 = 4*3=12. Correct.
        start = ((m // i) + 1) * i
        for v in range(start, m + L + 1, i):
            if v > m and v <= m + L:
                right_idx = v - (m + 1)
                if 0 <= right_idx < L:
                    adj[li].append(right_idx)
    
    # Bipartite matching: match left to right
    # matchR[right_idx] = left_idx or -1
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
    
    for li in range(N):
        visited = [False] * L
        if not dfs(li, visited):
            return False
    return True


def _compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has 12 distinct b_i with i|b_i."""
    # Lower bound: at least 12 (need 12 distinct numbers)
    # Upper bound: 144 (LCM of 1..12 is 27720, but empirically bounded)
    # Actually the max possible is bounded. Let's use a safe upper bound.
    # For L=144, we have 144 numbers, and each i has ~144/i multiples, so matching should exist.
    # Let's verify: for i=12, need at least 1 multiple of 12 in (m, m+144]. Always true since 144>=12.
    # For i=1, 2, ..., 12: each has at least floor(144/i) >= 12 candidates (for i<=12).
    # So L=144 should always work. Let's use 144 as upper bound.
    
    lo = 1
    hi = 144  # safe upper bound
    
    # Binary search for smallest L
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
    DEADLINE = 85.0  # safety margin before 90s kill
    
    best_m = 12
    best_f = _compute_f(12)
    
    # Strategy 1: Coarse scan
    # Scan m in steps to find regions with high f
    # f(12,m) depends on the local structure of multiples
    # High f occurs when for small L, some i has very few multiples in (m, m+L]
    
    # Coarse scan: step size 100 initially
    step = 100
    m = 0
    while m <= MAX_M:
        if time.time() - start_time > DEADLINE:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
        m += step
    
    # Strategy 2: Local refinement around best found
    # Check neighbors of best_m with smaller step
    if time.time() - start_time < DEADLINE:
        # Refine around best_m
        for dm in range(-50, 51):
            if time.time() - start_time > DEADLINE:
                break
            m_test = best_m + dm
            if 0 <= m_test <= MAX_M:
                f_val = _compute_f(m_test)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m_test
    
    # Strategy 3: More targeted search
    # Try m values that are just below multiples of large numbers
    # e.g., m = k*12 - 1, so that 12 has to wait
    if time.time() - start_time < DEADLINE:
        # Try m = k*L - 1 for various L
        for k in range(0, MAX_M // 12 + 1, 50):
            if time.time() - start_time > DEADLINE:
                break
            m_test = k * 12 - 1
            if 0 <= m_test <= MAX_M:
                f_val = _compute_f(m_test)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m_test
    
    # Strategy 4: Try m values near LCM-related positions
    # LCM(1..12) = 27720
    # Try m = 27720*k + offset
    if time.time() - start_time < DEADLINE:
        lcm_12 = 27720
        for k in range(0, MAX_M // lcm_12 + 1):
            if time.time() - start_time > DEADLINE:
                break
            base = k * lcm_12
            for offset in range(0, min(200, MAX_M - base + 1), 5):
                if time.time() - start_time > DEADLINE:
                    break
                m_test = base + offset
                if 0 <= m_test <= MAX_M:
                    f_val = _compute_f(m_test)
                    if f_val > best_f:
                        best_f = f_val
                        best_m = m_test
    
    # Strategy 5: Fine-grained scan in promising ranges
    # Based on where we found high f, do finer scan
    if time.time() - start_time < DEADLINE:
        # Do a finer scan with step 10 in the range around best_m
        lo_scan = max(0, best_m - 500)
        hi_scan = min(MAX_M, best_m + 500)
        m = lo_scan
        while m <= hi_scan:
            if time.time() - start_time > DEADLINE:
                break
            f_val = _compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
            m += 10
    
    # Final: verify and return
    # Ensure best_m is a valid Python int
    best_m = int(best_m)
    if best_m < 0:
        best_m = 0
    if best_m > MAX_M:
        best_m = MAX_M
    
    return best_m

# EVOLVE-BLOCK-END
