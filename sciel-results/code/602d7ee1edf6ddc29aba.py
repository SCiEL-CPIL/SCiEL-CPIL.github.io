import time
import sys

# Increase recursion limit globally for safety
sys.setrecursionlimit(10000)

# LCM(1..12)
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _lcm(a, b):
    return a * b // _gcd(a, b)

PERIOD = 27720
N = 12
MAX_L = 144

# Precompute change points for each residue and i
# change_points[i][r] = list of L values (1..MAX_L) where a multiple of i enters (m, m+L]
# given m % i == r
_change_cache = {}

def get_changes(m, i):
    key = (m % i, i)
    if key not in _change_cache:
        # First multiple of i greater than m
        # m % i == r
        # If r == 0, next multiple is m + i (L=i)
        # If r > 0, next multiple is m + (i - r) (L=i-r)
        r = m % i
        if r == 0:
            first_l = i
        else:
            first_l = i - r
        
        points = []
        l = first_l
        while l <= MAX_L:
            points.append(l)
            l += i
        _change_cache[key] = points
    return _change_cache[key]

def has_perfect_matching(adj, left_size, right_size):
    """Kuhn's algorithm. adj[u] = list of right-node indices. Returns bool."""
    match_to_right = [-1] * right_size

    def dfs(u, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_to_right[v] == -1 or dfs(match_to_right[v], seen):
                match_to_right[v] = u
                return True
        return False

    for u in range(left_size):
        seen = [False] * right_size
        if not dfs(u, seen):
            return False
    return True

def can_match(m, L):
    """Return True if (m, m+L] contains 12 distinct b_i with i | b_i."""
    right_size = L
    adj = [[] for _ in range(N)]
    for i in range(1, N + 1):
        for Lc in get_changes(m, i):
            if Lc <= L:
                # Slot index is Lc - 1
                adj[i - 1].append(Lc - 1)
    return has_perfect_matching(adj, N, right_size)

def f_of_m(m):
    """Least positive L such that can_match(m, L)."""
    lo = 0
    for i in range(1, N + 1):
        d = ((m // i) + 1) * i - m
        if d > lo:
            lo = d
    lo = max(lo, 1)
    hi = MAX_L + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start = time.time()
    deadline = 55.0  # safety margin before 60s kill

    best_m = 0
    best_L = -1

    for m in range(PERIOD):
        if time.time() - start > deadline:
            break
        L = f_of_m(m)
        if L > best_L:
            best_L = L
            best_m = m

    return int(best_m)

# EVOLVE-BLOCK-START
# Entry point: construct() -> int in [0, 1000000]
# EVOLVE-BLOCK-END
