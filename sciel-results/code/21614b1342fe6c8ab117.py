"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import defaultdict

DEADLINE = 85.0  # safety margin before 90s limit

def _can_match(m, L):
    """Check if interval (m, m+L] has distinct b_1..b_12 with i|b_i."""
    # Build candidate lists for each i
    candidates = {}
    for i in range(1, 13):
        # Multiples of i in (m, m+L]
        first = ((m // i) + 1) * i  # smallest multiple of i > m
        if first > m + L:
            return False  # no candidates for i
        cands = []
        val = first
        while val <= m + L:
            cands.append(val)
            val += i
        candidates[i] = cands
    
    # Bipartite matching: left = i (1..12), right = actual values
    # Use DFS-based matching
    # Map each candidate value to an index for the right side
    # Since values can repeat across different i's, we need to handle that
    
    # Build adjacency: for each i, list of candidate values
    # We'll do matching on the fly
    
    match_to_val = {}  # value -> i (which i is matched to this value)
    
    def dfs(i, visited):
        for val in candidates[i]:
            if val in visited:
                continue
            visited.add(val)
            if val not in match_to_val or dfs(match_to_val[val], visited):
                match_to_val[val] = i
                return True
        return False
    
    for i in range(1, 13):
        visited = set()
        if not dfs(i, visited):
            return False
    return True

def _f_val(m):
    """Compute f(12, m): least positive L such that (m, m+L] works."""
    # Binary search on L, range [1, 144]
    lo, hi = 1, 144
    while lo < hi:
        mid = (lo + hi) // 2
        if _can_match(m, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

def construct():
    start_time = time.time()
    best_m = 12
    best_f = _f_val(12)
    
    # Check a range of promising m values
    # Focus on m that are multiples of LCM of large numbers, or near them
    # Also check m = 0, and various multiples
    
    # First, check some structured candidates
    candidates_to_check = set()
    
    # Multiples of LCM(8,9,10,11,12) = 3960
    for k in range(0, 255):  # 3960*254 = 1005840, within range
        m_val = 3960 * k
        if 0 <= m_val <= 1000000:
            candidates_to_check.add(m_val)
    
    # Multiples of 12
    for k in range(0, 83334):
        m_val = 12 * k
        if 0 <= m_val <= 1000000:
            candidates_to_check.add(m_val)
    
    # Some random-ish samples
    import random
    random.seed(42)
    for _ in range(20000):
        m_val = random.randint(0, 1000000)
        candidates_to_check.add(m_val)
    
    # Sort for deterministic order
    candidates_to_check = sorted(candidates_to_check)
    
    for m in candidates_to_check:
        if time.time() > DEADLINE:
            break
        f = _f_val(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
