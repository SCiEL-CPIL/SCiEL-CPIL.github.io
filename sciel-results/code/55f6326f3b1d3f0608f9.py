"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import lru_cache

P = {'n': 12, 'maximum_start': 1000000}
N = 12
MAX_M = 1000000
LCM_1_12 = 27720  # lcm(1,2,...,12)

def _has_matching(m, L):
    """Check if (m, m+L] contains distinct b_1..b_12 with i | b_i."""
    # For each i in 1..12, candidate b values in (m, m+L] divisible by i
    # Build bipartite matching: left = indices 1..12, right = candidate numbers
    # Use DFS-based bipartite matching (Hopcroft-Karp would be better but 12 is small)
    
    # Candidate numbers in (m, m+L]
    lo = m + 1
    hi = m + L
    
    # For each i, find all multiples of i in (m, m+L]
    # We need to assign distinct numbers to each i
    
    # Approach: build adjacency list for each i (left node) to candidate numbers (right nodes)
    # But candidate numbers can be up to L distinct values, and we only need 12 distinct ones.
    
    # Better approach: for each i, the candidates are multiples of i in (m, m+L].
    # We do a bipartite matching where left side is {1,...,12} and right side is the set of all
    # candidate numbers that appear in at least one i's candidate set.
    
    # Actually, simpler: use backtracking/DFS matching.
    # For each i, candidates are: first_mult = ((m // i) + 1) * i, then step by i up to m+L.
    
    # Let's use a standard bipartite matching (Kuhn's algorithm) with left = {1..12}
    # Right nodes are the candidate numbers. But there could be many candidates.
    # Instead, let's think of it differently:
    
    # For a matching to exist, we need to pick b_i for each i such that:
    # - b_i is in (m, m+L]
    # - i divides b_i
    # - all b_i are distinct
    
    # Use DFS matching. Left nodes: 1..12. For each left node i, its neighbors are
    # multiples of i in (m, m+L]. Right nodes: the actual numbers.
    
    # To keep it efficient, note that each i has at most ceil(L/i) candidates.
    # Total candidates across all i is at most sum(ceil(L/i)) <= L * H_12 ~ L * 3.1
    
    # Let's collect all candidate numbers and build the matching.
    
    # Map: for each number x in (m, m+L], which i's can use it? i | x and i <= 12.
    # Actually, we want: for each i, which x's can be assigned? x in (m,m+L], i|x.
    
    # Let's do Kuhn's algorithm.
    # adj[i] = list of candidate x values for i (multiples of i in (m, m+L])
    
    adj = [None] * (N + 1)
    all_candidates = set()
    for i in range(1, N + 1):
        # First multiple of i > m
        first = (m // i + 1) * i
        if first <= m:  # shouldn't happen but safety
            first += i
        candidates = []
        x = first
        while x <= hi:
            candidates.append(x)
            all_candidates.add(x)
            x += i
        adj[i] = candidates
    
    if not all_candidates:
        return False
    
    # Now do bipartite matching: left = {1..12}, right = all_candidates
    # match_right: maps a candidate number to the left node (i) it's matched to, or None
    # We use Kuhn's DFS
    
    # To speed up, limit the number of candidates per i to something reasonable
    # If L is large, there are many candidates. But we only need 12 matches.
    # Let's cap: for each i, if there are more than 50 candidates, take the first 50 and last 50.
    # Actually, let's just try all but with a time limit on the DFS.
    
    # Standard Kuhn's algorithm
    match_to_left = {}  # x -> i (which left node is matched to x)
    
    def try_kuhn(i, visited):
        for x in adj[i]:
            if x in visited:
                continue
            visited.add(x)
            if x not in match_to_left or try_kuhn(match_to_left[x], visited):
                match_to_left[x] = i
                return True
        return False
    
    for i in range(1, N + 1):
        visited = set()
        if not try_kuhn(i, visited):
            return False
    
    return True

def _compute_f(m):
    """Compute f(12, m): least positive L such that (m, m+L] has distinct b_i with i|b_i."""
    # Binary search on L. f(12,m) is at most 144 per the problem statement.
    # Actually let's just iterate L from 1 upward since max is 144.
    for L in range(1, 145):
        if _has_matching(m, L):
            return L
    return 144  # should not reach here

def construct():
    """Search for m in [0, 1000000] that maximizes f(12, m)."""
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin (evaluator kills at 90s)
    
    best_m = 12
    best_f = _compute_f(12)
    
    # Strategy: search structured m values
    # 1. Small m values (0 to ~200)
    # 2. Values near multiples of LCM_1_12 = 27720
    # 3. Random sampling
    
    # Phase 1: small m
    for m in range(0, 200):
        if time.time() > deadline:
            break
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    # Phase 2: near multiples of LCM
    # m = k * 27720 + offset, for various k and offsets
    # The idea: when m is a multiple of LCM, the pattern of divisibility repeats.
    # Near boundaries, we might get larger gaps.
    
    num_multiples = MAX_M // LCM_1_12  # ~36
    offsets_to_try = list(range(0, 200)) + list(range(27720 - 200, 27720))
    
    for k in range(num_multiples + 1):
        base = k * LCM_1_12
        for off in offsets_to_try:
            m = base + off
            if m > MAX_M:
                break
            if time.time() > deadline:
                break
            f_val = _compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
    
    # Phase 3: broader scan around multiples
    for k in range(num_multiples + 1):
        base = k * LCM_1_12
        # Scan a wider window
        for off in range(0, min(500, LCM_1_12)):
            m = base + off
            if m > MAX_M:
                break
            if time.time() > deadline:
                break
            f_val = _compute_f(m)
            if f_val > best_f:
                best_f = f_val
                best_m = m
    
    # Phase 4: random sampling
    import random
    rng = random.Random(42)
    attempts = 0
    while time.time() < deadline - 5:
        m = rng.randint(0, MAX_M)
        f_val = _compute_f(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
        attempts += 1
        if attempts > 5000:  # cap to avoid infinite loop
            break
    
    # Phase 5: local refinement around best_m
    if best_f >= 10:
        for dm in range(-50, 51):
            m = best_m + dm
            if 0 <= m <= MAX_M:
                if time.time() > deadline:
                    break
                f_val = _compute_f(m)
                if f_val > best_f:
                    best_f = f_val
                    best_m = m
    
    return int(best_m)

# EVOLVE-BLOCK-END
