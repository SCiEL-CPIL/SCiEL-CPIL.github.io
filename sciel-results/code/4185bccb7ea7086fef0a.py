"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from functools import lru_cache

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m), where f(12, m) is the
    least positive L such that (m, m+L] contains 12 distinct integers b_1..b_12
    with i | b_i.
    
    We use Hall's theorem to check feasibility efficiently, then perform a
    smart search over m.
    """
    n = 12
    max_m = 1000000
    
    # Precompute divisors structure for indices 1..12
    # For each index i (1..12), the numbers in (m, m+L] divisible by i are:
    # ceil((m+1)/i)*i, ceil((m+1)/i)*i + i, ... up to m+L
    
    # Hall's condition: for every subset S of {1,...,12}, 
    # |union_{i in S} {x in (m,m+L] : i|x}| >= |S|
    
    # Since 2^12 = 4096, we can precompute all subsets and their LCMs.
    # The union of multiples of i for i in S has density related to LCM(S).
    # Actually, the count of numbers in (m, m+L] divisible by at least one i in S
    # can be computed via inclusion-exclusion, but that's complex.
    
    # Alternative: for a given m and L, we can compute the set of available numbers
    # for each i, then check Hall's condition by checking all subsets.
    
    # But 4096 subsets per (m,L) check, and binary search over L (up to ~200),
    # and we need to search over m... this is expensive if done naively.
    
    # Key insight: f(12, m) is small (at most 144 per problem statement).
    # So L is at most 144. We can iterate L from 1 upward and check if
    # there exists a matching.
    
    # For a fixed m and L, the problem is: can we assign distinct b_i in (m, m+L]
    # such that i | b_i?
    
    # This is a bipartite matching problem. With 12 left nodes and at most 144 right nodes,
    # we can use a simple augmenting path algorithm.
    
    # Strategy:
    # 1. For each m in a smart subset of [0, 1000000], compute f(12, m).
    # 2. Return the m with maximum f(12, m).
    
    # How to search m efficiently?
    # f(12, m) depends on the distribution of multiples of 1..12 in (m, m+L].
    # The "worst" m would be one where the interval (m, m+L] has poor coverage.
    
    # Let's first understand: what is the maximum possible f(12, m)?
    # The problem says it's at most 144. Let's try to find m that achieves high values.
    
    # Approach: 
    # - For a given m, compute f(12, m) by binary search on L (1 to 144).
    # - For each L, check if a perfect matching exists using Hopcroft-Karp or simple BFS.
    # - Search m by sampling and local optimization.
    
    def can_match(m, L):
        """Check if there exist distinct b_1..b_12 in (m, m+L] with i|b_i."""
        if L <= 0:
            return False
        # Build adjacency: for each i in 1..12, find all b in (m, m+L] with i|b
        adj = {}
        for i in range(1, 13):
            adj[i] = []
            # First multiple of i greater than m
            start = (m // i + 1) * i
            for b in range(start, m + L + 1, i):
                adj[i].append(b)
        
        # Now check if perfect matching exists
        # Simple DFS-based bipartite matching
        # Left side: indices 1..12, Right side: the numbers
        # We'll use the standard augmenting path approach
        
        # Collect all right nodes
        right_nodes = set()
        for i in range(1, 13):
            for b in adj[i]:
                right_nodes.add(b)
        
        # If any i has no candidates, fail
        for i in range(1, 13):
            if len(adj[i]) == 0:
                return False
        
        # Match: match[i] = b means i is matched to b
        # For right nodes: rmatch[b] = i
        match = {}  # i -> b
        rmatch = {}  # b -> i
        
        def dfs(i, visited):
            for b in adj[i]:
                if b in visited:
                    continue
                visited.add(b)
                if b not in rmatch or dfs(rmatch[b], visited):
                    match[i] = b
                    rmatch[b] = i
                    return True
            return False
        
        for i in range(1, 13):
            visited = set()
            if not dfs(i, visited):
                return False
        
        return True
    
    def f_value(m):
        """Compute f(12, m): least positive L such that matching exists."""
        # Binary search on L from 1 to 144
        lo, hi = 1, 144
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        if can_match(m, lo):
            return lo
        return 145  # shouldn't happen per problem statement
    
    # Now search for m that maximizes f_value(m)
    # Strategy: 
    # 1. Sample a range of m values
    # 2. Use local search around promising candidates
    
    # First, let's sample m values
    # The behavior of f(12, m) is periodic-ish with period lcm(1..12) = 27720
    # So we can search within one period and that should be sufficient.
    
    # Let's search m in [0, 27720] first, then check if better values exist elsewhere.
    # Actually, since the pattern repeats every 27720, the max over [0, 1000000]
    # is the same as the max over [0, 27720).
    
    best_m = 0
    best_f = 0
    
    # Search in chunks to stay within time limits
    # 27720 values, each f_value does binary search (7-8 iterations) x can_match
    # can_match is O(12 * 144 * 12) worst case = ~20000 ops, so total ~ 27720 * 8 * 20000 = 4.4e9... too slow.
    
    # Need to optimize. Let's precompute more cleverly.
    
    # Alternative: for each L from 1 to 144, find the set of m where matching fails.
    # f(12, m) >= L+1 iff matching fails for L.
    
    # Let's just do a smarter search: sample m values and use local refinement.
    
    import random
    random.seed(42)
    
    # Sample some m values
    candidates = set()
    # Uniform samples
    for _ in range(500):
        candidates.add(random.randint(0, 27719))
    # Also try structured values
    for m in range(0, 27720, 100):
        candidates.add(m)
    for m in range(0, 27720, 10):
        candidates.add(m)
    
    # Evaluate candidates
    for m in sorted(candidates):
        f = f_value(m)
        if f > best_f:
            best_f = f
            best_m = m
    
    # Local search around best_m
    for _ in range(100):
        improved = False
        for delta in range(-20, 21):
            if delta == 0:
                continue
            m_new = (best_m + delta) % 27720
            f_new = f_value(m_new)
            if f_new > best_f:
                best_f = f_new
                best_m = m_new
                improved = True
                break
        if not improved:
            break
    
    # Also try a broader search in case the period assumption is off
    for _ in range(200):
        m_try = random.randint(0, max_m)
        f_try = f_value(m_try)
        if f_try > best_f:
            best_f = f_try
            best_m = m_try
    
    return int(best_m)

# EVOLVE-BLOCK-END
