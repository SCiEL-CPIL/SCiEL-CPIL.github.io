"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    max_m = 1000000
    
    def can_assign(m, L):
        """Check if there's a perfect matching: assign distinct b_i in (m, m+L] with i | b_i."""
        # Build adjacency: for each index i (1..12), list of valid value-positions
        adj = [set() for _ in range(n)]
        for i in range(n):
            idx = i + 1  # actual index value
            first = ((m + 1 + idx - 1) // idx) * idx
            if first > m + L:
                adj[i] = set()
                continue
            for v in range(first, m + L + 1, idx):
                pos = v - (m + 1)
                if 0 <= pos < L:
                    adj[i].add(pos)
        
        # Hall's theorem pre-filter: for every subset S of indices, |N(S)| >= |S|
        # We check a few key subsets for speed
        # Check individual indices
        for i in range(n):
            if len(adj[i]) < 1:
                return False
        
        # Check pairs
        for i in range(n):
            for j in range(i+1, n):
                if len(adj[i] | adj[j]) < 2:
                    return False
        
        # Check triples (sample some)
        for i in range(0, n, 2):
            for j in range(i+1, n, 2):
                for k in range(j+1, n, 2):
                    union = adj[i] | adj[j] | adj[k]
                    if len(union) < 3:
                        return False
        
        # Now run bipartite matching (Kuhn's algorithm)
        # Use list version for speed
        adj_list = [list(s) for s in adj]
        
        match = [-1] * L
        
        def dfs(i, visited):
            for pos in adj_list[i]:
                if visited[pos]:
                    continue
                visited[pos] = True
                if match[pos] == -1 or dfs(match[pos], visited):
                    match[pos] = i
                    return True
            return False
        
        for i in range(n):
            visited = [False] * L
            if not dfs(i, visited):
                return False
        return True
    
    def f_of_m(m):
        """Find minimum L such that can_assign(m, L) is True."""
        lo, hi = 12, 144
        if not can_assign(m, hi):
            return -1
        while lo < hi:
            mid = (lo + hi) // 2
            if can_assign(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    best_m = 0
    best_f = -1
    
    # Generate targeted candidates
    # Key insight: to maximize f(12,m), we want m such that for small L,
    # the interval (m, m+L] doesn't have enough multiples of each i to form a matching.
    # This happens when m is chosen so that many multiples of small numbers are "blocked"
    
    candidates = set()
    
    # Strategy 1: m values that are 1 less than multiples of lcm(1..12)=27720
    # This makes the interval start just after a "complete" set of multiples
    for k in range(0, 37):  # 27720 * 37 ≈ 1025640
        candidates.add(27720 * k - 1)
        candidates.add(27720 * k)
        candidates.add(27720 * k + 1)
    
    # Strategy 2: m values that are 1 less than multiples of large primes in 1..12
    # Primes: 2,3,5,7,11
    for p in [2, 3, 5, 7, 11]:
        for k in range(0, max_m // p + 1, 100):
            candidates.add(p * k - 1)
    
    # Strategy 3: Random samples with bias toward values near multiples of 27720
    random.seed(42)
    for _ in range(3000):
        base = random.randint(0, 36) * 27720
        offset = random.randint(-100, 100)
        m = base + offset
        if 0 <= m <= max_m:
            candidates.add(m)
    
    # Strategy 4: Uniform random
    for _ in range(2000):
        candidates.add(random.randint(0, max_m))
    
    # Strategy 5: m values that make the first few multiples of small numbers fall at edges
    # For i=1: every number works
    # For i=2: even numbers
    # For i=3: multiples of 3
    # Try m where m+1 is odd (so first even is m+2), m+2 is not div by 3, etc.
    for m in range(0, max_m + 1, 720):  # lcm(1..12)/27720*720... let's just use 720
        candidates.add(m)
        candidates.add(m + 1)
        candidates.add(m - 1)
    
    # Sort candidates for deterministic ordering, but prioritize structured ones
    candidate_list = list(candidates)
    
    # Evaluate candidates
    for m in candidate_list:
        if time.time() - start_time > deadline:
            break
        if m < 0 or m > max_m:
            continue
        f_val = f_of_m(m)
        if f_val > best_f:
            best_f = f_val
            best_m = m
    
    return int(best_m)
# EVOLVE-BLOCK-END
