"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def _compute_f(m, n=12):
    """Compute f(n, m): least positive L such that (m, m+L] has distinct b_1..b_n with i|b_i."""
    for L in range(1, 200):
        # For each i, the candidates in (m, m+L] divisible by i
        # We need to check if we can assign distinct candidates
        # Use greedy: for each i, the earliest available candidate
        # But greedy might not work due to conflicts, so use matching
        
        # Build adjacency: for each i, which numbers in (m, m+L] are divisible by i
        # Numbers are m+1, m+2, ..., m+L
        # For matching, map each number to its index 0..L-1
        
        # adj[i] = set of indices j (0-based) such that (m+1+j) % i == 0
        adj = [set() for _ in range(n + 1)]
        for i in range(1, n + 1):
            # Find all j in 0..L-1 such that (m+1+j) % i == 0
            # (m+1+j) % i == 0 => j ≡ (-m-1) mod i
            rem = (-m - 1) % i
            # j = rem, rem+i, rem+2i, ... < L
            j = rem
            while j < L:
                adj[i].add(j)
                j += i
        
        # Bipartite matching: left side = indices 1..n, right side = positions 0..L-1
        match_right = [-1] * L  # match_right[j] = i that is matched to position j, or -1
        match_left = [-1] * (n + 1)  # match_left[i] = position j matched to i, or -1
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_right[j] == -1 or dfs(match_right[j], visited):
                    match_right[j] = i
                    match_left[i] = j
                    return True
            return False
        
        ok = True
        for i in range(1, n + 1):
            visited = [False] * L
            if not dfs(i, visited):
                ok = False
                break
        
        if ok:
            return L
    return 200

def construct():
    """Find m in [0, 1000000] that maximizes f(12, m)."""
    best_m = 12
    best_f = 0
    
    # Search over m values. Focus on ranges where f might be large.
    # f(12,m) is large when m is just below a number that is a multiple of many small i's
    # but the next multiples are spread out.
    
    # Let's do a structured search
    # First, check m values near multiples of LCM(1..12) = 27720
    # LCM(1..12) = 27720
    
    import math
    lcm_1_12 = 27720
    
    # Check m values around multiples of 27720
    candidates = set()
    for k in range(0, 1000000 // lcm_1_12 + 2):
        base = k * lcm_1_12
        for offset in range(-10, 11):
            m_val = base + offset
            if 0 <= m_val <= 1000000:
                candidates.add(m_val)
    
    # Also add some random samples
    import random
    random.seed(123)
    for _ in range(100000):
        candidates.add(random.randint(0, 1000000))
    
    # Also add small values
    for m_val in range(0, 5000):
        candidates.add(m_val)
    
    for m_val in candidates:
        f_val = _compute_f(m_val, 12)
        if f_val > best_f:
            best_f = f_val
            best_m = m_val
    
    return best_m

# EVOLVE-BLOCK-END
