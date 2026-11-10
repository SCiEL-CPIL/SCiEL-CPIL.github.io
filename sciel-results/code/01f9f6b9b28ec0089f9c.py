"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s
    
    n = 12
    LCM = 27720  # lcm(1..12)
    
    # Precompute all subsets of {0,1,...,11} and for each subset S,
    # compute the set of divisors (i+1) for i in S
    # For Hall's theorem: for every subset S of indices,
    # |{v in (m, m+L] : v divisible by at least one (i+1) for i in S}| >= |S|
    
    # We'll precompute for each subset mask (12 bits), the list of divisors
    # and the size of the subset
    subset_info = []  # (mask, size, divisors_list)
    for mask in range(1 << n):
        divisors = []
        size = 0
        for i in range(n):
            if mask & (1 << i):
                divisors.append(i + 1)
                size += 1
        subset_info.append((mask, size, divisors))
    
    def can_assign_hall(m, L):
        """
        Check Hall's condition: for every subset S of {0..11},
        the number of integers in (m, m+L] divisible by at least one
        element of {i+1 : i in S} is >= |S|.
        
        This is necessary and sufficient for the existence of a matching
        (by Hall's marriage theorem) since we're looking at a bipartite
        graph where left side is indices and right side is positions.
        
        Actually, Hall's condition is necessary and sufficient for the
        existence of a system of distinct representatives, which is exactly
        what we need.
        """
        # For efficiency, we only need to check "critical" subsets.
        # But let's first try all 4096 subsets and see if it's fast enough.
        
        for mask, size, divisors in subset_info:
            if size == 0:
                continue
            # Count integers in (m, m+L] divisible by at least one divisor in divisors
            # Use inclusion-exclusion or just check each integer
            # For small L (up to 144), just check each integer
            count = 0
            for pos in range(L):
                v = m + 1 + pos
                for d in divisors:
                    if v % d == 0:
                        count += 1
                        break
            if count < size:
                return False
        return True
    
    def can_assign_matching(m, L):
        """
        Direct bipartite matching check using DFS (Kuhn's algorithm).
        This is the exact check.
        """
        adj = [[] for _ in range(n)]
        for i in range(n):
            idx = i + 1
            first = ((m + 1 + idx - 1) // idx) * idx
            if first > m + L:
                return False
            for v in range(first, m + L + 1, idx):
                pos = v - (m + 1)
                if 0 <= pos < L:
                    adj[i].append(pos)
        
        match = [-1] * L  # match[pos] = index assigned to pos
        
        def dfs(i, visited):
            for pos in adj[i]:
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
        m_mod = m % LCM
        lo, hi = 12, 144
        if not can_assign_matching(m_mod, hi):
            return -1
        while lo < hi:
            mid = (lo + hi) // 2
            if can_assign_matching(m_mod, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo
    
    # Strategy: Use Hall's theorem as a fast pre-filter to identify candidate
    # residues with high f values, then verify with exact matching.
    # 
    # Key insight: f(12,m) depends only on m mod 27720.
    # We can do a two-phase search:
    # Phase 1: For each residue, use Hall's condition to find the minimum L
    #           that satisfies Hall's condition (this gives a lower bound on f).
    #           Track residues with high lower bounds.
    # Phase 2: For the top candidates, compute exact f using matching.
    
    best_f = -1
    best_m = 0
    
    # Phase 1: For each residue, find the minimum L satisfying Hall's condition
    # This is a lower bound on the true f value.
    hall_lower_bounds = []
    for m_mod in range(LCM):
        if time.time() - start_time > deadline - 10:
            break
        
        # Binary search for minimum L satisfying Hall's condition
        lo, hi = 12, 144
        if not can_assign_hall(m_mod, hi):
            hall_lower_bounds.append((m_mod, -1))
            continue
        while lo < hi:
            mid = (lo + hi) // 2
            if can_assign_hall(m_mod, mid):
                hi = mid
            else:
                lo = mid + 1
        hall_lower_bounds.append((m_mod, lo))
    
    # Sort by lower bound descending
    hall_lower_bounds.sort(key=lambda x: x[1], reverse=True)
    
    # Phase 2: Verify top candidates with exact matching
    # Check top 5000 candidates (or as many as time allows)
    verified = set()
    for m_mod, lower_bound in hall_lower_bounds:
        if time.time() - start_time > deadline:
            break
        if lower_bound <= best_f:
            break  # No point checking further since lower bound is already too low
        if m_mod in verified:
            continue
        verified.add(m_mod)
        
        f_val = f_of_m(m_mod)
        if f_val > best_f:
            best_f = f_val
            best_m = m_mod
    
    # If we haven't found anything good, try a broader search
    if best_f < 13:
        # Try all residues with exact matching, but with early termination
        for m_mod in range(LCM):
            if time.time() - start_time > deadline:
                break
            f_val = f_of_m(m_mod)
            if f_val > best_f:
                best_f = f_val
                best_m = m_mod
    
    return int(best_m)
# EVOLVE-BLOCK-END
