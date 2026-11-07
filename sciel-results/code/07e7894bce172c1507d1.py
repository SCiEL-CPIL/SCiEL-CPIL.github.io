"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys

P = {'n': 12, 'maximum_start': 1000000}

def construct():
    """
    Find m in [0, 1000000] that maximizes f(12, m), where f(n, m) is the
    least positive L such that (m, m+L] contains distinct b_1,...,b_n with
    i dividing b_i.
    
    We exploit the periodicity of the divisibility conditions.
    The pattern of divisibility for numbers in (m, m+L] repeats every 
    lcm(1, ..., 12) = 27720.
    Thus, f(m) depends only on m % 27720.
    We search for the worst m in [0, 27720).
    """
    n = 12
    max_L = 144
    
    # LCM of 1..12
    period = 27720
    
    def can_match(m, L):
        """
        Check if there is a perfect matching between indices 1..12 
        and distinct values in (m, m+L] such that i divides b_i.
        Uses Kuhn's algorithm (DFS-based bipartite matching).
        """
        # Left side: indices 1..12 (0..11)
        # Right side: values m+1 .. m+L (0..L-1)
        
        # Build adjacency list for left nodes
        # adj[i] = list of right nodes j such that (m+1+j) % (i+1) == 0
        adj = [[] for _ in range(n)]
        for i in range(n):
            d = i + 1
            # We need j such that m + 1 + j is divisible by d
            # m + 1 + j = k * d  => j = k * d - m - 1
            # j >= 0 => k * d >= m + 1 => k >= ceil((m+1)/d)
            # j < L => k * d - m - 1 < L => k * d < m + 1 + L => k < (m+1+L)/d
            # Alternatively, use modulo arithmetic:
            # j ≡ -m - 1 (mod d)
            r = (-m - 1) % d
            j = r
            while j < L:
                adj[i].append(j)
                j += d
        
        match_to_right = [-1] * L
        
        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_to_right[v] == -1 or dfs(match_to_right[v], visited):
                    match_to_right[v] = u
                    return True
            return False
        
        matching_size = 0
        for u in range(n):
            visited = [False] * L
            if dfs(u, visited):
                matching_size += 1
        
        return matching_size == n

    best_m = 0
    best_L = 0
    
    # Search all residues modulo period
    for m in range(period):
        # Find minimal L such that can_match(m, L) is True
        # Binary search on L in [1, max_L]
        lo, hi = 1, max_L
        result_L = max_L + 1
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                result_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        if result_L <= max_L and result_L > best_L:
            best_L = result_L
            best_m = m
            
    return int(best_m)

# EVOLVE-BLOCK-END
