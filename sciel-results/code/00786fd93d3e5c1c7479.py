import sys

def construct():
    n = 12
    lcm_period = 27720
    
    def can_match(m, L):
        # Candidates are m+1, ..., m+L
        # We need to assign distinct candidates to i=1..n such that i divides candidate.
        # Bipartite matching: Left {0..n-1}, Right {0..L-1}
        # Edge (i, j) exists if (m+1+j) % (i+1) == 0
        
        match_to = [-1] * L  # match_to[j] = i
        match_from = [-1] * n # match_from[i] = j
        
        # DFS for bipartite matching
        # To avoid recursion limit issues or overhead, we can use iterative DFS or just standard recursive since depth is small (max 12)
        
        def try_match(i, visited):
            # i is 0-indexed, represents divisor i+1
            divisor = i + 1
            for j in range(L):
                if visited[j]:
                    continue
                if (m + 1 + j) % divisor != 0:
                    continue
                visited[j] = True
                if match_to[j] == -1 or try_match(match_to[j], visited):
                    match_to[j] = i
                    match_from[i] = j
                    return True
            return False
        
        found = 0
        for i in range(n):
            visited = [False] * L
            if try_match(i, visited):
                found += 1
        
        return found == n

    best_m = 0
    best_L = 0
    
    # We only need to check m in 0..lcm_period-1
    for m in range(lcm_period):
        # Binary search for the minimum L such that can_match(m, L) is True
        # L is in [1, 144]
        lo, hi = 1, 144
        min_L = 144 # Default to max if not found (shouldn't happen)
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                min_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        # min_L is f(12, m)
        if min_L > best_L:
            best_L = min_L
            best_m = m
            
    return best_m
