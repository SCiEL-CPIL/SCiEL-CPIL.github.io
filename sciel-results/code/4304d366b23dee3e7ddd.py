import math

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    L = 30030 # LCM of divisors
    
    # We search for minimal W
    # Max W is bounded, let's try up to 1000
    for W in range(0, 1000):
        # Iterate start A in [1, L]
        # Since the pattern repeats every L, we only need to check one period.
        for A in range(1, L + 1):
            # Build adjacency list for bipartite matching
            # Left: divisors (0..n-1)
            # Right: offsets (0..W)
            
            adj = [[] for _ in range(n)]
            feasible = True
            
            for i in range(n):
                d = divisors[i]
                # Find smallest r >= 0 such that (A+r) % d == 0
                # (A+r) % d == 0  => r % d == (-A) % d
                target = (-A) % d
                if target > W:
                    # No candidates in [0, W]
                    feasible = False
                    break
                # Add all r = target, target+d, ... <= W
                adj[i].extend(range(target, W + 1, d))
                
                if not adj[i]:
                    feasible = False
                    break
            
            if not feasible:
                continue
                
            # Check if perfect matching exists
            match_right = [-1] * (W + 1)
            
            def dfs(i, visited):
                for j in adj[i]:
                    if visited[j]:
                        continue
                    visited[j] = True
                    if match_right[j] == -1 or dfs(match_right[j], visited):
                        match_right[j] = i
                        return True
                return False
            
            found = True
            for i in range(n):
                visited = [False] * (W + 1)
                if not dfs(i, visited):
                    found = False
                    break
                    
            if found:
                # Construct result
                result = [0] * n
                for j in range(W + 1):
                    if match_right[j] != -1:
                        i = match_right[j]
                        result[i] = A + j
                        
                return result
                
    # Fallback
    return divisors[:]
