import itertools

def min_span(m):
    # Find least L such that (m, m+L] has distinct b_i div by i
    # Max L is bounded, say 100
    for L in range(1, 101):
        # Candidates for each i
        # b_i in (m, m+L]
        # Check if matching exists
        # Left: 1..12
        # Right: m+1 .. m+L
        # Edge if i divides b
        
        # Simple backtracking or Hopcroft-Karp
        # Since N=12, simple DFS is enough
        used = [False] * (L + 1) # used offset
        match = {}
        
        def dfs(i, visited):
            for k in range(1, L + 1):
                b = m + k
                if i == 0: continue
                if b % i != 0: continue
                if visited[k]: continue
                visited[k] = True
                if k not in match or dfs(match[k], visited):
                    match[k] = i
                    return True
            return False
        
        # We need to match all 1..12
        # Wait, the standard matching algorithm matches Left to Right.
        # Here Left is 1..12.
        # We need to see if we can assign distinct k to each i.
        
        # Reset match for each L
        match = {} # k -> i
        
        def try_match(i, visited):
            for k in range(1, L + 1):
                b = m + k
                if b % i != 0: continue
                if visited[k]: continue
                visited[k] = True
                if k not in match or try_match(match[k], visited):
                    match[k] = i
                    return True
            return False
            
        # We need to match all i in 1..12
        # This is a bipartite matching where we want to cover all Left nodes.
        # Standard Kuhn's algorithm.
        
        # Note: The order of i might matter for efficiency, but 12 is small.
        # We must ensure we can match ALL i.
        
        # Let's just use a simple check:
        # Is there a system of distinct representatives?
        
        # We'll use the standard algorithm.
        # match[k] = i means k is assigned to i.
        # We iterate i from 1 to 12.
        
        # Clear match
        match = {}
        
        for i in range(1, 13):
            visited = [False] * (L + 1)
            if not try_match(i, visited):
                break
        else:
            # All matched
            return L
            
    return 100 # Fallback

def construct():
    best_m = 0
    best_val = 0
    # Search a subset
    for m in range(0, 200):
        val = min_span(m)
        if val > best_val:
            best_val = val
            best_m = m
    return best_m
