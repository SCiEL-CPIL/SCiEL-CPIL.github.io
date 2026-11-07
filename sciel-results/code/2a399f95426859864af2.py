import sys

def min_span(m):
    # Find least L such that (m, m+L] has distinct b_i div by i
    # Max L is bounded, say 150
    for L in range(1, 151):
        # Build adjacency list: adj[i] contains k (offset) if (m+k) % i == 0
        # i is 0..11 corresponding to divisor i+1
        adj = [[] for _ in range(12)]
        for i in range(12):
            mod = i + 1
            # Optimization: start from first multiple
            # First k such that (m+k) % mod == 0
            # k % mod == (-m) % mod
            rem = (-m) % mod
            if rem > L:
                continue
            # k = rem, rem+mod, ... <= L
            # Note: k must be >= 1
            # If rem == 0, first k is mod (since k>=1 and k%mod==0 => k=mod, 2mod...)
            # Wait, if rem == 0, k can be mod, 2mod...
            # If rem > 0, k can be rem, rem+mod...
            # But k must be in [1, L].
            
            if rem == 0:
                start = mod
            else:
                start = rem
            
            if start <= L:
                # Append start, start+mod, ...
                for k in range(start, L + 1, mod):
                    adj[i].append(k)
        
        # Check if perfect matching exists
        # match[k] = i (0-indexed)
        match = [-1] * (L + 1)
        
        def dfs(i, visited):
            for k in adj[i]:
                if visited[k]:
                    continue
                visited[k] = 1
                if match[k] == -1 or dfs(match[k], visited):
                    match[k] = i
                    return True
            return False

        # To improve speed, match the most constrained i first?
        # Sort i by len(adj[i])
        indices = sorted(range(12), key=lambda x: len(adj[x]))
        
        for i in indices:
            visited = [0] * (L + 1)
            if not dfs(i, visited):
                break
        else:
            return L
            
    return 150 # Fallback

def construct():
    best_m = 0
    best_val = 0
    # The function f(12, m) is periodic with period lcm(1..12) = 27720
    # So we only need to check m in 0..27719
    for m in range(27720):
        val = min_span(m)
        if val > best_val:
            best_val = val
            best_m = m
    return best_m
