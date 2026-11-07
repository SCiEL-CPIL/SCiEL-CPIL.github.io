import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 60  # 60s limit, keep margin
    
    n = 12
    mod = 1
    for i in range(1, n + 1):
        mod = mod * i // __import__('math').gcd(mod, i)
    # mod is 27720
    
    best_m = 0
    best_L = 0
    
    # We can just iterate through all residues
    # Since 27720 is small, we do all of them.
    for m in range(mod):
        if time.time() > deadline - 5:
            break
            
        # Find min L for this m
        L = 1
        while True:
            # Check if matching exists for interval (m, m+L]
            # Left nodes: 1..12
            # Right nodes: 1..L (representing offsets k)
            # Edge (i, k) exists if (m + k) % i == 0
            
            # Build adjacency list for left nodes
            adj = [[] for _ in range(n)]
            for i in range(1, n + 1):
                for k in range(1, L + 1):
                    if (m + k) % i == 0:
                        adj[i - 1].append(k - 1) # 0-indexed right nodes
            
            # Bipartite matching
            # matchR[j] = i (1-indexed left) if right node j is matched to left i
            # -1 if not matched
            matchR = [-1] * L
            
            def dfs(u, seen):
                for v in adj[u]:
                    if seen[v]:
                        continue
                    seen[v] = True
                    if matchR[v] == -1 or dfs(matchR[v], seen):
                        matchR[v] = u + 1
                        return True
                return False
            
            matching_count = 0
            for u in range(n):
                seen = [False] * L
                if dfs(u, seen):
                    matching_count += 1
            
            if matching_count == n:
                # Found min L
                if L > best_L:
                    best_L = L
                    best_m = m
                break
            
            L += 1
            # Safety: L won't be huge. If L > 100, something is wrong or very rare.
            if L > 200:
                break
                
    return int(best_m)
