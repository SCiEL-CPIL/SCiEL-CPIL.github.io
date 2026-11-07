import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 10s margin
    
    # Precompute LCM of 1..12
    # LCM(1..12) = 27720
    period = 27720
    
    def has_matching(m, L):
        # Left: 1..12, Right: m+1 .. m+L
        # adj[i] = list of indices j (0-based in right set) such that (m+1+j) % i == 0
        # Actually, right node value is v = m + 1 + j. Condition: v % i == 0.
        # j ranges from 0 to L-1.
        
        # Build adjacency list for left nodes 0..11 (representing i=1..12)
        adj = [[] for _ in range(12)]
        for i in range(1, 13):
            # v must be in (m, m+L] -> m < v <= m+L
            # v = k*i. 
            # k*i > m => k > m/i => k >= floor(m/i) + 1
            # k*i <= m+L => k <= (m+L)/i => k <= floor((m+L)/i)
            start_k = (m // i) + 1
            end_k = (m + L) // i
            for k in range(start_k, end_k + 1):
                v = k * i
                if m < v <= m + L:
                    j = v - (m + 1) # index in right set 0..L-1
                    if 0 <= j < L:
                        adj[i - 1].append(j)
        
        # Bipartite matching using DFS (Kuhn's algorithm)
        match_to_right = [-1] * L # match_to_right[j] = i matched to j, or -1
        # We need to find if we can match all 12 left nodes.
        
        def dfs(i, visited):
            for j in adj[i]:
                if visited[j]:
                    continue
                visited[j] = True
                if match_to_right[j] == -1 or dfs(match_to_right[j], visited):
                    match_to_right[j] = i
                    return True
            return False
        
        matched_count = 0
        for i in range(12):
            visited = [False] * L
            if dfs(i, visited):
                matched_count += 1
            else:
                return False
        return matched_count == 12

    best_L = 0
    best_m = 0
    
    # Iterate through one period
    # We can optimize by checking L in increasing order for each m
    for m in range(period):
        if time.time() > deadline:
            break
            
        # Find the smallest L such that has_matching(m, L) is True
        # L is at least 12 (pigeonhole, need 12 distinct numbers)
        # Max L is 144 per contract
        found = False
        for L in range(12, 145):
            if has_matching(m, L):
                if L > best_L:
                    best_L = L
                    best_m = m
                found = True
                break
        
    return best_m
