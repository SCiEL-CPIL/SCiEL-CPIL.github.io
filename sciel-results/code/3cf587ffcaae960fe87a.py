import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin for safety
    
    N = 12
    MAX_M = 1000000
    MAX_L = 144
    
    # Precompute divisors or just use modulo
    # For speed, we can precompute for each i in 1..12, the list of numbers in a window
    # But since L is small, direct check is fast enough.
    
    def can_match(m, L):
        # Check if there is a matching for indices 1..N in (m, m+L]
        # Candidates are m+1, ..., m+L
        # We represent the right side as 0..L-1 corresponding to m+1..m+L
        
        # adj[i] is a list of candidate indices j (0-based) such that (m + j + 1) % i == 0
        adj = [[] for _ in range(N)]
        
        for i in range(1, N + 1):
            # We need to find j in 0..L-1 such that (m + j + 1) % i == 0
            # (m + 1) % i = r. We need j such that (r + j) % i == 0.
            # j = (i - r) % i, (i - r + i) % i, ...
            r = (m + 1) % i
            if r == 0:
                first_j = 0
            else:
                first_j = i - r
            
            for j in range(first_j, L, i):
                adj[i-1].append(j)
        
        # Bipartite matching using DFS (Kuhn's algorithm)
        # match_to[i] = candidate index j matched to index i
        match_to = [-1] * N
        # match_from[j] = index i matched to candidate j
        match_from = [-1] * L
        
        def dfs(u, visited):
            for v in adj[u]:
                if visited[v]:
                    continue
                visited[v] = True
                if match_from[v] == -1 or dfs(match_from[v], visited):
                    match_from[v] = u
                    match_to[u] = v
                    return True
            return False
        
        for u in range(N):
            visited = [False] * L
            if not dfs(u, visited):
                return False
        return True

    def min_L(m):
        # Binary search for the smallest L such that can_match(m, L) is True
        # L must be at least N (12)
        lo = N
        hi = MAX_L
        # If no match even at MAX_L, return MAX_L (or a large value)
        # But problem says max L is 144.
        
        # Quick check for hi
        if not can_match(m, hi):
            return hi # Should not happen if L is large enough, but safe fallback
            
        while lo < hi:
            mid = (lo + hi) // 2
            if can_match(m, mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    best_m = 0
    best_L = 0
    
    # Initial random samples to get a baseline
    for _ in range(100):
        m = random.randint(0, MAX_M)
        L = min_L(m)
        if L > best_L:
            best_L = L
            best_m = m

    # Local search / Hill climbing around best_m
    # Also continue random sampling
    # To maximize L, we want to find "holes" in the divisibility structure.
    
    # Use a set to avoid re-evaluating the same m
    evaluated = set()
    evaluated.add(best_m)
    
    # Strategy:
    # 1. Random sampling
    # 2. Neighbors of current best
    # 3. Neighbors of other good candidates (keep a list of top candidates)
    
    top_candidates = [(best_L, best_m)]
    
    while time.time() < deadline:
        # Determine next m to evaluate
        if random.random() < 0.3 and top_candidates:
            # Explore neighbors of a top candidate
            # Pick a random top candidate
            _, base_m = random.choice(top_candidates)
            # Random step size
            step = random.randint(-100, 100)
            if step == 0:
                step = 1
            m = base_m + step
        else:
            m = random.randint(0, MAX_M)
            
        # Bound m
        if m < 0:
            m = 0
        elif m > MAX_M:
            m = MAX_M
            
        if m in evaluated:
            continue
            
        evaluated.add(m)
        
        L = min_L(m)
        
        if L > best_L:
            best_L = L
            best_m = m
            top_candidates.append((L, m))
            # Keep only top 20 candidates to manage memory and focus search
            if len(top_candidates) > 20:
                top_candidates.sort(key=lambda x: x[0], reverse=True)
                top_candidates = top_candidates[:20]
        elif L > top_candidates[0][0] * 0.9 and len(top_candidates) < 20:
            # Keep some good candidates even if not the absolute best
            top_candidates.append((L, m))
            top_candidates.sort(key=lambda x: x[0], reverse=True)
            top_candidates = top_candidates[:20]

    return int(best_m)

# EVOLVE-BLOCK-END
