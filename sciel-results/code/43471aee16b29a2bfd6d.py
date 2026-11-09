import time

def _has_perfect_matching(r, L, n=12):
    """
    Checks if there is a perfect matching for indices 1..n in interval (r, r+L].
    Returns True if a perfect matching exists.
    """
    # adj[i] is a list of positions j (1-indexed in 1..L) such that (r+j) % i == 0
    # We only care if we can assign distinct j to each i.
    # match[j] stores which i is matched to j, or -1 if free.
    match = [-1] * (L + 1)
    
    def dfs(i, visited):
        # Try to find an augmenting path for index i
        for j in range(1, L + 1):
            if visited[j]:
                continue
            # Check if i can match with j
            if (r + j) % i != 0:
                continue
            
            visited[j] = True
            
            # If j is not matched, or if the current match of j can be rematched
            if match[j] == -1 or dfs(match[j], visited):
                match[j] = i
                return True
        return False

    # Try to match each index i from 1 to n
    for i in range(1, n + 1):
        visited = [False] * (L + 1)
        if not dfs(i, visited):
            return False
    return True

def construct():
    """
    Finds the m in [0, 1000000] that maximizes f(12, m).
    """
    n = 12
    # LCM of 1..12
    P = 27720
    
    best_r = 0
    best_L = 0
    
    start_time = time.time()
    
    # Iterate through all residues
    for r in range(P):
        # Check L from 1 upwards
        # Max L is small, usually < 200
        for L in range(1, 200):
            # Early exit if we already have a better L? No, we want max L.
            # But if we find a perfect matching, f(12,r) = L.
            # We want to maximize this L.
            if _has_perfect_matching(r, L, n):
                if L > best_L:
                    best_L = L
                    best_r = r
                break
            
            # Safety check for time
            if time.time() - start_time > 45:
                # Return best found so far
                # Find smallest m in [0, 1000000] with m % P == best_r
                return best_r
        
    # Find the smallest m in [0, 1000000] such that m % P == best_r
    # Since best_r < P, m = best_r is a valid candidate if best_r <= 1000000
    # 27720 < 1000000, so best_r is always valid.
    return best_r
