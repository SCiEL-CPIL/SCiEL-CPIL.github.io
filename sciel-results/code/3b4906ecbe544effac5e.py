import math
import time

def construct():
    """
    Finds m in [0, 1000000] maximizing f(12, m).
    f(12, m) is the least L such that (m, m+L] contains distinct b_1..b_12 with i|b_i.
    """
    start_time = time.time()
    deadline = start_time + 80  # Safety margin
    
    n = 12
    # LCM of 1..12
    K = 1
    for i in range(1, n + 1):
        K = K * i // math.gcd(K, i)
    
    best_m = 0
    best_f = 0
    
    # We know f(12, m) is periodic with period K
    # So we only need to check m in [0, K-1]
    # But we must return an m in [0, 1000000]
    # Any m congruent to best_m mod K works.
    # We will pick the smallest such m >= 0.
    
    for m in range(K):
        if time.time() > deadline:
            break
            
        # Find least L for this m
        # L is at most 144 according to task description, but let's allow a bit more just in case
        # Actually, the problem states "raw value ... is at most 144".
        # So we can cap L at 144. If no match found by 144, f is effectively > 144, 
        # but the task implies it's always found within 144? 
        # "The raw value is f(12, m), which is at most 144."
        # This suggests for all m, f(12, m) <= 144.
        
        found = False
        for L in range(1, 145):
            if _can_match(m, L, n):
                if L > best_f:
                    best_f = L
                    best_m = m
                found = True
                break
            # Optional: if we know an upper bound, we could stop early if L is too small?
            # No, we need the least L.
        
        # If not found by 144, it means f > 144, which shouldn't happen per contract.
        # But if it does, we ignore it or handle it.

    return best_m

def _can_match(m, L, n):
    """
    Checks if there is a perfect matching for indices 1..n
    in the interval (m, m+L].
    """
    # Precompute available numbers for each index i
    # adj[i] is a list of numbers in (m, m+L] divisible by i
    # We use 0-based indexing for i (0 represents index 1)
    adj = [[] for _ in range(n)]
    
    # Iterate through numbers in the interval
    # num = m + 1, ..., m + L
    for j in range(1, L + 1):
        num = m + j
        for i in range(1, n + 1):
            if num % i == 0:
                adj[i - 1].append(num)
                
    # Now we have a bipartite graph.
    # Left side: 0..n-1
    # Right side: numbers (but we don't care about their identity, just that they are distinct)
    # Actually, standard matching requires distinct right nodes.
    # We can use a simple DFS based matching (Kuhn's algorithm)
    # match_to[i] = which number is assigned to index i? No, match_to is usually for right nodes.
    # Let's use match[i] = number assigned to index i.
    # But a number can only be assigned to one index.
    # So we need to track which numbers are used.
    
    # Since L is small, we can map numbers to indices in a set or just use the value.
    # Let's use a dictionary or list for used numbers.
    # Better: Hopcroft-Karp is overkill. Simple DFS backtracking on indices.
    
    used = [False] * (L + 1) # used[j] is True if m+j is used. j is 1..L
    
    def dfs(i, visited):
        # Try to match index i (0-based)
        if i == n:
            return True
        for num in adj[i]:
            j = num - m # offset 1..L
            if not used[j] and not visited[j]:
                visited[j] = True
                used[j] = True
                if dfs(i + 1, visited):
                    return True
                used[j] = False
        return False

    # Optimization: Order indices by number of choices (most constrained first)
    # This helps pruning.
    indices = list(range(n))
    indices.sort(key=lambda x: len(adj[x]))
    
    # We need to permute adj to match the sorted indices? 
    # Or just pass the index in dfs.
    # Let's modify dfs to take the current index in the sorted order.
    
    def dfs_opt(idx, visited):
        if idx == n:
            return True
        i = indices[idx]
        for num in adj[i]:
            j = num - m
            if not used[j] and not visited[j]:
                visited[j] = True
                used[j] = True
                if dfs_opt(idx + 1, visited):
                    return True
                used[j] = False
        return False

    # Note: visited array is reset for each top-level call? 
    # In Kuhn's algorithm, we try to find an augmenting path.
    # Here, we are doing a simple backtracking search for a perfect matching.
    # The `visited` array prevents trying the same number multiple times in the current path?
    # Actually, in this specific formulation (assigning distinct numbers), 
    # `used` is global state for the current recursive branch.
    # `visited` is not strictly necessary if we just iterate through candidates, 
    # but it can prevent re-checking if we were doing standard bipartite matching with re-matching.
    # Here we are just picking a subset. So `used` is sufficient.
    
    # Let's simplify: just use `used`.
    def dfs_simple(idx):
        if idx == n:
            return True
        i = indices[idx]
        for num in adj[i]:
            j = num - m
            if not used[j]:
                used[j] = True
                if dfs_simple(idx + 1):
                    return True
                used[j] = False
        return False

    return dfs_simple(0)
