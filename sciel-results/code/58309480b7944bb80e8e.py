import time
import sys

# EVOLVE-BLOCK-START

def construct():
    """
    Finds the integer m in [0, 1000000] that maximizes f(12, m).
    f(12, m) is the least positive L such that (m, m+L] contains distinct 
    b_1, ..., b_12 with i dividing b_i.
    """
    N = 12
    PERIOD = 27720  # LCM(1, ..., 12)
    
    # Deadline guard: 90s limit, leave 10s margin
    deadline = time.time() + 80.0
    
    def has_matching(m, L):
        """
        Checks if there is a perfect matching for indices 1..12 
        to distinct values in (m, m+L] where i divides value.
        """
        # candidates[i] will be a list of values in (m, m+L] divisible by i
        # We use 1-based indexing for i in 1..12, so candidates[1]..candidates[12]
        # Values are distinct across all i? No, values are distinct in the matching.
        # The values themselves are just integers.
        
        # To speed up DFS, we can represent the right side as a set of available values
        # and use a standard bipartite matching algorithm.
        # Left side: 0..11 (representing i=1..12)
        # Right side: values v in (m, m+L]
        
        # Precompute candidates for each i
        # Since L is small, we can just iterate through the range
        candidates = [[] for _ in range(N)]
        # Map from value to list of indices it satisfies? Or just check divisibility in DFS?
        # Better: for each i, find multiples in (m, m+L].
        
        for i in range(1, N + 1):
            # Smallest multiple of i greater than m
            # k = floor(m / i) + 1
            # val = k * i
            # while val <= m + L:
            #   candidates[i-1].append(val)
            #   val += i
            
            first_val = (m // i + 1) * i
            if first_val <= m + L:
                for v in range(first_val, m + L + 1, i):
                    candidates[i - 1].append(v)
        
        # Check if any candidate list is empty
        for i in range(N):
            if not candidates[i]:
                return False
        
        # Bipartite matching
        # match_to[j] = i (0-indexed) if value j is matched to index i
        # We need to map values to indices in the right side.
        # Let's collect all unique values in the window that are candidates for at least one i.
        # Actually, it's simpler to just use the values directly as the right node IDs.
        
        # Standard DFS augmenting path
        # match_r: dict mapping value -> index (1..12) it is matched to, or -1
        # We use a dictionary because values are large.
        match_r = {}
        
        def dfs(i, visited):
            # Try to find an augmenting path for index i (0-indexed)
            for v in candidates[i]:
                if v in visited:
                    continue
                visited.add(v)
                # If v is not matched, or if the index currently matched to v can find another value
                if v not in match_r or dfs(match_r[v], visited):
                    match_r[v] = i
                    return True
            return False
        
        # Run matching
        match_count = 0
        for i in range(N):
            visited = set()
            if dfs(i, visited):
                match_count += 1
            else:
                return False
        return match_count == N

    best_m = 0
    best_L = 0
    
    # Iterate over one full period
    # We can optimize by checking time periodically
    for m in range(PERIOD):
        if time.time() > deadline:
            break
        
        # Find min L for this m
        # Max L is likely small. Let's start from 1.
        # Upper bound: 144 is given as a safe upper bound in the prompt context.
        # Actually, worst case L is at most 12 * max(i) / something? 
        # The prompt says "at most 144".
        found = False
        for L in range(1, 145):
            if has_matching(m, L):
                if L > best_L:
                    best_L = L
                    best_m = m
                found = True
                break
        
        # If not found in 144, something is wrong, but problem says max 144.
        # We continue to next m.
        
    return int(best_m)

# EVOLVE-BLOCK-END
