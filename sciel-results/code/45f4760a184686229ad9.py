import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    N = 12
    # LCM(1..12) = 27720
    PERIOD = 27720
    
    def has_matching(m, L):
        # m is the start, interval is (m, m+L]
        # Left nodes: 0..11 corresponding to i=1..12
        # Right nodes: values in (m, m+L]
        
        # We need to map right nodes to indices 0..L-1 for the matching algorithm
        # But the values themselves are just identifiers.
        # We can use a dictionary or just list of lists for adjacency.
        
        # adj[i] = list of right-node indices (or values) connected to i
        # Let's use values directly in the matching logic to avoid mapping overhead if L is small?
        # No, standard DFS matching uses indices.
        
        # Let's collect all unique values in the interval first?
        # No, a value v can be a candidate for multiple i's.
        # The bipartite graph is between I={1..12} and V={m+1..m+L}.
        # Edge (i, v) exists if i divides v.
        
        # To run standard bipartite matching, we need to index the right side.
        # Let's map value v to index v - (m+1).
        
        adj = [[] for _ in range(N)]
        
        for i in range(1, N + 1):
            # Find multiples of i in (m, m+L]
            # First multiple > m:
            first = ((m // i) + 1) * i
            if first > m + L:
                continue
            for v in range(first, m + L + 1, i):
                idx = v - (m + 1)
                adj[i - 1].append(idx)
                
        # Standard DFS Bipartite Matching
        match_to_left = [-1] * L  # match_to_right[j] = i
        
        def dfs(i, seen):
            for j in adj[i]:
                if seen[j]:
                    continue
                seen[j] = True
                if match_to_left[j] == -1 or dfs(match_to_left[j], seen):
                    match_to_left[j] = i
                    return True
            return False
        
        count = 0
        for i in range(N):
            seen = [False] * L
            if dfs(i, seen):
                count += 1
            else:
                return False
        return count == N

    best_L = 0
    best_m = 0
    
    for m in range(PERIOD):
        # Check time
        if time.time() > deadline:
            break
            
        # Find minimal L for this m
        # L is at least 1. Max L is likely small.
        # We can binary search or linear search.
        # Linear search from 1 upwards is fine since L is small.
        found = False
        for L in range(1, 200):  # 200 is a safe upper bound
            if has_matching(m, L):
                if L > best_L:
                    best_L = L
                    best_m = m
                found = True
                break
        # If no L found up to 200, it's unlikely but possible?
        # The problem states L is at most 144. So 200 is safe.
        
    return best_m
