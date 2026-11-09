import sys

P = 27720  # LCM(1..12)

def precompute_best():
    """
    Precompute f(12, m) for all m in [0, P) and return the best m.
    f(12, m) is the smallest L > 0 such that (m, m+L] contains distinct b_1..b_12 with i | b_i.
    We want to maximize this L.
    """
    
    # Precompute for each residue r in [0, P), which indices i (1..12) divide r.
    # We store this as a list of lists: divisors[r] = list of i such that i | r
    divisors = [[] for _ in range(P)]
    for r in range(P):
        for i in range(1, 13):
            if r % i == 0:
                divisors[r].append(i)
    
    best_m = 0
    best_L = 0
    
    # We iterate over all possible starting residues m
    for start in range(P):
        # Binary search for the minimum L such that a perfect matching exists
        # The interval is (start, start+L], i.e., numbers start+1, ..., start+L
        
        # Lower bound: 12 (at least 12 numbers needed)
        # Upper bound: The prompt says max is 144. Let's use 144.
        lo, hi = 12, 144
        min_L = hi + 1  # If no L in range works, it's invalid (shouldn't happen)
        
        while lo <= hi:
            mid = (lo + hi) // 2
            if has_matching(start, mid, divisors):
                min_L = mid
                hi = mid - 1
            else:
                lo = mid + 1
        
        # If a valid L was found
        if min_L <= 144:
            if min_L > best_L:
                best_L = min_L
                best_m = start
                if best_L == 144:
                    # Can't do better than 144
                    break
    
    return best_m, best_L

def has_matching(start, L, divisors):
    """
    Check if there is a perfect matching for interval (start, start+L].
    Left nodes: i in 1..12
    Right nodes: j in 1..L (representing number start+j)
    Edge (i, j) exists if i divides (start+j).
    """
    # Build adjacency list for left nodes (indices 0..11 representing i=1..12)
    # adj[i-1] will contain list of j (1..L) such that i | (start+j)
    adj = [[] for _ in range(12)]
    
    for j in range(1, L + 1):
        num = start + j
        r = num % P
        # divisors[r] contains i values
        for i in divisors[r]:
            # i is in 1..12
            adj[i-1].append(j)
    
    # Kuhn's algorithm for bipartite matching
    # match_to[j] = i (which left node is matched to right node j)
    # We use 1-based indexing for j to match the list construction, 
    # but arrays are 0-based, so match_to[j-1] = i
    
    match_to = [-1] * L
    
    def dfs(u, visited):
        # u is index in 0..11 (representing i=u+1)
        for v in adj[u]:
            # v is in 1..L
            if visited[v]:
                continue
            visited[v] = True
            # If v is not matched, or the node currently matched to v can find another match
            if match_to[v-1] == -1 or dfs(match_to[v-1], visited):
                match_to[v-1] = u
                return True
        return False
    
    match_count = 0
    for i in range(12):
        visited = [False] * (L + 1)
        if dfs(i, visited):
            match_count += 1
        else:
            # If any node fails to match, no perfect matching
            return False
            
    return match_count == 12

# Precompute at module load
_BEST_M, _BEST_L = precompute_best()

def construct():
    """
    Return the precomputed best m that maximizes f(12, m).
    """
    return int(_BEST_M)
