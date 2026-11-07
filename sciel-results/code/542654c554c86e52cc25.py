"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 12, 'maximum_start': 1000000}
L = 27720  # LCM(1..12)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Precompute: for each i in 1..12, the set of residues r in [0, L)
    # such that (m + r) is divisible by i, i.e., m + r ≡ 0 (mod i)
    # i.e., r ≡ -m (mod i). But we want, for a given m, which positions r
    # in a window (m, m+L] are divisible by i.
    # Position p = m + r where r is 1..L (since window is (m, m+L], i.e., m+1 to m+L)
    # p is divisible by i iff m + r ≡ 0 (mod i) iff r ≡ -m (mod i).
    
    # For a given m, for each i, the valid residues r in 1..L such that r ≡ -m (mod i).
    # We need to assign distinct b_i = m + r_i where r_i is valid for i, and all r_i distinct.
    # This is a bipartite matching: left = {1..12}, right = {1..L}, edge if r ≡ -m (mod i).
    
    # Key insight: the structure depends only on m mod L (since L = LCM(1..12)).
    # So we can precompute f for all m mod L, then find the max.
    
    # For each residue class m0 = m mod L (0 <= m0 < L), compute the maximum L_window
    # such that there exist distinct r_1,...,r_12 in 1..L_window with r_i ≡ -m0 (mod i).
    
    # Actually, let's think more carefully. The window is (m, m+L], so positions are m+1, ..., m+L.
    # Let r = p - m, so r ranges from 1 to L.
    # b_i = m + r_i where r_i must satisfy r_i ≡ -m (mod i).
    # Since m ≡ m0 (mod L) and L is a multiple of i, m ≡ m0 (mod i).
    # So r_i ≡ -m0 (mod i).
    # The valid r values for dimension i are those r in 1..L with r ≡ -m0 (mod i).
    # There are exactly L/i such values.
    
    # We need to find the smallest W such that in {1, ..., W}, we can pick distinct
    # r_1, ..., r_12 with r_i ≡ -m0 (mod i) for each i.
    
    # This is a bipartite matching problem. For each m0, we can compute the minimal W.
    
    # Let's precompute for all m0 in 0..L-1.
    
    # For a given m0, for each i, the set of valid r in 1..L is:
    #   V_i(m0) = {r in 1..L : r ≡ -m0 (mod i)}
    # We want the smallest W such that there's a system of distinct representatives
    # from V_1(m0) ∩ {1..W}, ..., V_12(m0) ∩ {1..W}.
    
    # We can use Hall's theorem or just do bipartite matching with binary search on W.
    # But L is only 27720, and we have 12 left nodes. So for each m0, we can try W = 1, 2, ...
    # and check if matching exists. Or binary search.
    
    # Actually, since L is small (27720) and we have 12 items, we can use a simpler approach:
    # For each m0, compute for each r in 1..L which i's can use it. Then do a matching.
    
    # Let's use a greedy/backtracking approach or just use a standard bipartite matching.
    
    # Given the small size (12 left nodes, up to 27720 right nodes), we can do:
    # For each m0, for W from 1 to L, check if matching exists in the subgraph with right nodes 1..W.
    # But that's L * L * (matching cost) which might be too slow.
    
    # Better: for each m0, use binary search on W. For a given W, build the bipartite graph
    # and check if perfect matching exists.
    
    # Actually, even better: since there are only 12 left nodes, we can use a DFS-based
    # bipartite matching (Kuhn's algorithm) which is O(12 * E) where E is the number of edges.
    
    # Let me just compute f for all m0 and find the max.
    
    def can_match(m0, W):
        """Check if we can assign distinct r_i in 1..W with r_i ≡ -m0 (mod i) for i=1..12."""
        # For each i, the valid r in 1..W are those with r ≡ -m0 (mod i)
        # Build adjacency: for each i, list of valid r values
        adj = [[] for _ in range(12)]
        for i in range(1, 13):
            # r ≡ -m0 (mod i), r in 1..W
            # r = k*i + (-m0 % i), but we need r >= 1
            rem = (-m0) % i
            if rem == 0:
                start_r = i  # smallest positive r with r ≡ 0 (mod i)
            else:
                start_r = rem
            # r values: start_r, start_r + i, start_r + 2i, ... <= W
            r = start_r
            while r <= W:
                adj[i-1].append(r)
                r += i
        
        # Now do bipartite matching using DFS
        match_r = [-1] * (W + 1)  # match_r[r] = i that is matched to r, or -1
        
        def dfs(i, visited):
            for r in adj[i]:
                if visited[r]:
                    continue
                visited[r] = True
                if match_r[r] == -1 or dfs(match_r[r], visited):
                    match_r[r] = i
                    return True
            return False
        
        for i in range(12):
            visited = [False] * (W + 1)
            if not dfs(i, visited):
                return False
        return True
    
    # For each m0, find the minimal W such that can_match(m0, W) is True.
    # Then f(12, m) = that minimal W.
    # We want to maximize f(12, m) over m in 0..1000000.
    # Since f depends only on m mod L, we just need max over m0 in 0..L-1.
    
    best_m0 = 0
    best_W = 0
    
    for m0 in range(L):
        # Binary search for minimal W
        lo, hi = 1, L
        ans = L
        while lo <= hi:
            mid = (lo + hi) // 2
            if can_match(m0, mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        if ans > best_W:
            best_W = ans
            best_m0 = m0
        
        # Check deadline
        if time.time() > deadline:
            break
    
    # Now we need to find an m in 0..1000000 with m mod L == best_m0
    # m = best_m0 + k*L for some k >= 0, and m <= 1000000
    # Smallest such m is best_m0 (if best_m0 <= 1000000)
    m = best_m0
    if m > 1000000:
        m = best_m0 % L  # This shouldn't happen since best_m0 < L = 27720 < 1000000
    
    return int(m)

# EVOLVE-BLOCK-END
