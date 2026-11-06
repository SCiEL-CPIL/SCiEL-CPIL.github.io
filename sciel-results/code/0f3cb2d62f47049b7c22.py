"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # We know R(3,10) <= 41, so max n is 40.
    # The seed provides n=26.
    # Known constructions:
    # - Andrasfai(9) gives n=26.
    # - There are known graphs of order 38 and 39 for R(3,10).
    # - Specifically, the largest known triangle-free graph with no independent set of size 10 has 39 vertices.
    # - A specific construction for n=39 is based on the Clebsch graph or similar, but often cited as a specific known instance.
    
    # Let's try to construct a larger graph.
    # A common approach is to use known specific graphs.
    # The graph of order 39 is often constructed by specific means.
    # One known construction for R(3,10) > 39 involves a graph on 39 vertices.
    
    # Since we cannot easily generate the specific 39-vertex graph from scratch without a database or complex search,
    # and the directive says to maximize n, let's see if we can find a simpler larger graph.
    
    # Actually, the Andrasfai graph And(9) is on 26 vertices.
    # Is there a simple parametric family?
    # The general Andrasfai graph And(2k+1) is on 3(2k+1) vertices? No.
    # And(r) is on 3(r-1) vertices?
    # And(3) = C5 (5 vertices, alpha=2). 3*(3-1)=6? No, C5 is 5.
    # And(5) is on 10 vertices? No, And(5) is the 5-cycle? No.
    # Andrasfai graphs: And(2k+1) is a graph on 3(2k+1) vertices?
    # Let's check: And(3) is C5 (5 vertices). 3*(3-1) = 6? No.
    # And(3) is defined on Z_5?
    # The standard Andrasfai graph And(r) for odd r >= 3 is on 3(r-1) vertices?
    # For r=3, 3(2) = 6 vertices? But C5 is 5.
    # Wait, the seed says And(9) is on 26 vertices.
    # 3*(9-1) = 24. 26 is not 24.
    # Let's re-read the seed comment: "Andrasfai graph And(9): Z_26 with i ~ j iff (j - i) mod 26 is 1 mod 3."
    # This is a circulant graph C_26(1, 2, ...)?
    # If (j-i) % 3 == 1, then differences are 1, 4, 7, 10, 13, 16, 19, 22, 25.
    # These are 9 differences.
    # Degree is 9.
    # This is the graph C_26(1, 4, 7, 10, 13, 16, 19, 22, 25).
    # This is a specific graph.
    
    # Can we scale this?
    # If we take n = 3k+2?
    # For k=8, n=26.
    # If we take n=32? (3*10+2).
    # Differences 1 mod 3: 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31. (11 diffs)
    # Let's test if this works for larger n.
    # The condition for triangle-free:
    # In a circulant graph, if a, b, c are edges, a+b+c != 0 mod n.
    # Here edges are 1 mod 3.
    # Sum of three numbers 1 mod 3 is 3 mod 3 = 0 mod 3.
    # So we need a+b+c != 0 mod n.
    # Since a,b,c are 1 mod 3, a+b+c is divisible by 3.
    # If n is not divisible by 3, then a+b+c cannot be 0 mod n unless a+b+c = n, 2n, etc.
    # Max sum of 3 diffs: 3*(n-1).
    # If n=26, max sum 75. 26, 52, 78.
    # Possible sums divisible by 3: 26 (no), 52 (no), 78 (yes).
    # Can we get 78? 25+25+28? No, max diff is 25. 25+25+25=75.
    # So for n=26, no triangle.
    
    # What about n=32?
    # Diffs: 1, 4, ..., 31.
    # Sum of 3 diffs divisible by 3.
    # n=32 is not div by 3.
    # Multiples of 32: 32, 64, 96.
    # None are divisible by 3.
    # So sum of 3 diffs (div by 3) can never be 0 mod 32.
    # So it is triangle-free!
    
    # What is the independence number?
    # For C_n(diffs 1 mod 3), the independence number is roughly n/3?
    # For n=26, alpha=9. 26/3 = 8.66.
    # For n=32, alpha should be around 32/3 = 10.66?
    # If alpha >= 10, it's invalid.
    # Let's check n=32.
    # If alpha(32) >= 10, we can't use it.
    # The seed says alpha(26)=9.
    # If the pattern holds, alpha(n) ~ n/3.
    # We need alpha < 10, so alpha <= 9.
    # This implies n/3 < 10 => n < 30?
    # Or n/3 <= 9 => n <= 27?
    # But n=26 works.
    # What about n=29?
    # 29/3 = 9.66.
    # If alpha(29) = 10, it fails.
    # If alpha(29) = 9, it works.
    
    # Let's try to generate graphs for n from 40 down to 26 using this circulant family and check alpha.
    # But checking alpha exactly is hard.
    # However, we can use a greedy bound or known results.
    
    # Actually, there is a known result: The largest triangle-free graph with alpha < 10 has 39 vertices.
    # But we don't have that graph.
    # The seed is 26.
    # If we can find a valid graph with n > 26, we improve.
    
    # Let's test n=27, 28, 29, 30, 31, 32 with the circulant construction C_n(1 mod 3).
    # We need to verify alpha < 10.
    # We can run a simple branch and bound for alpha for small n to verify.
    # But we have a time limit.
    
    # Let's try n=27.
    # Diffs: 1, 4, ..., 25.
    # Triangle-free? n=27 is div by 3.
    # Sum of 3 diffs (div by 3) can be 0 mod 27.
    # e.g., 1+4+22 = 27. 1, 4, 22 are all 1 mod 3.
    # So (0,1), (1,5), (5,27=0)?
    # 0~1 (diff 1), 1~5 (diff 4), 5~0 (diff 22).
    # Triangle! So n=27 fails.
    
    # n=28?
    # n=28 not div by 3.
    # Multiples of 28: 28, 56, 84.
    # None div by 3.
    # So triangle-free.
    # What is alpha(28)?
    # If alpha(26)=9, maybe alpha(28)=10?
    # Let's assume the density is similar.
    # If alpha(28) >= 10, it fails.
    
    # Let's try to compute alpha for these specific graphs using a simple solver within the time limit.
    # We can use a simple backtracking for independent set.
    
    def has_independent_set_of_size_k(n, adj_matrix, k):
        # Simple backtracking
        # adj_matrix[i][j] is True if i~j
        # We want to find a set of k vertices with no edges between them.
        # This is equivalent to finding a clique of size k in the complement graph.
        # Or just independent set.
        
        # Use Bron-Kerbosch on complement? Or simple recursion.
        # Given time constraints, let's try a simple heuristic or bound first.
        # But for n<=32, exact check is fast.
        
        # Let's use a recursive function with memoization or simple pruning.
        # Actually, for n~30, alpha~10, it might be slow.
        # But we only need to check a few n values.
        
        def backtrack(candidates, current_set):
            if len(current_set) == k:
                return True
            if len(candidates) < k - len(current_set):
                return False
            
            # Heuristic: pick vertex with lowest degree in induced subgraph?
            # Or just pick first.
            v = candidates[0]
            remaining = candidates[1:]
            
            # Include v
            if backtrack([x for x in remaining if not adj_matrix[v][x]], current_set + [v]):
                return True
            
            # Exclude v
            if backtrack(remaining, current_set):
                return True
            
            return False
        
        return backtrack(list(range(n)), [])
    
    best_n = 26
    best_edges = [[i, j] for i in range(26) for j in range(i + 1, 26) if (j - i) % 26 % 3 == 1]
    
    # Try n from 40 down to 27
    for n in range(40, 26, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Construct circulant graph C_n(1 mod 3)
        adj = [[False]*n for _ in range(n)]
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                diff = (j - i) % n
                if diff % 3 == 1:
                    adj[i][j] = True
                    adj[j][i] = True
                    edges.append([i, j])
        
        # Check triangle-free
        # For circulant, we can just check if any sum of 3 diffs is 0 mod n.
        # But we already argued n not div by 3 implies triangle-free.
        # If n div by 3, it likely has triangles.
        if n % 3 == 0:
            continue
        
        # Check alpha < 10
        # We need to ensure NO independent set of size 10.
        # So we check if has_independent_set_of_size_k(n, adj, 10) is False.
        
        # To speed up, we can first check a lower bound on alpha.
        # If we can find an independent set of size 10, it fails.
        # Greedy:
        remaining = set(range(n))
        independent = []
        while remaining:
            # Pick vertex with min degree in remaining
            v = min(remaining, key=lambda x: sum(adj[x][y] for y in remaining))
            independent.append(v)
            remaining.discard(v)
            remaining = {x for x in remaining if not adj[v][x]}
        
        if len(independent) >= 10:
            continue # Fails
        
        # If greedy finds < 10, it might still have one.
        # But for these regular-ish graphs, greedy is often close.
        # Let's do a quick exact check if greedy < 10.
        # But exact check might be slow.
        # Let's assume if greedy < 10, it's likely valid?
        # No, greedy is a lower bound. If greedy < 10, alpha could be 10.
        # We need to be sure.
        
        # Let's run the exact check but with a time limit per n.
        t_start = time.time()
        
        # Use a more efficient check: Max Independent Set via branch and bound
        # But for n=30, it's small enough.
        
        if has_independent_set_of_size_k(n, adj, 10):
            continue # Fails
        
        if time.time() - t_start > 10.0:
            continue # Too slow, skip to be safe? Or assume valid?
            # Better to skip if too slow to avoid timeout.
        
        # If we get here, n is a valid candidate
        best_n = n
        best_edges = edges
        break # We want max n, so stop at first valid found from top down
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
