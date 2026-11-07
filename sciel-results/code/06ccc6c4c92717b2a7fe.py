import sys
sys.setrecursionlimit(10000)

# Precompute the answer for all residue classes modulo 27720
# period = lcm(1, 2, ..., 12) = 27720

def _precompute():
    n = 12
    period = 27720
    max_L = 144  # upper bound on f(12, m)
    
    # For each residue m mod period, find the minimal L such that
    # there exist distinct b_1, ..., b_n in (m, m+L] with i | b_i for all i
    
    # We'll do this by checking L from 1 to max_L for each residue.
    # For a given m and L, we need to check if there's a valid matching.
    # The window is (m, m+L], i.e., numbers m+1, m+2, ..., m+L.
    # For each i in 1..n, the valid numbers in the window are those d in 1..L
    # such that i divides (m+d).
    
    # To speed up, we can use bipartite matching (Hopcroft-Karp or just DFS since n=12 is small)
    # For each residue class, we can precompute which d values work for each i.
    
    # Actually, let's just brute-force with a recursive matching for each (m, L) pair.
    # But that's 27720 * 144 = ~4 million calls, each with a small matching. Should be OK.
    
    # Better: for each m, incrementally check L=1,2,... until we find a valid one.
    
    # Let's precompute for each i and each d in 1..max_L, what residues m make i|(m+d).
    # i divides (m+d) iff m ≡ -d (mod i)
    
    # For a given m (residue mod period), and a given L, the candidates for i are:
    # d in 1..L such that m ≡ -d (mod i), i.e., d ≡ -m (mod i)
    
    # Let's just do it directly:
    
    results = [0] * period  # results[r] = minimal L for m ≡ r (mod period)
    
    for m in range(period):
        # Find minimal L
        found = False
        for L in range(1, max_L + 1):
            # For each i, collect valid d values in 1..L where i | (m+d)
            # valid[i] = list of d in 1..L such that (m + d) % i == 0
            
            # Build bipartite graph: left side = i in 1..n, right side = d in 1..L
            # Edge exists if i | (m+d)
            
            # Use DFS-based matching (n=12, L up to 144, but we can limit)
            # Actually, we need to match each i to a distinct d.
            # Left side has n=12 nodes, right side has L nodes.
            
            # For efficiency, let's represent the valid d's for each i as a bitmask or list.
            
            # Let's just do a simple backtracking/DFS matching.
            
            # adj[i] = list of d values (1-indexed d, so d in 1..L)
            adj = [[] for _ in range(n)]
            for i in range(1, n + 1):
                # d in 1..L such that (m + d) % i == 0
                # d ≡ -m (mod i)
                start_d = (-m) % i
                if start_d == 0:
                    start_d = i
                # d values: start_d, start_d + i, start_d + 2i, ... <= L
                d = start_d
                while d <= L:
                    adj[i-1].append(d)
                    d += i
            
            # Now check if there's a perfect matching from {1..n} to distinct d values
            # Use DFS augmenting path
            
            match = [-1] * (L + 1)  # match[d] = i that is matched to d (0-indexed i stored as i-1)
            
            def dfs(i_idx, visited):
                """Try to find an augmenting path for node i_idx (0-indexed)"""
                for d in adj[i_idx]:
                    if visited[d]:
                        continue
                    visited[d] = True
                    if match[d] == -1 or dfs(match[d], visited):
                        match[d] = i_idx
                        return True
                return False
            
            # Try to match all n nodes
            matched_count = 0
            for i_idx in range(n):
                visited = [False] * (L + 1)
                if dfs(i_idx, visited):
                    matched_count += 1
            
            if matched_count == n:
                results[m] = L
                found = True
                break
        
        if not found:
            results[m] = max_L  # Shouldn't happen, but set to max
    
    return results

# Precompute at module load time
_results = _precompute()

def construct():
    # We need to find m in 0..1000000 that maximizes f(12, m)
    # f(12, m) = results[m % 27720]
    # So we want the residue r that maximizes results[r], then pick any m ≡ r (mod 27720) in [0, 1000000]
    
    period = 27720
    best_r = 0
    best_val = -1
    for r in range(period):
        if _results[r] > best_val:
            best_val = _results[r]
            best_r = r
    
    # Find the smallest m in [0, 1000000] with m ≡ best_r (mod period)
    # m = best_r is in [0, 27719], which is within [0, 1000000]
    # So m = best_r works (since best_r < 27720 <= 1000000)
    
    return best_r
