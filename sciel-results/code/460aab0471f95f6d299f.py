import time
from math import gcd
from functools import reduce

def construct():
    divisors = [6, 10, 14, 15, 21, 22, 26, 33, 35, 39]
    n = len(divisors)
    MAX_V = 1000000
    deadline = time.time() + 80.0
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    M = reduce(lcm, divisors)
    
    # Precompute for each divisor the residues mod something? No, just check divisibility.
    
    # Kuhn's algorithm for maximum bipartite matching
    # Left side: n divisors (0..n-1)
    # Right side: values in [L, L+W] that are multiples of at least one divisor
    # But we only care about values that are multiples of their specific divisor.
    # Actually, let's build adjacency: for each divisor i, list of values v in [L,L+W] where v%d_i==0
    
    def has_perfect_matching(adj):
        """
        adj: list of lists, adj[i] = list of right-nodes (values) adjacent to left-node i
        Returns True if there's a perfect matching.
        Uses Kuhn's algorithm.
        """
        # match_r[v] = which left node is matched to right node v
        # We'll use a dict for match_r since values can be large
        match_r = {}
        
        def dfs(u, visited):
            for v in adj[u]:
                if v in visited:
                    continue
                visited.add(v)
                if v not in match_r or dfs(match_r[v], visited):
                    match_r[v] = u
                    return True
            return False
        
        for u in range(n):
            visited = set()
            if not dfs(u, visited):
                return False
        return True
    
    for W in range(9, 500):
        if time.time() > deadline:
            break
            
        for L in range(1, M + 1):
            if time.time() > deadline:
                break
                
            upper = L + W
            adj = []
            valid = True
            
            for d in divisors:
                first_k = (L + d - 1) // d
                last_k = upper // d
                if first_k > last_k:
                    valid = False
                    break
                c = [k * d for k in range(first_k, last_k + 1)]
                adj.append(c)
                
            if not valid:
                continue
            
            # Check perfect matching
            if has_perfect_matching(adj):
                # Now extract the actual matching
                # Re-run to get the assignment
                match_r = {}
                
                def dfs2(u, visited):
                    for v in adj[u]:
                        if v in visited:
                            continue
                        visited.add(v)
                        if v not in match_r or dfs2(match_r[v], visited):
                            match_r[v] = u
                            return True
                    return False
                
                for u in range(n):
                    visited = set()
                    dfs2(u, visited)
                
                # match_r[v] = u means value v is assigned to divisor u
                # Build result: for each divisor i, find v such that match_r[v] == i
                result = [0] * n
                for v, u in match_r.items():
                    result[u] = v
                
                # Verify all are distinct and within bounds
                if len(set(result)) == n and all(1 <= v <= MAX_V for v in result):
                    return [int(v) for v in result]
                
    # Fallback
    return [int(d) for d in divisors]
