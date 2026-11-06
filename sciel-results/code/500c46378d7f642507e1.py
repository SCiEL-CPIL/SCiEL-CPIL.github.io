def construct():
    import time
    import random
    import numpy as np
    
    start_time = time.time()
    deadline = start_time + 115
    
    def is_triangle_free_circulant(n, S):
        # S is a set of differences in 1..n//2
        # Triangle free iff no a,b,c in S such that a+b=c or a+b=n-c (i.e. a+b+c=n)
        # Actually, if S is symmetric, we check if S+S intersects S or -S.
        # -S is {n-s for s in S}.
        # So we check if (a+b) % n is in S or if (a+b) % n is in {n-s for s in S}.
        # Since we only store positive reps, let's just check all triples.
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)): # a,b can be same? 
                # In simple graph, no multi-edges.
                # Triangle vertices 0, a, a+b.
                # Edges: (0,a) in S, (a, a+b) in S (diff b), (0, a+b) in S (diff a+b).
                # So we need a, b, a+b all in S (mod n).
                # Also 0, a, a-b. Edges (0,a) diff a, (a, a-b) diff b, (0, a-b) diff a-b.
                # So we need a, b, a-b all in S.
                # This is equivalent to checking if S is sum-free in Z_n.
                # Let's just check all a,b in S.
                pass
        
        # Efficient check:
        # For each a in S, for each b in S:
        # sum = (a+b) % n
        # if sum in S: return False
        # diff = (a-b) % n
        # if diff in S: return False (This is covered by sum check if we consider -b? No, -b is n-b. If n-b in S, then a + (n-b) = a-b. So if S is symmetric, checking a+b in S is sufficient? 
        # If S is symmetric, then -b is in S. So a + (-b) = a-b. If a-b in S, then a + (n-b) is a sum of two elements in S.
        # So yes, if S is symmetric, checking a+b in S for all a,b in S is sufficient.
        
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                s = (a + b) % n
                if s in S_set:
                    return False
                d = (a - b) % n
                if d in S_set:
                    return False
        return True

    def alpha_exact(adj_bits, n):
        best = 0
        
        def dfs(candidates, current_size):
            nonlocal best
            if current_size + len(candidates) <= best:
                return
            if not candidates:
                best = max(best, current_size)
                return
                
            v = candidates[0]
            rest = candidates[1:]
            
            new_cands = [u for u in rest if not ((adj_bits[v] >> u) & 1)]
            dfs(new_cands, current_size + 1)
            
            dfs(rest, current_size)
            
            if best >= 10:
                return

        dfs(list(range(n)), 0)
        return best

    def build_adj(n, S):
        adj_bits = [0] * n
        for d in S:
            for i in range(n):
                j = (i + d) % n
                adj_bits[i] |= (1 << j)
        return adj_bits

    def find_circulant(n):
        # Try to find a large sum-free set S for n
        # Max possible size is roughly n/4? No, sum-free sets in Z_n can be size n/3 (e.g. {1, 2, ..., n/3} is not sum free? 1+2=3. {n/3+1, ..., 2n/3} is sum free? 
        # Actually, we want dense graph.
        # Let's try to build S greedily.
        
        best_S = set()
        
        for restart in range(200):
            if time.time() > deadline:
                return None
            
            # Randomize order of differences
            diffs = list(range(1, n // 2 + 1))
            random.shuffle(diffs)
            
            S = set()
            for d in diffs:
                # Check if adding d keeps S sum-free
                # We need to check if (S U {d}) is sum-free.
                # This is expensive if S is large.
                # Optimization: Only check interactions involving d.
                # d+d, d+s, s+d, s1+s2 (old)
                # Old S is sum-free.
                # New violations:
                # 1. d+d = 2d in S?
                # 2. d+s in S for s in S?
                # 3. s+d in S for s in S? (Same as 2)
                # 4. d-d = 0 (not in S)
                # 5. s-d in S? (Covered by sum check if S symmetric? 
                #    If s-d in S, then s = d + (s-d). So s is sum of d and (s-d).
                #    So checking if (d+s) in S for all s in S covers both addition and subtraction if we iterate all s.
                
                ok = True
                # Check 2d
                if (2 * d) % n in S:
                    ok = False
                # Check d+s
                if ok:
                    for s in S:
                        if (d + s) % n in S:
                            ok = False
                            break
                        if (d - s) % n in S:
                            ok = False
                            break
                
                if ok:
                    S.add(d)
            
            if len(S) > len(best_S):
                best_S = S.copy()
                
                # If we find a reasonably large S, check alpha
                # Heuristic: if len(S) > 5, it might be good.
                # For n=39, max degree is ~6.
                if len(S) >= 5:
                    adj_bits = build_adj(n, S)
                    alpha = alpha_exact(adj_bits, n)
                    if alpha < 10:
                        edges = []
                        for i in range(n):
                            for j in range(i+1, n):
                                if (adj_bits[i] >> j) & 1:
                                    edges.append([i, j])
                        return edges

        return None

    for n in range(40, 30, -1):
        if time.time() > deadline:
            break
        edges = find_circulant(n)
        if edges is not None:
            return {'n': n, 'edges': edges}
            
    # Fallback to a known small graph or random search
    # If circulant fails, try the old random search for smaller n
    # But we are running out of time.
    # Let's return a valid graph for n=26 or similar if we found one?
    # The fallback in the original code was n=26.
    
    # Let's try a quick random search for n=35..39 just in case circulant failed?
    # No, circulant is the structural change.
    
    # If we didn't find a circulant graph, we should return a valid graph for a smaller n to ensure score > 0.
    # The original code returned n=26.
    # Let's try to find a circulant graph for n=30?
    # Or just return the n=26 fallback.
    
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
