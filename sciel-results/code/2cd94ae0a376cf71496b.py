<<<<<<< SEARCH
    # Final verification and conversion
    # Verify best_adj is valid
    # Just to be safe, though the logic should hold
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                common = best_adj[u] & best_adj[v]
                if popcount(common) > 2:
                    valid = False
                    break
        if not valid:
            break
            
    if not valid:
        # Fallback to a known valid construction (e.g. 5xK4) if something went wrong
        # But the code should be correct.
        pass
=======
    # Final verification and conversion
    # Verify best_adj is valid
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj[u] >> v) & 1:
                common = best_adj[u] & best_adj[v]
                if popcount(common) > 2:
                    valid = False
                    break
        if not valid:
            break
            
    if not valid:
        # Fallback: rebuild from scratch with a simple valid construction
        best_adj = [0] * n
        best_cc = [[0] * n for _ in range(n)]
        # 5 disjoint K4's: vertices 0-3, 4-7, 8-11, 12-15, 16-19
        for base in range(0, 20, 4):
            for i in range(4):
                for j in range(i+1, 4):
                    u = base + i
                    v = base + j
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        # Recompute cc
        for u in range(n):
            for v in range(u + 1, n):
                if (best_adj[u] >> v) & 1:
                    common = best_adj[u] & best_adj[v]
                    best_cc[u][v] = popcount(common)
                    best_cc[v][u] = popcount(common)
>>>>>>> REPLACE
