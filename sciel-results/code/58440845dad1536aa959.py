def _construct_luw(k, g, deadline):
    # ... existing code ...
    
    q = k
    # Find smallest d such that 2d + 4 >= g
    d = max(2, (g - 4) // 2)
    while 2 * d + 4 < g:
        d += 1
        
    # Ensure n is large enough (heuristic: use bipartite Moore bound)
    # M(k, g) for bipartite: 2 * k * sum((k-1)^i for i in range(g//2 - 1))
    # For g=10, k=4: 2*4*(1+3+9+27) = 320.
    # If 2*q^d < M, increase d.
    
    def moore_bound_bipartite(k, g):
        if g % 2 != 0:
            # Not bipartite bound, use general
            r = (g - 1) // 2
            return 1 + k * sum((k-1)**i for i in range(r))
        else:
            r = g // 2 - 1
            return 2 * k * sum((k-1)**i for i in range(r))
            
    min_n = moore_bound_bipartite(k, g)
    
    while 2 * (q ** d) < min_n:
        d += 1
        
    # ... rest of code ...
