"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 5 blocks * 7 Strassen terms
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Pre-define the 7 Strassen terms for a 2x2 block
    # u: indices 0 (0,0), 1 (0,1), 2 (1,0), 3 (1,1)
    # v: local indices 0 (0,k0), 1 (0,k1), 2 (1,k0), 3 (1,k1)
    # w: local indices 0 (k0,0), 1 (k0,1), 2 (k1,0), 3 (k1,1)
    
    terms = [
        # u, v, w
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1: C00+=1, C11+=1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]), # M2: C10+=1, C11+=-1
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]), # M3: C01+=1, C11+=1
        ([0, 0, 0, 1], [-1, 1, 0, 0], [1, 1, 0, 0]), # M4: C00+=1, C10+=1
        ([1, 0, 1, 0], [0, 0, 0, 1], [-1, 0, 1, 0]), # M5: C00+=-1, C01+=1
        ([-1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6: C11+=1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7: C00+=1
    ]
    
    col = 0
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Global V indices for local v indices 0,1,2,3
        # Local 0: j=0, k=k0 -> 0*p + k0 = k0
        # Local 1: j=0, k=k1 -> 0*p + k1 = k1
        # Local 2: j=1, k=k0 -> 1*p + k0 = 10 + k0
        # Local 3: j=1, k=k1 -> 1*p + k1 = 10 + k1
        v_indices = [k0, k1, 10 + k0, 10 + k1]
        
        # Global W indices for local w indices 0,1,2,3
        # Local 0: k=k0, i=0 -> k0*n + 0 = 2*k0
        # Local 1: k=k0, i=1 -> k0*n + 1 = 2*k0 + 1
        # Local 2: k=k1, i=0 -> k1*n + 0 = 2*k1
        # Local 3: k=k1, i=1 -> k1*n + 1 = 2*k1 + 1
        w_indices = [2 * k0, 2 * k0 + 1, 2 * k1, 2 * k1 + 1]
        
        for u, v, w in terms:
            # Fill U
            for i in range(4):
                if u[i] != 0:
                    U[i][col] = u[i]
            
            # Fill V
            for i in range(4):
                if v[i] != 0:
                    V[v_indices[i]][col] = v[i]
            
            # Fill W
            for i in range(4):
                if w[i] != 0:
                    W[w_indices[i]][col] = w[i]
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
