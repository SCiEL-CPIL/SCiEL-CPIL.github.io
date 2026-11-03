import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition
    rank = 34
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, val=1):
        nonlocal col
        for u in u_indices:
            U[u, col] += val
        for v in v_indices:
            V[v, col] += val
        for w in w_indices:
            W[w, col] += val
        col += 1
        
    # Global indices:
    # A: i*3 + j
    # B: r*6 + s
    # C: q*2 + p
    
    # 1. Strassen for A_L (cols 0,1) * B_top (rows 0,1, cols 0,1) -> C_top (cols 0,1)
    # A_L: i in 0,1; j in 0,1
    # B_top: r in 0,1; s in 0,1
    # C_top: q in 0,1; p in 0,1
    
    # Term 1
    add_term([0*3+0, 1*3+1], [0*6+0, 1*6+1], [0*2+0, 1*2+1])
    # Term 2
    add_term([1*3+0, 1*3+1], [0*6+0], [1*2+0, 0*2+1]) # W2: C21 += 1, C12 += -1? No.
    # C21 = M2 + M4. C22 = M1 - M2 + M3 + M6.
    # So M2 contributes to C21 (+1) and C22 (-1).
    # W2 indices: [1*2+0, 1*2+1] with values [1, -1].
    # My add_term uses single val. I need to support multiple values or call multiple times.
    # I'll modify add_term to take a list of (idx, val).
    
    # Let's restart the helper.
