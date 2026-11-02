"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 decomposition: 4 blocks of 2x2x2 (Strassen) + 1 block of 2x2x1 (Schoolbook)
    # Total rank = 4 * 7 + 4 = 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_idx = 0
    term_idx = 0
    
    # 4 Strassen blocks for columns 0-1, 2-3, 4-5, 6-7
    for block in range(4):
        j0 = block * 2
        j1 = block * 2 + 1
        
        # M1 = (a+d)(u_j0 + v_j1) ? No, standard Strassen:
        # M1 = (a+d)(e+h) where e=u_j0, h=v_j1
        # Indices in V: row 0 is u, row 1 is v.
        # V index for u_j0: j0 * 1 + 0? No, V is (m*p, rank).
        # V[i, k] where i is index in B.
        # B is 2x9. Index in B: row 0 (u), row 1 (v).
        # Col j corresponds to B[0, j] and B[1, j].
        # In V, index is row * p + col.
        # So u_j is V[0, j], v_j is V[1, j].
        
        # M1: (a+d) * (u_j0 + v_j1)
        U[0][term_idx] += 1 # a
        U[1][term_idx] += 0 # b
        U[2][term_idx] += 0 # c
        U[3][term_idx] += 1 # d
        
        V[0 * p + j0][term_idx] += 1 # u_j0
        V[1 * p + j1][term_idx] += 1 # v_j1
        
        # W for M1:
        # C = ... + M1 * (something)
        # Strassen reconstruction:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        # Wait, standard Strassen:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)(g+j)
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-d)(e+f)
        # M7 = (b-d)(g+j)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # W index: k*n + i.
        # k is col in C (0..8), i is row in C (0..1).
        # C11 is C[0, j0]. Index 0*n + 0 = 0.
        # C12 is C[0, j1]. Index 0*n + 1 = 1? No.
        # W index is k*n + i.
        # C[0, j0] -> k=0, i=0 -> 0.
        # C[0, j1] -> k=1, i=0 -> 1.
        # C[1, j0] -> k=0, i=1 -> n=2.
        # C[1, j1] -> k=1, i=1 -> 3.
        
        # But j0, j1 are global column indices.
        # C[0, j0] index: j0 * n + 0.
        # C[0, j1] index: j1 * n + 0.
        # C[1, j0] index: j0 * n + 1.
        # C[1, j1] index: j1 * n + 1.
        
        W[j0 * n + 0][term_idx] += 1 # M1
        W[j0 * n + 1][term_idx] += 1 # M4 (part of C21? No, M4 is in C11 and C21)
        # Let's re-verify Strassen reconstruction.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So for M1:
        W[j0 * n + 0][term_idx] += 1
        W[j1 * n + 1][term_idx] += 1
        term_idx += 1
        
        # M2 = (c+d)(v_j0 + v_j1)
        U[0][term_idx] += 0
        U[1][term_idx] += 0
        U[2][term_idx] += 1 # c
        U[3][term_idx] += 1 # d
        
        V[0 * p + j0][term_idx] += 0
        V[1 * p + j0][term_idx] += 1 # v_j0
        V[1 * p + j1][term_idx] += 1 # v_j1
        
        W[j0 * n + 1][term_idx] += 1 # C21
        W[j1 * n + 1][term_idx] += -1 # C22
        term_idx += 1
        
        # M3 = a(u_j1 - v_j1)
        U[0][term_idx] += 1 # a
        U[1][term_idx] += 0
        U[2][term_idx] += 0
        U[3][term_idx] += 0
        
        V[0 * p + j1][term_idx] += 1 # u_j1
        V[1 * p + j1][term_idx] += -1 # v_j1
        
        W[j0 * n + 0][term_idx] += 0 # C11
        W[j1 * n + 0][term_idx] += 1 # C12
        W[j1 * n + 1][term_idx] += 1 # C22
        term_idx += 1
        
        # M4 = d(v_j0 - u_j0)
        U[0][term_idx] += 0
        U[1][term_idx] += 0
        U[2][term_idx] += 0
        U[3][term_idx] += 1 # d
        
        V[0 * p + j0][term_idx] += -1 # u_j0
        V[1 * p + j0][term_idx] += 1 # v_j0
        
        W[j0 * n + 0][term_idx] += 1 # C11
        W[j0 * n + 1][term_idx] += 1 # C21
        term_idx += 1
        
        # M5 = (a+b)v_j1
        U[0][term_idx] += 1 # a
        U[1][term_idx] += 1 # b
        U[2][term_idx] += 0
        U[3][term_idx] += 0
        
        V[0 * p + j1][term_idx] += 0
        V[1 * p + j1][term_idx] += 1 # v_j1
        
        W[j0 * n + 0][term_idx] += -1 # C11
        W[j1 * n + 0][term_idx] += 1 # C12
        term_idx += 1
        
        # M6 = (c-d)(u_j0 + u_j1)
        U[0][term_idx] += 0
        U[1][term_idx] += 0
        U[2][term_idx] += 1 # c
        U[3][term_idx] += -1 # d
        
        V[0 * p + j0][term_idx] += 1 # u_j0
        V[0 * p + j1][term_idx] += 1 # u_j1
        
        W[j1 * n + 1][term_idx] += 1 # C22
        term_idx += 1
        
        # M7 = (b-d)(v_j0 + v_j1)
        U[0][term_idx] += 0
        U[1][term_idx] += 1 # b
        U[2][term_idx] += 0
        U[3][term_idx] += -1 # d
        
        V[0 * p + j0][term_idx] += 1 # v_j0
        V[1 * p + j1][term_idx] += 1 # v_j1
        
        W[j0 * n + 0][term_idx] += 1 # C11
        W[j1 * n + 1][term_idx] += 0 # C22? No, M7 is only in C11?
        # C11 = M1 + M4 - M5 + M7. Yes.
        term_idx += 1

    # 1 Schoolbook block for column 8
    j0 = 8
    # M8 = a*u_j0
    U[0][term_idx] += 1
    U[1][term_idx] += 0
    U[2][term_idx] += 0
    U[3][term_idx] += 0
    V[0 * p + j0][term_idx] += 1
    W[j0 * n + 0][term_idx] += 1
    term_idx += 1
    
    # M9 = b*v_j0
    U[0][term_idx] += 0
    U[1][term_idx] += 1
    U[2][term_idx] += 0
    U[3][term_idx] += 0
    V[0 * p + j0][term_idx] += 0
    V[1 * p + j0][term_idx] += 1
    W[j0 * n + 0][term_idx] += 1
    term_idx += 1
    
    # M10 = c*u_j0
    U[0][term_idx] += 0
    U[1][term_idx] += 0
    U[2][term_idx] += 1
    U[3][term_idx] += 0
    V[0 * p + j0][term_idx] += 1
    W[j0 * n + 1][term_idx] += 1
    term_idx += 1
    
    # M11 = d*v_j0
    U[0][term_idx] += 0
    U[1][term_idx] += 0
    U[2][term_idx] += 0
    U[3][term_idx] += 1
    V[0 * p + j0][term_idx] += 0
    V[1 * p + j0][term_idx] += 1
    W[j0 * n + 1][term_idx] += 1
    term_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
