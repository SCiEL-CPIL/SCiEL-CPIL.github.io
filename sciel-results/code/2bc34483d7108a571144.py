import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    # Rank 36 decomposition
    rank = 36
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # A is 2x2, B is 2x2, C is 2x2
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global offsets: i_off, j_off, k_off
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # M1 = (A00 + A11) * (B00 + B11)
        U[i_off * 2 + 0][col] += 1
        U[i_off * 2 + 1][col] += 1
        V[j_off * 2 + 0][col] += 1
        V[j_off * 2 + 1][col] += 1
        W[k_off * 2 + 0][col] += 1
        W[k_off * 2 + 1][col] += 1
        # C00 += M1, C11 += M1
        # W is indexed by [k*n + i, col]
        # C00: k=0, i=0 -> W[0*n + 0]
        # C11: k=1, i=1 -> W[1*n + 1]
        # Wait, W[k*n + i]
        # C00 is i=0, k=0. W[0*2 + 0] = W[0]
        # C11 is i=1, k=1. W[1*2 + 1] = W[3]
        # My previous W assignment was wrong.
        # Let's redefine W assignment carefully.
        
        # Actually, let's just build the full matrices by iterating over the terms.
        pass

    # It's safer to just iterate and add to the matrices directly.
    # U[i*m + j, col]
    # V[j*p + k, col]
    # W[k*n + i, col]
    
    # Strassen M1: (A00+A11)(B00+B11)
    # A00: i=0, j=0. A11: i=1, j=1.
    # B00: j=0, k=0. B11: j=1, k=1.
    # C00: i=0, k=0. C11: i=1, k=1.
    # W[k*n + i]
    # C00 -> W[0*2 + 0] = W[0]
    # C11 -> W[1*2 + 1] = W[3]
    
    # Let's re-do the helper with correct W indexing.
    def add_term(U_idx, V_idx, W_idx, val):
        nonlocal col
        U[U_idx][col] += val
        V[V_idx][col] += val
        W[W_idx][col] += val
        col += 1

    # Strassen 2x2x2
    # i_off, j_off, k_off are offsets for i, j, k
    def add_strassen(i_off, j_off, k_off):
        # M1 = (A00 + A11) * (B00 + B11)
        # U: i=0,1; j=0,1
        # V: j=0,1; k=0,1
        # W: k=0,1; i=0,1
        # C00 (i=0, k=0): W[k_off*n + i_off]
        # C11 (i=1, k=1): W[(k_off+1)*n + i_off+1]
        
        # We need to add 7 terms.
        # M1
        # U[i_off, j_off] += 1, U[i_off+1, j_off+1] += 1
        # V[j_off, k_off] += 1, V[j_off+1, k_off+1] += 1
        # W[k_off, i_off] += 1, W[k_off+1, i_off+1] += 1
        
        # This is getting complicated to do with single add_term calls because U and V have multiple indices.
        # We need a function that adds a rank-1 tensor to U, V, W.
        # But Strassen terms are sums of products.
        # Each M is a product of a linear form in A and a linear form in B.
        # So each M corresponds to ONE column in U, V, W.
        # U and V entries are the coefficients of the linear forms.
        # W entries are the coefficients of the linear combination for C.
        
        pass

    # Let's implement add_rank1_term(U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: dict {(i, j): val}
    # V_coeffs: dict {(j, k): val}
    # W_coeffs: dict {(k, i): val}
    
    def add_rank1(U_c, V_c, W_c):
        nonlocal col
        for (i, j), val in U_c.items():
            U[i * m + j][col] += val
        for (j, k), val in V_c.items():
            V[j * p + k][col] += val
        for (k, i), val in W_c.items():
            W[k * n + i][col] += val
        col += 1

    # Strassen 2x2x2
    def add_strassen(i_off, j_off, k_off):
        # M1 = (A00 + A11)(B00 + B11)
        U_c = {(i_off, j_off): 1, (i_off+1, j_off+1): 1}
        V_c = {(j_off, k_off): 1, (j_off+1, k_off+1): 1}
        W_c = {(k_off, i_off): 1, (k_off+1, i_off+1): 1}
        add_rank1(U_c, V_c, W_c)
        
        # M2 = (A10 + A11) B00
        U_c = {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1}
        V_c = {(j_off, k_off): 1}
        W_c = {(k_off, i_off+1): 1}
        add_rank1(U_c, V_c, W_c)
        
        # M3 = A00 (B01 - B11)
        U_c = {(i_off, j_off): 1}
        V_c = {(j_off, k_off+1): 1, (j_off+1, k_off+1): -1}
        W_c = {(k_off+1, i_off): 1}
        add_rank1(U_c, V_c, W_c)
        
        # M4 = A11 (B10 - B00)
        U_c = {(i_off+1, j_off+1): 1}
        V_c = {(j_off+1, k_off): 1, (j_off, k_off): -1}
        W_c = {(k_off, i_off+1): 1}
        add_rank1(U_c, V_c, W_c)
        
        # M5 = (A00 + A01) B11
        U_c = {(i_off, j_off): 1, (i_off, j_off+1): 1}
        V_c = {(j_off+1, k_off+1): 1}
        W_c = {(k_off+1, i_off): 1}
        add_rank1(U_c, V_c, W_c)
        
        # M6 = (A10 - A00)(B00 + B01)
        U_c = {(i_off+1, j_off): 1, (i_off, j_off): -1}
        V_c = {(j_off, k_off): 1, (j_off, k_off+1): 1}
        W_c = {(k_off+1, i_off+1): 1}
        add_rank1(U_c, V_c, W_c)
        
        # M7 = (A01 - A11)(B10 + B11)
        U_c = {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1}
        V_c = {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1}
        W_c = {(k_off+1, i_off): 1}
        add_rank1(U_c, V_c, W_c)

    # Schoolbook 2x2x1
    def add_schoolbook_221(i_off, j_off, k_off):
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # Terms:
        # T1 = A00 B0 -> C0
        U_c = {(i_off, j_off): 1}
        V_c = {(j_off, k_off): 1}
        W_c = {(k_off, i_off): 1}
        add_rank1(U_c, V_c, W_c)
        
        # T2 = A01 B1 -> C0
        U_c = {(i_off, j_off+1): 1}
        V_c = {(j_off+1, k_off): 1}
        W_c = {(k_off, i_off): 1}
        add_rank1(U_c, V_c, W_c)
        
        # T3 = A10 B0 -> C1
        U_c = {(i_off+1, j_off): 1}
        V_c = {(j_off, k_off): 1}
        W_c = {(k_off, i_off+1): 1}
        add_rank1(U_c, V_c, W_c)
        
        # T4 = A11 B1 -> C1
        U_c = {(i_off+1, j_off+1): 1}
        V_c = {(j_off+1, k_off): 1}
        W_c = {(k_off, i_off+1): 1}
        add_rank1(U_c, V_c, W_c)

    # Loop over m-blocks (2 blocks of size 2)
    for a in range(2):
        j_off = a * 2
        # Loop over p-subblocks (2 blocks of size 2, 1 block of size 1)
        # p=5: 0,1 (size 2), 2,3 (size 2), 4 (size 1)
        
        # p-block 0: k=0,1
        add_strassen(0, j_off, 0)
        
        # p-block 1: k=2,3
        add_strassen(0, j_off, 2)
        
        # p-block 2: k=4
        add_schoolbook_221(0, j_off, 4)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
