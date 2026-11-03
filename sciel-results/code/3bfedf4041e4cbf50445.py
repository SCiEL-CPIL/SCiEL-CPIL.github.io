import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=3, m=5, p=6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term to U, V, W
    # a_idx: index in A (0..14)
    # b_idx: index in B (0..29)
    # c_idx: index in C (0..17)
    # coeff: coefficient (1, -1, etc)
    def add_term(a_idx, b_idx, c_idx, coeff):
        nonlocal col
        if a_idx is not None:
            U[a_idx][col] += coeff
        if b_idx is not None:
            V[b_idx][col] += coeff
        if c_idx is not None:
            W[c_idx][col] += coeff
        col += 1

    # Helper for schoolbook term A[i,j] * B[j,k] -> C[k,i]
    # Note: W index is k*n + i
    def add_schoolbook(i, j, k):
        a_idx = i * m + j
        b_idx = j * p + k
        c_idx = k * n + i
        add_term(a_idx, b_idx, c_idx, 1)

    # Helper for Strassen 2x2x2 block
    # i0, i1: row indices in A
    # j0, j1: col indices in A (row indices in B)
    # k0, k1: col indices in B (col indices in C)
    def add_strassen(i0, i1, j0, j1, k0, k1):
        # A indices
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1
        
        # B indices
        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1
        
        # C indices
        c00 = k0 * n + i0
        c01 = k0 * n + i1 # Wait, C[k,i]
        # C[k0, i0], C[k0, i1], C[k1, i0], C[k1, i1]
        c_k0_i0 = k0 * n + i0
        c_k0_i1 = k0 * n + i1
        c_k1_i0 = k1 * n + i0
        c_k1_i1 = k1 * n + i1

        # M1 = (A00+A11)(B00+B11)
        # We need to represent linear combinations.
        # The decomposition is a sum of rank-1 tensors.
        # Each rank-1 tensor is u * v * w.
        # Here u, v, w are vectors.
        # For M1, u = [1, 1, 0, 0] (on A00, A11), v = [1, 1, 0, 0] (on B00, B11), w = [1, 1, 1, 1] (on all C)?
        # No, Strassen works by defining intermediate products M_k, and then C is linear combination of M_k.
        # This means the tensor is sum of M_k * (coeffs for C).
        # But M_k themselves are products of linear combinations of A and B.
        # So each M_k is a rank-1 tensor in the A-B space?
        # No, M_k = (linear A) * (linear B).
        # So the whole term is (linear A) * (linear B) * (linear C).
        # This is a rank-1 tensor in the 3-way space.
        
        # So for each M_k, we add ONE column to U, V, W.
        # The column in U has the coefficients of the linear combination of A.
        # The column in V has the coefficients of the linear combination of B.
        # The column in W has the coefficients of the linear combination of C.

        # M1
        add_term(None, None, None, 0) # Placeholder, we need to fill specific indices
        # U: A00=1, A11=1
        U[a00][col] = 1
        U[a11][col] = 1
        # V: B00=1, B11=1
        V[b00][col] = 1
        V[b11][col] = 1
        # W: C00=1, C01=1, C10=1, C11=1 ?
        # C00 = M1 + ...
        # C01 = ...
        # C10 = ...
        # C11 = M1 - ...
        # So W coeffs: C_k0_i0=1, C_k0_i1=1, C_k1_i0=1, C_k1_i1=1
        W[c_k0_i0][col] = 1
        W[c_k0_i1][col] = 1
        W[c_k1_i0][col] = 1
        W[c_k1_i1][col] = 1
        col += 1

        # M2 = (A10+A11)B00
        U[a10][col] = 1
        U[a11][col] = 1
        V[b00][col] = 1
        # W: C10=1, C11=-1
        W[c_k1_i0][col] = 1
        W[c_k1_i1][col] = -1
        col += 1

        # M3 = A00(B10-B11)
        U[a00][col] = 1
        V[b10][col] = 1
        V[b11][col] = -1
        # W: C01=1, C11=1
        W[c_k0_i1][col] = 1
        W[c_k1_i1][col] = 1
        col += 1

        # M4 = A11(B01-B00)
        U[a11][col] = 1
        V[b01][col] = 1
        V[b00][col] = -1
        # W: C00=1, C10=1
        W[c_k0_i0][col] = 1
        W[c_k1_i0][col] = 1
        col += 1

        # M5 = (A00+A01)B11
        U[a00][col] = 1
        U[a01][col] = 1
        V[b11][col] = 1
        # W: C00=-1, C01=1
        W[c_k0_i0][col] = -1
        W[c_k0_i1][col] = 1
        col += 1

        # M6 = (A10-A00)(B00+B01)
        U[a10][col] = 1
        U[a00][col] = -1
        V[b00][col] = 1
        V[b01][col] = 1
        # W: C11=1
        W[c_k1_i1][col] = 1
        col += 1

        # M7 = (A01-A11)(B10+B11)
        U[a01][col] = 1
        U[a11][col] = -1
        V[b10][col] = 1
        V[b11][col] = 1
        # W: C00=1
        W[c_k0_i0][col] = 1
        col += 1

    # Loop over k blocks
    for k_block in range(3):
        k0 = k_block * 2
        k1 = k_block * 2 + 1
        
        # i in {0, 1}
        # j in {0, 1}
        add_strassen(0, 1, 0, 1, k0, k1)
        # j in {2, 3}
        add_strassen(0, 1, 2, 3, k0, k1)
        # j = 4
        # Schoolbook for i=0,1 and j=4
        add_schoolbook(0, 4, k0)
        add_schoolbook(0, 4, k1)
        add_schoolbook(1, 4, k0)
        add_schoolbook(1, 4, k1)
        
        # i = 2
        # j in {0, 1}
        add_schoolbook(2, 0, k0)
        add_schoolbook(2, 0, k1)
        add_schoolbook(2, 1, k0)
        add_schoolbook(2, 1, k1)
        # j in {2, 3}
        add_schoolbook(2, 2, k0)
        add_schoolbook(2, 2, k1)
        add_schoolbook(2, 3, k0)
        add_schoolbook(2, 3, k1)
        # j = 4
        add_schoolbook(2, 4, k0)
        add_schoolbook(2, 4, k1)

    return {"U": U, "V": V, "W": W}
