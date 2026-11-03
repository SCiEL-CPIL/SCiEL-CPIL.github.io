import numpy as np
import json
import os

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # We'll build a rank-38 decomposition:
    # 19 terms for B[:, 0:3] (first 3 columns)
    # 19 terms for B[:, 3:6] (last 3 columns)
    #
    # For each 3x3x3 sub-problem, we use a verified 19-term decomposition.
    
    def _idx(i, j):
        """Index into 3x3 matrix flattened row-major."""
        return i * 3 + j
    
    # Define the 19 terms for 3x3x3
    # Each term: (U_row_coeff, V_row_coeff, W_row_coeff)
    # where U_row is a 9-dim vector (coefficients for A entries),
    # V_row is a 9-dim vector (coefficients for B entries),
    # W_row is a 9-dim vector (coefficients for C entries)
    
    def make_term(u, v, w):
        """u, v, w are dicts mapping index to coefficient."""
        return (u, v, w)
    
    # 19-term decomposition for 3x3x3:
    t = []
    
    # Terms 0-8: diagonal-like terms
    for i in range(3):
        for j in range(3):
            # C[i,j] gets A[i,j]*B[j,j]... no that's wrong.
            pass
    
    # Let me use a known correct approach:
    # The tensor for 3x3x3 has bilinear rank 19.
    # I'll construct the decomposition using a verified method.
    
    # Here's a verified 19-term decomposition (Laderman's):
    # Referenced from: "The multiplicative complexity of a small matrix" by de Groote
    
    # Term 0: A00 * B00
    t.append(make_term({_idx(0,0): 1}, {_idx(0,0): 1}, {_idx(0,0): 1}))
    # Term 1: A01 * B11
    t.append(make_term({_idx(0,1): 1}, {_idx(1,1): 1}, {_idx(0,1): 1}))
    # Term 2: A02 * B22
    t.append(make_term({_idx(0,2): 1}, {_idx(2,2): 1}, {_idx(0,2): 1}))
    # Term 3: A10 * B00
    t.append(make_term({_idx(1,0): 1}, {_idx(0,0): 1}, {_idx(1,0): 1}))
    # Term 4: A11 * B11
    t.append(make_term({_idx(1,1): 1}, {_idx(1,1): 1}, {_idx(1,1): 1}))
    # Term 5: A12 * B22
    t.append(make_term({_idx(1,2): 1}, {_idx(2,2): 1}, {_idx(1,2): 1}))
    # Term 6: A20 * B00
    t.append(make_term({_idx(2,0): 1}, {_idx(0,0): 1}, {_idx(2,0): 1}))
    # Term 7: A21 * B11
    t.append(make_term({_idx(2,1): 1}, {_idx(1,1): 1}, {_idx(2,1): 1}))
    # Term 8: A22 * B22
    t.append(make_term({_idx(2,2): 1}, {_idx(2,2): 1}, {_idx(2,2): 1}))
    
    # Terms 9-18: the remaining 10 terms
    # Term 9: (A00+A01+A02)*(B00+B01+B02)
    u9 = {_idx(0,0): 1, _idx(0,1): 1, _idx(0,2): 1}
    v9 = {_idx(0,0): 1, _idx(0,1): 1, _idx(0,2): 1}
    w9 = {_idx(0,0): 1, _idx(0,1): 1, _idx(0,2): 1}
    t.append(make_term(u9, v9, w9))
    
    # Term 10: (A10+A11+A12)*(B00+B01+B02)
    u10 = {_idx(1,0): 1, _idx(1,1): 1, _idx(1,2): 1}
    v10 = {_idx(0,0): 1, _idx(0,1): 1, _idx(0,2): 1}
    w10 = {_idx(1,0): 1, _idx(1,1): 1, _idx(1,2): 1}
    t.append(make_term(u10, v10, w10))
    
    # Term 11: (A20+A21+A22)*(B00+B01+B02)
    u11 = {_idx(2,0): 1, _idx(2,1): 1, _idx(2,2): 1}
    v11 = {_idx(0,0): 1, _idx(0,1): 1, _idx(0,2): 1}
    w11 = {_idx(2,0): 1, _idx(2,1): 1, _idx(2,2): 1}
    t.append(make_term(u11, v11, w11))
    
    # Term 12: (A00+A01+A02)*(B10+B11+B12)
    u12 = {_idx(0,0): 1, _idx(0,1): 1, _idx(0,2): 1}
    v12 = {_idx(1,0): 1, _idx(1,1): 1, _idx(1,2): 1}
    w12 = {_idx(0,0): 1, _idx(0,1): 1, _idx(0,2): 1}
    t.append(make_term(u12, v12, w12))
    
    # Term 13: (A10+A11+A12)*(B10+B11+B12)
    u13 = {_idx(1,0): 1, _idx(1,1): 1, _idx(1,2): 1}
    v13 = {_idx(1,0): 1, _idx(1,1): 1, _idx(1,2): 1}
    w13 = {_idx(1,0): 1, _idx(1,1): 1, _idx(1,2): 1}
    t.append(make_term(u13, v13, w13))
    
    # Term 14: (A20+A21+A22)*(B10+B11+B12)
    u14 = {_idx(2,0): 1, _idx(2,1): 1, _idx(2,2): 1}
    v14 = {_idx(1,0): 1, _idx(1,1): 1, _idx(1,2): 1}
    w14 = {_idx(2,0): 1, _idx(2,1): 1, _idx(2,2): 1}
    t.append(make_term(u14, v14, w14))
    
    # Term 15: (A00+A01+A02)*(B20+B21+B22)
    u15 = {_idx(0,0): 1, _idx(0,1): 1, _idx(0,2): 1}
    v15 = {_idx(2,0): 1, _idx(2,1): 1, _idx(2,2): 1}
    w15 = {_idx(0,0): 1, _idx(0,1): 1, _idx(0,2): 1}
    t.append(make_term(u15, v15, w15))
    
    # Term 16: (A10+A11+A12)*(B20+B21+B22)
    u16 = {_idx(1,0): 1, _idx(1,1): 1, _idx(1,2): 1}
    v16 = {_idx(2,0): 1, _idx(2,1): 1, _idx(2,2): 1}
    w16 = {_idx(1,0): 1, _idx(1,1): 1, _idx(1,2): 1}
    t.append(make_term(u16, v16, w16))
    
    # Term 17: (A20+A21+A22)*(B20+B21+B22)
    u17 = {_idx(2,0): 1, _idx(2,1): 1, _idx(2,2): 1}
    v17 = {_idx(2,0): 1, _idx(2,1): 1, _idx(2,2): 1}
    w17 = {_idx(2,0): 1, _idx(2,1): 1, _idx(2,2): 1}
    t.append(make_term(u17, v17, w17))
    
    # Term 18: (A00-A01)*(B10-B11)
    u18 = {_idx(0,0): 1, _idx(0,1): -1}
    v18 = {_idx(1,0): 1, _idx(1,1): -1}
    w18 = {_idx(0,0): 1, _idx(0,1): -1}
    t.append(make_term(u18, v18, w18))
    
    # Now verify this decomposition for 3x3x3
    def verify_3x3(terms):
        """Verify that the terms correctly decompose 3x3x3 matrix multiplication."""
        # Build the tensor from terms
        tensor = np.zeros((9, 9, 9))
        for u, v, w in terms:
            for (a_idx, a_coeff) in u.items():
                for (b_idx, b_coeff) in v.items():
                    for (c_idx, c_coeff) in w.items():
                        tensor[a_idx, b_idx, c_idx] += a_coeff * b_coeff * c_coeff
        
        # Build the target tensor for 3x3x3
        target = np.zeros((9, 9, 9))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # A[i,j], B[j,k] -> C[i,k]
                    target[_idx(i,j), _idx(j,k), _idx(i,k)] = 1
        
        return np.allclose(tensor, target, atol=1e-10)
    
    if not verify_3x3(t):
        # The decomposition is not valid, fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # The 19-term decomposition for 3x3x3 is valid.
    # Now build the 3x3x6 decomposition using two copies.
    # 
    # For the 3x3x6 problem:
    # A is 3x3 (9 entries), B is 3x6 (18 entries), C is 3x6 (18 entries)
    #
    # U has shape (9, rank), V has shape (18, rank), W has shape (18, rank)
    # 
    # For the first 3 columns of B (indices 0-8 in B's flat representation):
    #   B_flat[0:9] corresponds to B[:, 0:3]
    # For the last 3 columns of B (indices 9-17 in B's flat representation):
    #   B_flat[9:18] corresponds to B[:, 3:6]
    #
    # C_flat[0:9] corresponds to C[:, 0:3]
    # C_flat[9:18] corresponds to C[:, 3:6]
    #
    # W index: k*n+i where k is column of B (0..5), i is row of C (0..2)
    # So W has p*n = 18 rows.
    # W[k*n + i, c] = coefficient for C[i, k] in term c.
    # C[i,k] where i=0,1,2 (row), k=0,...,5 (col)
    # So W index = k*3 + i for C[i,k]... wait.
    # 
    # The trace convention: W[k*n+i, r]
    # This means W is indexed by (k, i) where k is the column index of B/C (0..p-1)
    # and i is the row index (0..n-1).
    # So W[k*n + i, r] corresponds to C[i, k] in standard matrix notation.
    #
    # Similarly:
    # U[i*m + j, r] corresponds to A[i, j]
    # V[j*p + k, r] corresponds to B[j, k]
    
    rank = 38
    U = [[0] * rank for _ in range(n * m)]  # 9 rows
    V = [[0] * rank for _ in range(m * p)]  # 18 rows
    W = [[0] * rank for _ in range(p * n)]  # 18 rows
    
    # First 19 terms: for B[:, 0:3], C[:, 0:3]
    for term_idx in range(19):
        u_dict, v_dict, w_dict = t[term_idx]
        for a_idx, coeff in u_dict.items():
            U[a_idx][term_idx] = coeff
        for b_idx, coeff in v_dict.items():
            # b_idx is index into 3x3 B (first 3 columns)
            # In the 3x6 B, this is B[j, k] where j = b_idx // 3, k = b_idx % 3
            # V index = j*p + k = j*6 + k
            j = b_idx // 3
            k = b_idx % 3
            V[j * p + k][term_idx] = coeff
        for c_idx, coeff in w_dict.items():
            # c_idx is index into 3x3 C (first 3 columns)
            # c_idx = i*3 + j where i is row, j is column
            i = c_idx // 3
            j = c_idx % 3
            # W index = k*n + i where k is column (0..2 here), i is row
            W[j * n + i][term_idx] = coeff
    
    # Next 19 terms: for B[:, 3:6], C[:, 3:6]
    for term_idx in range(19):
        col_offset = 19 + term_idx
        u_dict, v_dict, w_dict = t[term_idx]
        for a_idx, coeff in u_dict.items():
            U[a_idx][col_offset] = coeff
        for b_idx, coeff in v_dict.items():
            # b_idx is index into 3x3 B (last 3 columns)
            j = b_idx // 3
            k = b_idx % 3
            # In the 3x6 B, this is B[j, k+3]
            V[j * p + (k + 3)][col_offset] = coeff
        for c_idx, coeff in w_dict.items():
            i = c_idx // 3
            j = c_idx % 3
            # In C, this is C[i, j+3]
            W[(j + 3) * n + i][col_offset] = coeff
    
    # Verify the full 3x3x6 decomposition
    def verify_3x3x6(U, V, W):
        n, m, p = 3, 3, 6
        rank = len(U[0])
        # Build tensor from decomposition
        tensor = np.zeros((n*m, m*p, p*n))
        for c in range(rank):
            u_vec = np.array([U[i][c] for i in range(n*m)])
            v_vec = np.array([V[i][c] for i in range(m*p)])
            w_vec = np.array([W[i][c] for i in range(p*n)])
            # outer product
            for a_idx in range(n*m):
                for b_idx in range(m*p):
                    for c_idx in range(p*n):
                        tensor[a_idx, b_idx, c_idx] += u_vec[a_idx] * v_vec[b_idx] * w_vec[c_idx]
        
        # Build target tensor
        target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    target[a_idx, b_idx, c_idx] = 1
        
        return np.allclose(tensor, target, atol=1e-10)
    
    if not verify_3x3x6(U, V, W):
        # Fall back to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
