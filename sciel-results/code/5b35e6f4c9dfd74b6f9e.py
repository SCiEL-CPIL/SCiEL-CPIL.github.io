import numpy as np
import time

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a rank-60 exact tensor decomposition for 3x3 * 3x7 matrix multiplication.
    Uses Strassen's algorithm for the 2x2 x 2x2 sub-blocks and schoolbook for the rest.
    """
    n, m, p = DIMENSIONS
    # Total rank is 60
    rank = 60
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1.0):
        nonlocal col
        if col >= rank:
            return
        U[u_idx, col] += val
        V[v_idx, col] += val
        W[w_idx, col] += val
        col += 1

    # Helper to add Strassen terms for 2x2 * 2x2
    # A indices: a00, a01, a10, a11
    # B indices: b00, b01, b10, b11
    # C indices: c00, c01, c10, c11
    def add_strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        nonlocal col
        # M1 = (a00 + a11) * (b00 + b11)
        add_term(a00, b00, c00, 1.0)
        add_term(a00, b11, c00, 1.0)
        add_term(a11, b00, c00, 1.0)
        add_term(a11, b11, c00, 1.0)
        # This is not rank 1, it's a product of sums.
        # We need to expand the products of sums into rank-1 terms?
        # No, Strassen is a decomposition of the tensor.
        # The tensor entries are coefficients.
        # Strassen decomposition:
        # M1 = (a00+a11)(b00+b11) -> contributes to c00, c11
        # M2 = (a10+a11)b00 -> contributes to c10, c11
        # M3 = a00(b01-b11) -> contributes to c01, c11
        # M4 = a11(b10-b00) -> contributes to c00, c10
        # M5 = (a00+a01)b11 -> contributes to c00, c01
        # M6 = (a10-a00)(b00+b01) -> contributes to c10, c11
        # M7 = (a01-a11)(b10+b11) -> contributes to c00, c11
        
        # We need to represent these linear combinations in U, V, W.
        # This requires expanding the "sums" in A and B into multiple rank-1 terms?
        # No, the rank is 7 because we share the products.
        # But U, V, W are matrices.
        # For M1:
        # We need U to pick (a00+a11). This is not a single index.
        # We must split M1 into 4 rank-1 terms if we can't share?
        # If we split, rank becomes 8 (schoolbook).
        # To get rank 7, we must use the structure.
        # But U, V, W are just arrays of numbers.
        # The decomposition is T = sum U_r V_r W_r.
        # This implies we are looking for 7 vectors u_r, v_r, w_r.
        # Strassen's algorithm provides 7 "multiplications" M1..M7.
        # Each M_i is a product of a linear form in A and a linear form in B.
        # Let L_A_i = sum u_ij A_ij and L_B_i = sum v_jk B_jk.
        # Then C_lk = sum_i w_ik L_A_i L_B_i.
        # This fits the tensor decomposition form if we define:
        # U[i, r] = coefficient of A_i in L_A_r
        # V[j, r] = coefficient of B_j in L_B_r
        # W[k, r] = coefficient of C_k in the sum?
        # Wait, W is indexed by C.
        # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
        # Yes, this matches.
        
        # So we need to define the 7 linear forms.
        
        # M1: L_A = a00 + a11, L_B = b00 + b11.
        # U: a00->1, a11->1. V: b00->1, b11->1.
        # W: c00 gets +1, c11 gets +1.
        # So we add 1 term to the decomposition for this "multiplication"?
        # No, the term in the sum is U[a,r] V[b,r] W[c,r].
        # If we set r=0 for M1:
        # U[a00, 0] = 1, U[a11, 0] = 1.
        # V[b00, 0] = 1, V[b11, 0] = 1.
        # W[c00, 0] = 1, W[c11, 0] = 1.
        # This single column r=0 generates:
        # T[a00, b00, c00] += 1*1*1 = 1. Correct.
        # T[a00, b11, c00] += 1*1*1 = 1. Correct.
        # T[a11, b00, c00] += 1*1*1 = 1. Correct.
        # T[a11, b11, c00] += 1*1*1 = 1. Correct.
        # T[a00, b00, c11] += 1*1*1 = 1. Correct.
        # ... and so on.
        # So one column in U, V, W represents one "Strassen multiplication".
        
        # We need to do this for all 7.
        
        # M1
        U[a00, col] = 1; U[a11, col] = 1
        V[b00, col] = 1; V[b11, col] = 1
        W[c00, col] = 1; W[c11, col] = 1
        col += 1
        
        # M2: (a10 + a11) * b00
        U[a10, col] = 1; U[a11, col] = 1
        V[b00, col] = 1
        W[c10, col] = 1; W[c11, col] = 1
        col += 1
        
        # M3: a00 * (b01 - b11)
        U[a00, col] = 1
        V[b01, col] = 1; V[b11, col] = -1
        W[c01, col] = 1; W[c11, col] = 1
        col += 1
        
        # M4: a11 * (b10 - b00)
        U[a11, col] = 1
        V[b10, col] = 1; V[b00, col] = -1
        W[c00, col] = 1; W[c10, col] = 1
        col += 1
        
        # M5: (a00 + a01) * b11
        U[a00, col] = 1; U[a01, col] = 1
        V[b11, col] = 1
        W[c00, col] = 1; W[c01, col] = 1
        col += 1
        
        # M6: (a10 - a00) * (b00 + b01)
        U[a10, col] = 1; U[a00, col] = -1
        V[b00, col] = 1; V[b01, col] = 1
        W[c10, col] = 1; W[c11, col] = 1
        col += 1
        
        # M7: (a01 - a11) * (b10 + b11)
        U[a01, col] = 1; U[a11, col] = -1
        V[b10, col] = 1; V[b11, col] = 1
        W[c00, col] = 1; W[c11, col] = 1
        col += 1

    # Indices for A (3x3)
    # 0 1 2
    # 3 4 5
    # 6 7 8
    
    # Indices for B (3x7)
    # 0 1 2 3 4 5 6
    # 7 8 9 10 11 12 13
    # 14 15 16 17 18 19 20
    
    # Indices for C (3x7) trace k*n+i
    # k=0: 0 1 2
    # k=1: 3 4 5
    # k=2: 6 7 8
    # k=3: 9 10 11
    # k=4: 12 13 14
    # k=5: 15 16 17
    # k=6: 18 19 20

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    # A: 0, 1, 3, 4
    # B: 0, 1, 7, 8
    # C: 0, 1, 3, 4
    add_strassen(0, 1, 3, 4, 0, 1, 7, 8, 0, 1, 3, 4)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    # B: 2, 3, 9, 10
    # C: 2, 3, 5, 6
    add_strassen(0, 1, 3, 4, 2, 3, 9, 10, 2, 3, 5, 6)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:6]
    # B: 4, 5, 11, 12
    # C: 4, 5, 7, 8
    add_strassen(0, 1, 3, 4, 4, 5, 11, 12, 4, 5, 7, 8)
    
    # Block 4: A[0:2, 0:2] * B[0:2, 6:7] (Schoolbook)
    # B: 6, 13
    # C: 6, 9
    # Terms:
    # A00 * B06 -> C06
    U[0, col] = 1; V[6, col] = 1; W[6, col] = 1; col += 1
    # A01 * B16 -> C06? No, A[0,1] * B[1,6]
    U[1, col] = 1; V[13, col] = 1; W[6, col] = 1; col += 1
    # A10 * B06 -> C16
    U[3, col] = 1; V[6, col] = 1; W[9, col] = 1; col += 1
    # A11 * B16 -> C16
    U[4, col] = 1; V[13, col] = 1; W[9, col] = 1; col += 1
    
    # Block 5: a_col (A[0:2, 2]) * b_row (B[2, 0:7])
    # A: 2, 5
    # B: 14..20
    # C: 0..6 (row 0) and 3..9 (row 1)
    for k in range(7):
        # A02 * B2k -> C0k
        U[2, col] = 1; V[14+k, col] = 1; W[k, col] = 1; col += 1
        # A12 * B2k -> C1k
        U[5, col] = 1; V[14+k, col] = 1; W[k+3, col] = 1; col += 1
        
    # Block 6: a_row (A[2, 0:2]) * B[0:2, 0:7]
    # A: 6, 7
    # B: 0..13
    # C: 6..20 (row 2)
    for k in range(7):
        # A20 * B0k -> C2k
        U[6, col] = 1; V[k, col] = 1; W[k+6, col] = 1; col += 1
        # A21 * B1k -> C2k
        U[7, col] = 1; V[k+7, col] = 1; W[k+6, col] = 1; col += 1
        
    # Block 7: a33 (A[2, 2]) * b_row (B[2, 0:7])
    # A: 8
    # B: 14..20
    # C: 6..20 (row 2)
    for k in range(7):
        # A22 * B2k -> C2k
        U[8, col] = 1; V[14+k, col] = 1; W[k+6, col] = 1; col += 1

    if col != rank:
        # Adjust if my count was wrong
        U = U[:, :col]
        V = V[:, :col]
        W = W[:, :col]
        rank = col

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
