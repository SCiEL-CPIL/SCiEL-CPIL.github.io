"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=7
    # Output index: k*n + i  (k=0..p-1, i=0..n-1)
    # Input A index: i*m + j (i=0..n-1, j=0..m-1)
    # Input B index: j*p + k (j=0..m-1, k=0..p-1)
    
    # We will build a rank-39 decomposition using Strassen for 2x2x2 blocks.
    # A is 2x3, B is 3x7, C is 2x7.
    # Split A into A_L (2x2) and A_R (2x1).
    # Split B into B_L (2x6) and B_R (1x7).
    # C = A_L * B_L + A_R * B_R
    # A_L * B_L: 2x2 * 2x6 -> 2x6.
    # We can view B_L as three 2x2 blocks: B_L[:, 0:2], B_L[:, 2:4], B_L[:, 4:6].
    # Each 2x2 * 2x2 multiplication can be done with rank 7 (Strassen).
    # So A_L * B_L requires 3 * 7 = 21 terms.
    # A_R * B_R: 2x1 * 1x7 -> 2x7.
    # This is just outer product of column of A_R and row of B_R? No.
    # A_R is 2x1, B_R is 1x7. C_R = A_R * B_R is 2x7.
    # C_R[i, k] = A_R[i, 0] * B_R[0, k].
    # This requires 2*7 = 14 terms in schoolbook.
    # Total rank = 21 + 14 = 35? Wait.
    # Let's re-verify the Strassen application.
    # Strassen multiplies two 2x2 matrices.
    # Here A_L is 2x2. B_L is 2x6.
    # We can multiply A_L (2x2) by each 2x2 block of B_L.
    # Let B_L1 = B_L[:, 0:2], B_L2 = B_L[:, 2:4], B_L3 = B_L[:, 4:6].
    # C_L1 = A_L * B_L1 (2x2)
    # C_L2 = A_L * B_L2 (2x2)
    # C_L3 = A_L * B_L3 (2x2)
    # Each of these is a 2x2x2 tensor. Rank 7 each.
    # So 21 terms for the left part.
    
    # Now the right part: A_R (2x1) * B_R (1x7).
    # A_R has indices i=0,1; col j=2. So A[0,2] and A[1,2].
    # B_R has row j=2; cols k=0..6. So B[2,0]...B[2,6].
    # C_R[i, k] = A[i, 2] * B[2, k].
    # Terms:
    # C[0,0] += A[0,2] B[2,0]
    # C[0,1] += A[0,2] B[2,1]
    # ...
    # C[0,6] += A[0,2] B[2,6]
    # C[1,0] += A[1,2] B[2,0]
    # ...
    # C[1,6] += A[1,2] B[2,6]
    # Total 14 terms.
    
    # Is it possible to do better on the right part?
    # The tensor for 2x1 * 1x7 is essentially a sum of 14 rank-1 tensors.
    # The rank of this tensor is 14?
    # T[i, k] = A[i] * B[k].
    # This is the outer product of vector A (size 2) and vector B (size 7).
    # The tensor is in A-space (2) x B-space (7) x C-space (14? no, C is 2x7=14 dims).
    # Wait, the tensor is defined on indices (a, b, c).
    # a = i*m + j. For right part, j=2. a = i*3 + 2. i=0,1. So a in {2, 5}.
    # b = j*p + k. For right part, j=2. b = 2*7 + k = 14 + k. k=0..6. So b in {14..20}.
    # c = k*n + i. k=0..6, i=0,1.
    # The non-zero entries of T are T[a, b, c] = 1 if a=i*3+2, b=14+k, c=k*2+i.
    # Let's check if this sub-tensor has rank less than 14.
    # It's a "matrix multiplication" of 2x1 and 1x7.
    # The rank of the tensor for m x 1 * 1 x p is m*p.
    # Why? Because the slices are independent in a way that prevents cancellation?
    # Actually, for 2x1 * 1x7, the tensor is sum_{i,k} e_i (in A) e_k (in B) e_{ik} (in C).
    # This is a set of 14 orthogonal-ish vectors?
    # In the A-mode, we have e_0 and e_1.
    # In the B-mode, we have e_0 ... e_6.
    # In the C-mode, we have e_{00}, e_{01}, ..., e_{06}, e_{10}, ..., e_{16}.
    # The terms are e_i \otimes e_k \otimes e_{ik}.
    # Since all e_{ik} are distinct basis vectors, the rank is exactly 14.
    # So 14 is minimal for that part.
    
    # What about the left part?
    # A_L (2x2) * B_L (2x6).
    # We split B_L into three 2x2 blocks.
    # Each block multiplication is 2x2x2. Rank 7.
    # Total 21.
    # Can we do better?
    # The tensor is A_L * B_L.
    # This is equivalent to multiplying a 2x2 matrix by a 2x6 matrix.
    # This can be seen as 3 independent 2x2x2 multiplications.
    # Since the output columns are disjoint (C_L1 uses cols 0,1; C_L2 uses 2,3; C_L3 uses 4,5),
    # the rank is the sum of the ranks of the individual 2x2x2 tensors.
    # Rank(2x2x2) = 7 (Strassen).
    # So 21 is minimal for this part.
    
    # Total rank = 21 + 14 = 35.
    
    # Let's double check if there's any overlap or sharing possible.
    # The A-indices used in Left part: i in {0,1}, j in {0,1}. a in {0,1,3,4}.
    # The A-indices used in Right part: i in {0,1}, j=2. a in {2,5}.
    # Disjoint.
    # The B-indices used in Left part: j in {0,1}, k in {0..5}.
    # j=0: b in {0..5}. j=1: b in {7..12}.
    # The B-indices used in Right part: j=2, k in {0..6}.
    # b in {14..20}.
    # Disjoint.
    # The C-indices used in Left part:
    # Block 1 (cols 0,1): k in {0,1}. c = k*2+i.
    # k=0: c in {0,1}. k=1: c in {2,3}.
    # Block 2 (cols 2,3): k in {2,3}.
    # k=2: c in {4,5}. k=3: c in {6,7}.
    # Block 3 (cols 4,5): k in {4,5}.
    # k=4: c in {8,9}. k=5: c in {10,11}.
    # The C-indices used in Right part:
    # k in {0..6}. c = k*2+i.
    # k=0: c in {0,1}.
    # k=1: c in {2,3}.
    # ...
    # k=6: c in {12,13}.
    
    # Wait. The C-indices overlap!
    # Left part produces C entries for k=0..5.
    # Right part produces C entries for k=0..6.
    # Specifically, for k=0..5, both parts contribute to C[i, k].
    # C[i, k] = (A_L * B_L)[i, k] + (A_R * B_R)[i, k].
    # So we cannot just sum the ranks. We have to combine them.
    # However, since the A and B indices are disjoint, the terms are linearly independent.
    # Term from Left: U_L[a, r] V_L[b, r] W_L[c, r].
    # Term from Right: U_R[a', r'] V_R[b', r'] W_R[c', r'].
    # a != a', b != b'.
    # If we sum them, do we get a lower rank?
    # Generally, if the supports in A and B modes are disjoint, the rank is additive.
    # Let's verify.
    # T = T_L + T_R.
    # supp_A(T_L) = {0,1,3,4}. supp_A(T_R) = {2,5}.
    # supp_B(T_L) = {0..5, 7..12}. supp_B(T_R) = {14..20}.
    # Since supp_A and supp_B are disjoint for L and R, the rank is rank(T_L) + rank(T_R).
    # This is a standard property of tensor rank for disjoint supports in at least one mode?
    # Actually, if supp_A(T_L) \cap supp_A(T_R) = empty, then rank(T_L + T_R) = rank(T_L) + rank(T_R).
    # Yes, because any decomposition of T_L + T_R can be split into components supported on the disjoint sets.
    # So rank = 21 + 14 = 35.
    
    # So the target rank is 35.
    
    # Let's construct the 35 terms.
    
    # Strassen 2x2x2 terms:
    # M1 = A00 * (B00 + B11)
    # M2 = (A00 + A10) * B11
    # M3 = (A01 + A11) * B00
    # M4 = A11 * (B01 - B10)
    # M5 = (A00 + A01) * (B10 - B11)
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A11 - A01) * (B01 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M4
    
    # We need to map this to the global indices.
    
    U = np.zeros((n*m, 35), dtype=np.int64)
    V = np.zeros((m*p, 35), dtype=np.int64)
    W = np.zeros((p*n, 35), dtype=np.int64)
    
    col = 0
    
    # Helper to add Strassen terms for a specific block
    # a_indices: [a00, a01, a10, a11]
    # b_indices: [b00, b01, b10, b11]
    # c_indices: [c00, c01, c10, c11]
    # Note: Strassen formulas assume A is 2x2, B is 2x2, C is 2x2.
    # A[i,j] is at a_indices[i*2+j]? No, just a list.
    # Let's define a_indices as [A00, A01, A10, A11]
    # b_indices as [B00, B01, B10, B11]
    # c_indices as [C00, C01, C10, C11]
    
    def add_strassen(a_idx, b_idx, c_idx, start_col):
        # M1 = A00 * (B00 + B11)
        U[a_idx[0], start_col] = 1
        V[b_idx[0], start_col] = 1
        V[b_idx[3], start_col] = 1
        W[c_idx[0], start_col] = 1
        W[c_idx[3], start_col] = -1  # C11 gets -M1? No.
        # Let's re-verify Strassen C formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M6
        # C11 = M1 - M2 + M3 + M4
        
        # M1: A00(B00+B11).
        # Contributes to C00 (+1), C11 (+1).
        U[a_idx[0], start_col] = 1
        V[b_idx[0], start_col] = 1
        V[b_idx[3], start_col] = 1
        W[c_idx[0], start_col] = 1
        W[c_idx[3], start_col] = 1
        
        # M2: (A00+A10)B11
        # Contributes to C10 (+1), C11 (-1).
        col = start_col + 1
        U[a_idx[0], col] = 1
        U[a_idx[2], col] = 1
        V[b_idx[3], col] = 1
        W[c_idx[2], col] = 1
        W[c_idx[3], col] = -1
        
        # M3: (A01+A11)B00
        # Contributes to C01 (+1), C11 (+1).
        col = start_col + 2
        U[a_idx[1], col] = 1
        U[a_idx[3], col] = 1
        V[b_idx[0], col] = 1
        W[c_idx[1], col] = 1
        W[c_idx[3], col] = 1
        
        # M4: A11(B01-B10)
        # Contributes to C00 (+1), C11 (+1).
        col = start_col + 3
        U[a_idx[3], col] = 1
        V[b_idx[1], col] = 1
        V[b_idx[2], col] = -1
        W[c_idx[0], col] = 1
        W[c_idx[3], col] = 1
        
        # M5: (A00+A01)(B10-B11)
        # Contributes to C00 (-1), C01 (+1).
        col = start_col + 4
        U[a_idx[0], col] = 1
        U[a_idx[1], col] = 1
        V[b_idx[2], col] = 1
        V[b_idx[3], col] = -1
        W[c_idx[0], col] = -1
        W[c_idx[1], col] = 1
        
        # M6: (A10-A00)(B00+B10)
        # Contributes to C10 (+1).
        col = start_col + 5
        U[a_idx[2], col] = 1
        U[a_idx[0], col] = -1
        V[b_idx[0], col] = 1
        V[b_idx[2], col] = 1
        W[c_idx[2], col] = 1
        
        # M7: (A11-A01)(B01+B11)
        # Contributes to C00 (+1).
        col = start_col + 6
        U[a_idx[3], col] = 1
        U[a_idx[1], col] = -1
        V[b_idx[1], col] = 1
        V[b_idx[3], col] = 1
        W[c_idx[0], col] = 1
        
        return start_col + 7
    
    # Block 1: A_L (cols 0,1) * B_L (cols 0,1) -> C (cols 0,1)
    # A indices:
    # A00: i=0, j=0 -> 0*3+0 = 0
    # A01: i=0, j=1 -> 0*3+1 = 1
    # A10: i=1, j=0 -> 1*3+0 = 3
    # A11: i=1, j=1 -> 1*3+1 = 4
    a_idx_1 = [0, 1, 3, 4]
    
    # B indices:
    # B00: j=0, k=0 -> 0*7+0 = 0
    # B01: j=0, k=1 -> 0*7+1 = 1
    # B10: j=1, k=0 -> 1*7+0 = 7
    # B11: j=1, k=1 -> 1*7+1 = 8
    b_idx_1 = [0, 1, 7, 8]
    
    # C indices:
    # C00: k=0, i=0 -> 0*2+0 = 0
    # C01: k=1, i=0 -> 1*2+0 = 2
    # C10: k=0, i=1 -> 0*2+1 = 1
    # C11: k=1, i=1 -> 1*2+1 = 3
    c_idx_1 = [0, 2, 1, 3]
    
    col = add_strassen(a_idx_1, b_idx_1, c_idx_1, col)
    
    # Block 2: A_L (cols 0,1) * B_L (cols 2,3) -> C (cols 2,3)
    a_idx_2 = [0, 1, 3, 4]
    
    # B indices:
    # B00: j=0, k=2 -> 0*7+2 = 2
    # B01: j=0, k=3 -> 0*7+3 = 3
    # B10: j=1, k=2 -> 1*7+2 = 9
    # B11: j=1, k=3 -> 1*7+3 = 10
    b_idx_2 = [2, 3, 9, 10]
    
    # C indices:
    # C00: k=2, i=0 -> 2*2+0 = 4
    # C01: k=3, i=0 -> 3*2+0 = 6
    # C10: k=2, i=1 -> 2*2+1 = 5
    # C11: k=3, i=1 -> 3*2+1 = 7
    c_idx_2 = [4, 6, 5, 7]
    
    col = add_strassen(a_idx_2, b_idx_2, c_idx_2, col)
    
    # Block 3: A_L (cols 0,1) * B_L (cols 4,5) -> C (cols 4,5)
    a_idx_3 = [0, 1, 3, 4]
    
    # B indices:
    # B00: j=0, k=4 -> 0*7+4 = 4
    # B01: j=0, k=5 -> 0*7+5 = 5
    # B10: j=1, k=4 -> 1*7+4 = 11
    # B11: j=1, k=5 -> 1*7+5 = 12
    b_idx_3 = [4, 5, 11, 12]
    
    # C indices:
    # C00: k=4, i=0 -> 4*2+0 = 8
    # C01: k=5, i=0 -> 5*2+0 = 10
    # C10: k=4, i=1 -> 4*2+1 = 9
    # C11: k=5, i=1 -> 5*2+1 = 11
    c_idx_3 = [8, 10, 9, 11]
    
    col = add_strassen(a_idx_3, b_idx_3, c_idx_3, col)
    
    # Now the Right Part: A_R (col 2) * B_R (row 2)
    # A indices:
    # A0: i=0, j=2 -> 0*3+2 = 2
    # A1: i=1, j=2 -> 1*3+2 = 5
    
    # B indices:
    # B0: j=2, k=0 -> 2*7+0 = 14
    # B1: j=2, k=1 -> 2*7+1 = 15
    # ...
    # B6: j=2, k=6 -> 2*7+6 = 20
    
    # C indices:
    # C[i, k] = k*2 + i
    
    # Terms:
    # For i=0:
    #   C[0, k] += A[0,2] * B[2, k]
    # For i=1:
    #   C[1, k] += A[1,2] * B[2, k]
    
    for k in range(7):
        # Term for i=0
        U[2, col] = 1
        V[14 + k, col] = 1
        W[k*2 + 0, col] = 1
        col += 1
        
        # Term for i=1
        U[5, col] = 1
        V[14 + k, col] = 1
        W[k*2 + 1, col] = 1
        col += 1
    
    # Total col should be 21 + 14 = 35.
    if col != 35:
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.int64)
        V = np.zeros((m*p, rank), dtype=np.int64)
        W = np.zeros((p*n, rank), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1
                    V[j*p+k, c] = 1
                    W[k*n+i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
