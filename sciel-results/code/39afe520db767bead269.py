import numpy as np
import time
import fractions

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 2, 3, 4
    
    # We will attempt to find a decomposition of rank 22.
    # Structure:
    # A is 2x3, B is 3x4.
    # Split A into A1 (2x2) and A2 (2x1).
    # Split B into B1 (2x4) and B2 (1x4).
    # C = A1 B1 + A2 B2.
    
    # A1 B1 (2x2 * 2x4):
    # Split B1 into B11 (2x2) and B12 (2x2).
    # A1 B11: Rank 7 (Strassen)
    # A1 B12: Rank 7 (Strassen)
    # A2 B2: Rank 8 (Schoolbook for 2x1x4)
    # Total Rank = 7 + 7 + 8 = 22.
    
    # Indices:
    # U is n*m x R = 6 x R. Index u = i*m + j = i*3 + j.
    # V is m*p x R = 12 x R. Index v = j*p + k = j*4 + k.
    # W is p*n x R = 8 x R. Index w = k*n + i = k*2 + i.
    
    R = 22
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    def idx_u(i, j):
        return i * m + j
        
    def idx_v(j, k):
        return j * p + k
        
    def idx_w(k, i):
        return k * n + i
        
    # Strassen Block Definition for 2x2 matrices.
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = AB
    #
    # M1 = A00 * (B00 + B11)
    # M2 = (A00 + A10) * B11
    # M3 = A11 * (B00 + B01)
    # M4 = (A11 + A00) * (B11 + B10)
    # M5 = (A00 + A11) * (B00 + B01)
    # M6 = (A00 - A11) * (B10 + B11)
    # M7 = (A01 - A11) * (B00 + B01)
    #
    # C00 = M1 + M4 - M2 + M6
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 + M7
    
    # Each term r is defined by coefficients u_ij, v_jk, w_ki.
    # sum_r u_ij * v_jk * w_ki
    
    # Term 0: M1
    # u: A00 (i=0, j=0) -> 1
    # v: B00 (j=0, k=0) -> 1, B11 (j=1, k=1) -> 1
    # w: C00 (k=0, i=0) -> 1
    r = 0
    U[idx_u(0,0), r] = 1
    V[idx_v(0,0), r] = 1
    V[idx_v(1,1), r] = 1
    W[idx_w(0,0), r] = 1
    
    # Term 1: M2
    # u: A00 (0,0)->1, A10 (1,0)->1
    # v: B11 (1,1)->1
    # w: C10 (k=0, i=1) -> 1
    r = 1
    U[idx_u(0,0), r] = 1
    U[idx_u(1,0), r] = 1
    V[idx_v(1,1), r] = 1
    W[idx_w(0,1), r] = 1
    
    # Term 2: M3
    # u: A11 (1,1)->1
    # v: B00 (0,0)->1, B01 (0,1)->1
    # w: C01 (k=1, i=0) -> 1
    r = 2
    U[idx_u(1,1), r] = 1
    V[idx_v(0,0), r] = 1
    V[idx_v(0,1), r] = 1
    W[idx_w(1,0), r] = 1
    
    # Term 3: M4
    # u: A11 (1,1)->1, A00 (0,0)->1
    # v: B11 (1,1)->1, B10 (1,0)->1
    # w: C00 (0,0)->1, C10 (0,1)->1, C11 (1,1)->1
    r = 3
    U[idx_u(1,1), r] = 1
    U[idx_u(0,0), r] = 1
    V[idx_v(1,1), r] = 1
    V[idx_v(1,0), r] = 1
    W[idx_w(0,0), r] = 1
    W[idx_w(0,1), r] = 1
    W[idx_w(1,1), r] = 1
    
    # Term 4: M5
    # u: A00 (0,0)->1, A11 (1,1)->1
    # v: B00 (0,0)->1, B01 (0,1)->1
    # w: C01 (1,0)->1, C11 (1,1)->1
    r = 4
    U[idx_u(0,0), r] = 1
    U[idx_u(1,1), r] = 1
    V[idx_v(0,0), r] = 1
    V[idx_v(0,1), r] = 1
    W[idx_w(1,0), r] = 1
    W[idx_w(1,1), r] = 1
    
    # Term 5: M6
    # u: A00 (0,0)->1, A11 (1,1)->-1
    # v: B10 (1,0)->1, B11 (1,1)->1
    # w: C00 (0,0)->1
    r = 5
    U[idx_u(0,0), r] = 1
    U[idx_u(1,1), r] = -1
    V[idx_v(1,0), r] = 1
    V[idx_v(1,1), r] = 1
    W[idx_w(0,0), r] = 1
    
    # Term 6: M7
    # u: A01 (0,1)->1, A11 (1,1)->-1
    # v: B00 (0,0)->1, B01 (0,1)->1
    # w: C11 (1,1)->1
    r = 6
    U[idx_u(0,1), r] = 1
    U[idx_u(1,1), r] = -1
    V[idx_v(0,0), r] = 1
    V[idx_v(0,1), r] = 1
    W[idx_w(1,1), r] = 1
    
    # Now we need to map these to the two 2x2x2 blocks.
    # The definitions above assumed local indices.
    # We defined them for the first block where A is rows 0-1, cols 0-1; B is rows 0-1, cols 0-1; C is rows 0-1, cols 0-1.
    # This corresponds to A1 B11.
    # Indices in this block:
    # A: i in {0,1}, j in {0,1}.
    # B: j in {0,1}, k in {0,1}.
    # C: i in {0,1}, k in {0,1}.
    #
    # Wait, the indices I used in idx_u, idx_v, idx_w are GLOBAL for the full problem.
    # Let's re-evaluate.
    # In the first block (A1 B11):
    # A part is A[0:2, 0:2]. So i in {0,1}, j in {0,1}.
    # B part is B[0:2, 0:2]. So j in {0,1}, k in {0,1}.
    # C part is C[0:2, 0:2]. So i in {0,1}, k in {0,1}.
    #
    # My previous assignments:
    # idx_u(0,0) -> 0. Correct for A[0,0].
    # idx_u(0,1) -> 1. Correct for A[0,1].
    # idx_u(1,0) -> 3. Correct for A[1,0]. (1*3+0)
    # idx_u(1,1) -> 4. Correct for A[1,1]. (1*3+1)
    #
    # idx_v(0,0) -> 0. Correct for B[0,0].
    # idx_v(0,1) -> 1. Correct for B[0,1].
    # idx_v(1,0) -> 4. Correct for B[1,0]. (1*4+0)
    # idx_v(1,1) -> 5. Correct for B[1,1]. (1*4+1)
    #
    # idx_w(0,0) -> 0. Correct for C[0,0] (k=0, i=0).
    # idx_w(0,1) -> 1. Correct for C[1,0] (k=0, i=1).
    # idx_w(1,0) -> 2. Correct for C[0,1] (k=1, i=0).
    # idx_w(1,1) -> 3. Correct for C[1,1] (k=1, i=1).
    #
    # So the assignments for r=0..6 are for the first block A1 B11.
    # This part is already in U, V, W.
    
    # Block 2: A1 B12
    # A part: i in {0,1}, j in {0,1}. (Same as before)
    # B part: j in {0,1}, k in {2,3}.
    # C part: i in {0,1}, k in {2,3}.
    #
    # We reuse the same U coefficients (since A is the same).
    # But V and W indices shift.
    # For B: j in {0,1}, k in {2,3}.
    #   B[0,2] -> idx_v(0,2) = 2
    #   B[0,3] -> idx_v(0,3) = 3
    #   B[1,2] -> idx_v(1,2) = 6
    #   B[1,3] -> idx_v(1,3) = 7
    #
    # For C: i in {0,1}, k in {2,3}.
    #   C[0,2] -> idx_w(2,0) = 4
    #   C[1,2] -> idx_w(2,1) = 5
    #   C[0,3] -> idx_w(3,0) = 6
    #   C[1,3] -> idx_w(3,1) = 7
    
    # We need to shift the local (j, k, i) indices of the previous block to the new ones.
    # Let's define a helper to replicate the Strassen terms with an offset for V and W.
    
    # Previous V indices (local j,k):
    # (0,0) -> 0, (0,1) -> 1, (1,0) -> 4, (1,1) -> 5.
    # New V indices for j in {0,1}, k in {2,3}:
    # (0,2) -> 2, (0,3) -> 3, (1,2) -> 6, (1,3) -> 7.
    # Mapping: old (0,0)->2, (0,1)->3, (1,0)->6, (1,1)->7.
    # Generally: new_v = j*4 + k.
    # Old k was 0 or 1. New k is old k + 2.
    
    # Previous W indices (local k, i):
    # (0,0) -> 0, (1,0) -> 2, (0,1) -> 1, (1,1) -> 3.
    # New W indices for k in {2,3}, i in {0,1}:
    # (2,0) -> 4, (3,0) -> 6, (2,1) -> 5, (3,1) -> 7.
    # Mapping: old (0,0)->4, (0,1)->5, (1,0)->6, (1,1)->7.
    # Generally: new_w = k*2 + i.
    # Old k was 0 or 1. New k is old k + 2.
    
    # Let's copy the U values from 0..6 to 7..13.
    U[0:6, 7:14] = U[0:6, 0:7]
    
    # Adjust V
    # We stored the structure in V[0:12, 0:7].
    # Let's iterate over the non-zero entries in V[0:12, 0:7] and shift them.
    # Or simply redefine them.
    for j in range(2):
        for k in range(2):
            # Old k is k. New k is k+2.
            old_idx = idx_v(j, k)
            new_idx = idx_v(j, k+2)
            V[new_idx, 7:14] = V[old_idx, 0:7]
            
    # Adjust W
    for k in range(2):
        for i in range(2):
            # Old k is k. New k is k+2.
            old_idx = idx_w(k, i)
            new_idx = idx_w(k+2, i)
            W[new_idx, 7:14] = W[old_idx, 0:7]
    
    # Block 3: A2 B2
    # A part: i in {0,1}, j in {2}.
    # B part: j in {2}, k in {0,1,2,3}.
    # C part: i in {0,1}, k in {0,1,2,3}.
    #
    # C_{i,k} = A_{i,2} B_{2,k}.
    # Schoolbook: Rank = 2*1*4 = 8.
    # Terms r = 14 .. 21.
    # For each i in {0,1}, k in {0,1,2,3}:
    #   U[idx_u(i,2), r] = 1
    #   V[idx_v(2,k), r] = 1
    #   W[idx_w(k,i), r] = 1
    
    r_idx = 14
    for i in range(2):
        for k in range(4):
            U[idx_u(i, 2), r_idx] = 1
            V[idx_v(2, k), r_idx] = 1
            W[idx_w(k, i), r_idx] = 1
            r_idx += 1
            
    # Verification
    # C_expected = A @ B
    # C_calc = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    # Let's test with random matrices
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    # Reshape for einsum
    U_r = U.reshape(n, m, R)
    V_r = V.reshape(m, p, R)
    W_r = W.reshape(p, n, R)
    
    C_calc = np.einsum('ijr,jkr,kir->ik', U_r, V_r, W_r)
    
    if not np.allclose(C_expected, C_calc):
        # Fallback to schoolbook rank 24
        R = 24
        U = np.zeros((n*m, R))
        V = np.zeros((m*p, R))
        W = np.zeros((p*n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
