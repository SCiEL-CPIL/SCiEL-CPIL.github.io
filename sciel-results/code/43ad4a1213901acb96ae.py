import numpy as np
import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    n, m, p = DIMENSIONS
    rank = 49

    # Define Strassen factors for 2x2 matrix multiplication
    # U, V, W are 4x7 matrices for 2x2 (indices 0..3 for 2x2 blocks)
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    
    # Standard Strassen 7 products:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 - M3 + M5 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M4

    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # Indices for 2x2 blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M0
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M1
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1
    
    # M2
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1
    
    # M3
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1
    
    # M4
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1
    
    # M5
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M6
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Now build 4x4 factors (16x49)
    # The 4x4 matrix multiplication is a tensor product of two 2x2 multiplications.
    # U_4x4[i, j, k, r] where i,j are row/col of A (4x4), k is rank index.
    # Rank index r = 7*s + t, where s is the outer Strassen step, t is the inner.
    # A[i,j] = A[2*i1+i2, 2*j1+j2]
    
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)

    for s in range(7):
        for t in range(7):
            r = 7 * s + t
            
            # U part: A coefficients
            # Outer Strassen step s uses blocks I, J
            # Inner Strassen step t uses blocks i, j
            # A index: (2*I + i, 2*J + j)
            # In U2, row index is 2*I + i? No.
            # U2[row, col] where row is block index (0..3).
            # Block 0 is (0,0), 1 is (0,1), 2 is (1,0), 3 is (1,1).
            # So U2[2*I + i, s] gives the coefficient for A[2*I+i, 2*J+j]?
            # Wait, U2 is defined for the 2x2 case.
            # In the 2x2 case, A is 2x2. U2[0] corresponds to A[0,0].
            # In the 4x4 case, the "block" A[I,J] is a 2x2 matrix.
            # The coefficient for the element A[2*I+i, 2*J+j] in the product M_s is:
            # coeff_outer = U2[2*I + i? No, U2 is indexed by block position in 2x2.
            # Let's map carefully.
            
            # For 2x2 Strassen:
            # U2[row, s] is the coefficient of A[row, col] ? No, U2 is just one matrix.
            # The tensor is T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
            # For 2x2: i,j,k in 0..3.
            # U2[i, s] is the coefficient for A[i, ?] No, U is just the left factor.
            # A is 2x2. i is row, j is col.
            # U2[i, r] * V2[j, r] * W2[k, r] ?
            # No, standard decomposition: (AB)_{ik} = sum_j A_{ij} B_{jk}.
            # Tensor T_{ijk}.
            # U_{ij, r} V_{jk, r} W_{ik, r}.
            
            # My U2 above is indexed by [block_index, strassen_step].
            # block_index 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1).
            # So U2[0, 0] = 1 means A[0,0] has coeff 1 in M0.
            
            # For 4x4:
            # A is 4x4.
            # Outer Strassen step s combines 2x2 blocks.
            # Inner Strassen step t combines 1x1 elements within those blocks.
            
            # Coefficient for A[i, j] in rank r=7s+t:
            # i = 2*I + i2, j = 2*J + j2.
            # Block (I, J) has index 2*I + J in the 2x2 block grid.
            # Let block_idx = 2*I + J.
            # The outer coefficient is U2[block_idx, s].
            # The inner coefficient is U2[i2 + 2*j2? No.
            # Within the block (I,J), the element is A[2*I+i2, 2*J+j2].
            # This corresponds to position (i2, j2) in the 2x2 block.
            # Block index for (i2, j2) is 2*i2 + j2.
            # Inner coefficient is U2[2*i2 + j2, t].
            
            # So U4[i, r] = U2[2*I + J, s] * U2[2*i2 + j2, t]
            
            for I in range(2):
                for J in range(2):
                    block_idx_outer = 2 * I + J
                    coeff_outer = U2[block_idx_outer, s]
                    if coeff_outer == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            block_idx_inner = 2 * i2 + j2
                            coeff_inner = U2[block_idx_inner, t]
                            if coeff_inner == 0: continue
                            
                            i = 2 * I + i2
                            j = 2 * J + j2
                            U4[i * 4 + j, r] = coeff_outer * coeff_inner

            # V part: B coefficients
            # B[j, k]. j = 2*J + j2, k = 2*K + k2.
            # Block (J, K) index: 2*J + K.
            # Outer coeff: V2[2*J + K, s].
            # Inner coeff: V2[2*j2 + k2, t].
            
            for J in range(2):
                for K in range(2):
                    block_idx_outer = 2 * J + K
                    coeff_outer = V2[block_idx_outer, s]
                    if coeff_outer == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            block_idx_inner = 2 * j2 + k2
                            coeff_inner = V2[block_idx_inner, t]
                            if coeff_inner == 0: continue
                            
                            j = 2 * J + j2
                            k = 2 * K + k2
                            V4[j * 4 + k, r] = coeff_outer * coeff_inner

            # W part: C coefficients
            # C[k, i] (Trace convention: W[k*n+i, r]).
            # i = 2*I + i2, k = 2*K + k2.
            # Block (I, K) index: 2*I + K.
            # Outer coeff: W2[2*I + K, s].
            # Inner coeff: W2[2*i2 + k2, t].
            
            for I in range(2):
                for K in range(2):
                    block_idx_outer = 2 * I + K
                    coeff_outer = W2[block_idx_outer, s]
                    if coeff_outer == 0: continue
                    for i2 in range(2):
                        for k2 in range(2):
                            block_idx_inner = 2 * i2 + k2
                            coeff_inner = W2[2 * i2 + k2, t]
                            if coeff_inner == 0: continue
                            
                            i = 2 * I + i2
                            k = 2 * K + k2
                            W4[k * 4 + i, r] = coeff_outer * coeff_inner

    # Construction is fast, no need for complex timeout handling inside
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
