"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    t0 = time.time()
    deadline = t0 + 100  # safety margin before 150s kill
    
    # Build the 2x2x2 Strassen decomposition (rank 7)
    # U: (4,7) indexed by (i,k) where i=row, k=col of A
    # V: (4,7) indexed by (k,j) where k=row, j=col of B  
    # W: (4,7) indexed by (j,i) where j=row, i=col of C
    # T[i,k,j] = sum_r U[i*2+k, r] * V[k*2+j, r] * W[j*2+i, r]
    
    U2 = np.zeros((4, 7))
    V2 = np.zeros((4, 7))
    W2 = np.zeros((4, 7))
    
    # M1 = (A11+A22)(B11+B22)
    # A11: i=0,k=0 -> U[0]; A22: i=1,k=1 -> U[3]
    # B11: k=0,j=0 -> V[0]; B22: k=1,j=1 -> V[3]
    # C11: j=0,i=0 -> W[0]; C22: j=1,i=1 -> W[3]
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2 = (A21+A22)B11
    # A21: i=1,k=0 -> U[2]; A22: i=1,k=1 -> U[3]
    # B11: k=0,j=0 -> V[0]
    # C21: j=0,i=1 -> W[1]; C22: j=1,i=1 -> W[3]
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # M3 = A11(B12-B22)
    # A11: i=0,k=0 -> U[0]
    # B12: k=0,j=1 -> V[1]; B22: k=1,j=1 -> V[3]
    # C12: j=1,i=0 -> W[2]; C22: j=1,i=1 -> W[3]
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1
    
    # M4 = A22(B21-B11)
    # A22: i=1,k=1 -> U[3]
    # B21: k=1,j=0 -> V[2]; B11: k=0,j=0 -> V[0]
    # C11: j=0,i=0 -> W[0]; C21: j=0,i=1 -> W[1]
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[1, 3] = 1
    
    # M5 = (A11+A12)B22
    # A11: i=0,k=0 -> U[0]; A12: i=0,k=1 -> U[1]
    # B22: k=1,j=1 -> V[3]
    # C11: j=0,i=0 -> W[0]; C12: j=1,i=0 -> W[2]
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    # A21: i=1,k=0 -> U[2]; A11: i=0,k=0 -> U[0]
    # B11: k=0,j=0 -> V[0]; B12: k=0,j=1 -> V[1]
    # C22: j=1,i=1 -> W[3]
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    # A12: i=0,k=1 -> U[1]; A22: i=1,k=1 -> U[3]
    # B21: k=1,j=0 -> V[2]; B22: k=1,j=1 -> V[3]
    # C11: j=0,i=0 -> W[0]
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1
    
    # Verify 2x2x2 Strassen
    for trial in range(5):
        A_test = np.random.randn(2, 2)
        B_test = np.random.randn(2, 2)
        C_expected = A_test @ B_test
        C_computed = np.zeros((2, 2))
        for i in range(2):
            for k in range(2):
                for j in range(2):
                    for r in range(7):
                        C_computed[i, j] += U2[i*2+k, r] * V2[k*2+j, r] * W2[j*2+i, r]
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            raise ValueError("2x2x2 Strassen verification failed")
    
    # Build 4x4x4 Strassen decomposition (rank 49)
    # Split 4x4 into 2x2 blocks, use Strassen at block level
    # Each block multiplication uses 2x2x2 Strassen (rank 7)
    # Total: 7 * 7 = 49
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Block definitions for 4x4:
    # A blocks: A[bi,bk] where bi in {0,1}, bk in {0,1}
    #   A[0,0]: rows 0-1, cols 0-1
    #   A[0,1]: rows 0-1, cols 2-3
    #   A[1,0]: rows 2-3, cols 0-1
    #   A[1,1]: rows 2-3, cols 2-3
    # B blocks: B[bk,bj] where bk in {0,1}, bj in {0,1}
    #   B[0,0]: rows 0-1, cols 0-1
    #   B[0,1]: rows 0-1, cols 2-3
    #   B[1,0]: rows 2-3, cols 0-1
    #   B[1,1]: rows 2-3, cols 2-3
    # C blocks: C[bj,bi] where bj in {0,1}, bi in {0,1}
    #   C[0,0]: rows 0-1, cols 0-1
    #   C[0,1]: rows 0-1, cols 2-3
    #   C[1,0]: rows 2-3, cols 0-1
    #   C[1,1]: rows 2-3, cols 2-3
    
    # Strassen at 4x4 level (7 big multiplications):
    big_mults = [
        # M1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
        {'A': [((0,0), 1), ((1,1), 1)], 'B': [((0,0), 1), ((1,1), 1)], 'C': [((0,0), 1), ((1,1), 1)]},
        # M2 = (A[1,0]+A[1,1])B[0,0]
        {'A': [((1,0), 1), ((1,1), 1)], 'B': [((0,0), 1)], 'C': [((1,0), 1), ((1,1), -1)]},
        # M3 = A[0,0](B[0,1]-B[1,1])
        {'A': [((0,0), 1)], 'B': [((0,1), 1), ((1,1), -1)], 'C': [((0,1), 1), ((1,1), 1)]},
        # M4 = A[1,1](B[1,0]-B[0,0])
        {'A': [((1,1), 1)], 'B': [((1,0), 1), ((0,0), -1)], 'C': [((0,0), 1), ((1,0), 1)]},
        # M5 = (A[0,0]+A[0,1])B[1,1]
        {'A': [((0,0), 1), ((0,1), 1)], 'B': [((1,1), 1)], 'C': [((0,0), -1), ((0,1), 1)]},
        # M6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
        {'A': [((1,0), 1), ((0,0), -1)], 'B': [((0,0), 1), ((0,1), 1)], 'C': [((1,1), 1)]},
        # M7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
        {'A': [((0,1), 1), ((1,1), -1)], 'B': [((1,0), 1), ((1,1), 1)], 'C': [((0,0), 1)]},
    ]
    
    for big_idx, bm in enumerate(big_mults):
        # For each big multiplication, we need to combine the A and B block contributions
        # Then apply 2x2x2 Strassen for each of the 4 entries
        
        # Build combined A and B "virtual" 2x2 matrices
        # Each A block contributes to specific (i,k) positions
        # A block (bi, bk): i = 2*bi + i2, k = 2*bk + k2 for i2,k2 in {0,1}
        # B block (bk, bj): k = 2*bk + k2, j = 2*bj + j2 for k2,j2 in {0,1}
        
        # For the 2x2x2 Strassen, we need to map the combined blocks
        # The combined A "matrix" has entries at positions determined by the block contributions
        # Similarly for B
        
        # For each of the 4 entries of the 2x2 result (corresponding to the 4 sub-entries
        # within the 2x2 block structure), we apply the 2x2x2 Strassen
        
        # Actually, let me think about this differently.
        # Each big multiplication is a 2x2 x 2x2 matrix multiplication where the "entries"
        # are themselves 2x2 matrices. So we can use the 2x2x2 Strassen structure directly.
        
        # The 2x2x2 Strassen U2, V2, W2 give us the decomposition for multiplying
        # two 2x2 matrices. The "entries" in U2[i*2+k] correspond to the (i,k) position.
        # For our 4x4 case, the (i,k) position in the 2x2 block structure maps to
        # actual 4x4 indices.
        
        # For a given big multiplication with A blocks and B blocks:
        # The "virtual" 2x2 A matrix has entry (bi, bk) = sum of signs for block (bi,bk)
        # The "virtual" 2x2 B matrix has entry (bk, bj) = sum of signs for block (bk,bj)
        
        # But we need to handle the fact that multiple blocks may contribute to the same
        # virtual entry. Let's build the virtual 2x2 matrices.
        
        A_virtual = np.zeros((2, 2))
        for (bi, bk), sign in bm['A']:
            A_virtual[bi, bk] += sign
        
        B_virtual = np.zeros((2, 2))
        for (bk, bj), sign in bm['B']:
            B_virtual[bk, bj] += sign
        
        # Now for each of the 7 Strassen multiplications in the 2x2x2 decomposition:
        # We need to map the virtual 2x2 entries to actual 4x4 entries.
        
        for r2 in range(7):
            col = big_idx * 7 + r2
            
            # U2[i*2+k, r2] gives the coefficient for A entry (i,k) in this Strassen step
            # This maps to 4x4 A entry at block (i,k) -> actual indices
            for i2 in range(2):
                for k2 in range(2):
                    u_val = U2[i2*2 + k2, r2]
                    if u_val != 0:
                        # A block (i2, k2): rows 2*i2 to 2*i2+1, cols 2*k2 to 2*k2+1
                        # But we need to add the virtual coefficient
                        # The virtual A[i2, k2] is the sum of signs
                        # Actually, U2 already encodes the Strassen structure
                        # We need to multiply by the virtual A coefficient
                        a_coeff = A_virtual[i2, k2]
                        if a_coeff != 0:
                            # Add to U4 for all 4 entries in this block
                            for di in range(2):
                                for dk in range(2):
                                    u4_idx = (2*i2 + di) * 4 + (2*k2 + dk)
                                    U4[u4_idx, col] += u_val * a_coeff
            
            # Similarly for V
            for k2 in range(2):
                for j2 in range(2):
                    v_val = V2[k2*2 + j2, r2]
                    if v_val != 0:
                        b_coeff = B_virtual[k2, j2]
                        if b_coeff != 0:
                            for dk in range(2):
                                for dj in range(2):
                                    v4_idx = (2*k2 + dk) * 4 + (2*j2 + dj)
                                    V4[v4_idx, col] += v_val * b_coeff
            
            # For W
            for j2 in range(2):
                for i2 in range(2):
                    w_val = W2[j2*2 + i2, r2]
                    if w_val != 0:
                        # C block (j2, i2): rows 2*j2 to 2*j2+1, cols 2*i2 to 2*i2+1
                        # But we need to check which C blocks this big mult contributes to
                        c_coeff = 0
                        for (cj, ci), sign in bm['C']:
                            if cj == j2 and ci == i2:
                                c_coeff += sign
                        if c_coeff != 0:
                            for dj in range(2):
                                for di in range(2):
                                    w4_idx = (2*j2 + dj) * 4 + (2*i2 + di)
                                    W4[w4_idx, col] += w_val * c_coeff
    
    # Verify 4x4x4 Strassen
    for trial in range(3):
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 4)
        C_expected = A_test @ B_test
        C_computed = np.zeros((4, 4))
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    for r in range(49):
                        C_computed[i, j] += U4[i*4+k, r] * V4[k*4+j, r] * W4[j*4+i, r]
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            raise ValueError(f"4x4x4 Strassen verification failed: max err = {np.max(np.abs(C_computed - C_expected))}")
    
    # Now build the full 4x4x9 decomposition
    # Columns 0-3: 4x4x4 with rank 49
    # Columns 4-7: 4x4x4 with rank 49
    # Column 8: 4x4x1 with rank 16 (schoolbook)
    # Total rank: 49 + 49 + 16 = 114
    
    rank = 114
    U = np.zeros((16, rank))  # n*m = 16
    V = np.zeros((36, rank))  # m*p = 36
    W = np.zeros((36, rank))  # p*n = 36
    
    # Block 1: columns 0-3 of B, rows 0-3 of C
    # B block: rows 0-3 (j=0..3), C block: rows 0-3 (k=0..3)
    # U: rows 0-15 (i=0..3, k=0..3)
    # V: rows 0-15 (k=0..3, j=0..3)
    # W: rows 0-15 (j=0..3, i=0..3)
    
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    
    # Block 2: columns 4-7 of B, rows 4-7 of C
    # B block: rows 4-7 (j=4..7), C block: rows 4-7 (k=4..7)
    # U: rows 0-15 (same A)
    # V: rows 16-31 (k=0..3, j=4..7) -> V4 mapped
    # W: rows 16-31 (j=4..7, i=0..3) -> W4 mapped
    
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    
    # Block 3: column 8 of B, row 8 of C
    # B: row 8 (j=8), C: row 8 (k=8)
    # Schoolbook: 16 multiplications
    # C[i,8] = sum_k A[i,k] * B[k,8]
    # W[8*4+i, col] = 1 for each (i,k)
    
    col = 98
    for i in range(4):
        for k in range(4):
            U[i*4+k, col] = 1
            V[k*4+8, col] = 1  # k*4 + j where j=8
            W[8*4+i, col] = 1  # j*4 + i where j=8
            col += 1
    
    # Verify full decomposition
    for trial in range(5):
        A_test = np.random.randn(4, 4)
        B_test = np.random.randn(4, 9)
        C_expected = A_test @ B_test
        C_computed = np.zeros((4, 9))
        for i in range(4):
            for k in range(4):
                for j in range(9):
                    for r in range(rank):
                        C_computed[i, j] += U[i*4+k, r] * V[k*4+j, r] * W[j*4+i, r]
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            max_err = np.max(np.abs(C_computed - C_expected))
            raise ValueError(f"Full decomposition verification failed: max err = {max_err}")
    
    if time.time() > deadline:
        # Should not happen, but safety
        pass
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
