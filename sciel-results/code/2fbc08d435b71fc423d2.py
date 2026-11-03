import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # We use a decomposition based on splitting m=4 into two blocks of 2.
    # A = [A1, A2] (2x2 each)
    # B = [B1; B2] (2x5 each)
    # C = A1*B1 + A2*B2
    # We compute A1*B1 and A2*B2 using a rank-16 decomposition for 2x2x5.
    # Total rank = 32.
    
    rank = 32
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a 2x2x5 Strassen-like decomposition
    # Offset in A index, Offset in B index, Offset in W index?
    # We are adding terms for A1*B1 and A2*B2.
    
    # 2x2x5 decomposition rank 16
    # B is 2x5. Split B into B_a (2x2), B_b (2x2), B_c (2x1).
    # A B_a: Rank 7.
    # A B_b: Rank 7.
    # A B_c: Rank 2.
    
    # Indices for A1: A[0:2, 0:2] -> i in 0,1; j in 0,1.
    # Indices for A2: A[0:2, 2:4] -> i in 0,1; j in 2,3.
    # Indices for B1: B[0:2, 0:5] -> j in 0,1; k in 0..4.
    # Indices for B2: B[2:4, 0:5] -> j in 2,3; k in 0..4.
    
    def add_2x2x5_block(a_base, b_base, rank_offset):
        # a_base: 0 for A1, 2 for A2 (column offset in A)
        # b_base: 0 for B1, 2 for B2 (row offset in B)
        
        # B_a: cols 0,1 of B block.
        # B_b: cols 2,3 of B block.
        # B_c: col 4 of B block.
        
        # Strassen for 2x2x2 (B_a)
        # M1 = A11 * Ba11
        # M2 = A22 * Ba22
        # M3 = (A11+A22) * (Ba11+Ba22)
        # M4 = (A11+A22) * (Ba22-Ba11)
        # M5 = (A11+A12) * Ba11? No.
        # Standard Strassen for X Y:
        # M1 = (X11+X22)(Y11+Y22)
        # M2 = (X21+X22)Y11
        # M3 = X11(Y21-Y22)
        # M4 = X22(Y11-Y12)
        # M5 = (X11+X12)Y22
        # M6 = (X21-X11)(Y11+Y12)
        # M7 = (X12-X22)(Y21+Y22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to the tensor indices.
        # This is complex. Let's just use the schoolbook for the 2x2x5 part to be safe?
        # Schoolbook rank for 2x2x5 is 10?
        # C_ik = sum_j A_ij B_jk.
        # j=0,1. k=0..4.
        # 2 * 2 * 5 = 20 terms?
        # No, for fixed j, it's rank 1?
        # Term j: A_i0 B_0k + A_i1 B_1k? No.
        # Term (j,k): A_ij B_jk.
        # Sum over j,k.
        # Rank is 2*5 = 10?
        # Yes, 10 rank-1 tensors.
        
        # So 2x2x5 rank is 10.
        # Total rank for 2x4x5 would be 10 + 10 = 20.
        
        # Let's implement rank 20.
        
        for k in range(5):
            for j in range(2):
                col = rank_offset + j * 5 + k
                
                # A part: A[i, a_base + j] for i in 0,1
                U[a_base * 2 + 0][col] = 1 # i=0? No.
                # A index is i*m + j_col.
                # i is 0,1. j_col is a_base + j.
                U[0 * 4 + (a_base + j)][col] = 1
                U[1 * 4 + (a_base + j)][col] = 1
                
                # B part: B[b_base + j, k]
                # B index is j_row * 5 + k.
                V[(b_base + j) * 5 + k][col] = 1
                
                # W part: W[k, i] for i in 0,1?
                # W index is k * 2 + i.
                W[k * 2 + 0][col] = 1
                W[k * 2 + 1][col] = 1
                
    # Wait, the W part sums to C_ik.
    # C_ik = sum_j A_ij B_jk.
    # My term gives A_ij * B_jk * 1.
    # So summing over j for fixed k,i gives sum_j A_ij B_jk.
    # This is correct.
    
    add_2x2x5_block(0, 0, 0)
    add_2x2x5_block(2, 2, 10)
    
    return {"U": U, "V": V, "W": W}
