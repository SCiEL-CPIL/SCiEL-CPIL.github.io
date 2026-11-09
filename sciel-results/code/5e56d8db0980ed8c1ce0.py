import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate: 66
    # 49 (Strassen 4x4x4) + 16 (4x4x1) + 1 (4x1x5)
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: 4x4x4 Strassen
    # Indices: i in 0..3, j in 0..3, k in 0..3
    # We need to generate the 49 terms for 4x4x4
    # Using recursive Strassen or hardcoded?
    # Hardcoding 49 terms is error-prone.
    # I will use a simpler decomposition for 4x4x4: 8 blocks of 2x2x2 (Rank 56)
    # This is safer to implement correctly.
    # Rank becomes 56 + 16 + 1 = 73.
    
    # Let's stick to 73 to ensure correctness.
    rank = 73
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_2x2x2(i_off, j_off, k_off, start_col):
        # Strassen 2x2x2 coefficients
        # U indices: (i_off+di, j_off+dj)
        # V indices: (j_off+dj, k_off+dk)
        # W indices: (k_off+dk, i_off+di)
        # 7 terms
        terms = [
            # (U_mask, V_mask, W_mask)
            # U: 2x2, V: 2x2, W: 2x2
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B10-B11)
            # M4 = A11(B01-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B10)
            # M7 = (A01-A11)(B01+B11)
            
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
        ]
        # Explicit indices for U, V, W in the global matrices
        # U row: (i_off+di)*m + (j_off+dj)
        # V row: (j_off+dj)*p + (k_off+dk)
        # W row: (k_off+dk)*n + (i_off+di)
        
        # Term 1: M1
        # U: (0,0), (1,1) -> +1
        U[(i_off+0)*m + (j_off+0)][start_col] += 1
        U[(i_off+1)*m + (j_off+1)][start_col] += 1
        # V: (0,0), (1,1) -> +1
        V[(j_off+0)*p + (k_off+0)][start_col] += 1
        V[(j_off+1)*p + (k_off+1)][start_col] += 1
        # W: (0,0), (1,1) -> +1
        W[(k_off+0)*n + (i_off+0)][start_col] += 1
        W[(k_off+1)*n + (i_off+1)][start_col] += 1
        col_ret = start_col + 1
        
        # Term 2: M2
        # U: (1,0), (1,1) -> +1
        U[(i_off+1)*m + (j_off+0)][col_ret] += 1
        U[(i_off+1)*m + (j_off+1)][col_ret] += 1
        # V: (0,0) -> +1
        V[(j_off+0)*p + (k_off+0)][col_ret] += 1
        # W: (0,0) -> +1 (C10 part? No, M2 is for C10 and C11)
        # C10 = M2 + M4. So W(0,1) gets M2?
        # W index is (k, i).
        # M2 contributes to C10 (i=1, k=0) and C11 (i=1, k=1)?
        # M2 = (A10+A11)B00.
        # B00 is B(0,0). So k=0.
        # So W(0, 1) gets +1.
        W[(k_off+0)*n + (i_off+1)][col_ret] += 1
        col_ret += 1
        
        # Term 3: M3
        # U: (0,0) -> +1
        U[(i_off+0)*m + (j_off+0)][col_ret] += 1
        # V: (1,0), (1,1) -> +1, -1
        V[(j_off+1)*p + (k_off+0)][col_ret] += 1
        V[(j_off+1)*p + (k_off+1)][col_ret] -= 1
        # W: (0,0) -> +1 (C01 part)
        # M3 = A00(B10-B11).
        # B10, B11. k=0,1.
        # Contributes to C01 (i=0, k=1).
        # So W(1, 0) gets +1.
        W[(k_off+1)*n + (i_off+0)][col_ret] += 1
        col_ret += 1
        
        # Term 4: M4
        # U: (1,1) -> +1
        U[(i_off+1)*m + (j_off+1)][col_ret] += 1
        # V: (0,1), (0,0) -> +1, -1
        V[(j_off+0)*p + (k_off+1)][col_ret] += 1
        V[(j_off+0)*p + (k_off+0)][col_ret] -= 1
        # W: (0,1), (1,1) -> +1, +1
        # M4 = A11(B01-B00).
        # Contributes to C10 (i=1, k=0) and C11 (i=1, k=1).
        W[(k_off+0)*n + (i_off+1)][col_ret] += 1
        W[(k_off+1)*n + (i_off+1)][col_ret] += 1
        col_ret += 1
        
        # Term 5: M5
        # U: (0,0), (0,1) -> +1
        U[(i_off+0)*m + (j_off+0)][col_ret] += 1
        U[(i_off+0)*m + (j_off+1)][col_ret] += 1
        # V: (1,1) -> +1
        V[(j_off+1)*p + (k_off+1)][col_ret] += 1
        # W: (0,0) -> -1 (C00 part)
        # M5 = (A00+A01)B11.
        # Contributes to C00 (i=0, k=0) and C01 (i=0, k=1).
        # C00 = ... - M5.
        W[(k_off+0)*n + (i_off+0)][col_ret] -= 1
        W[(k_off+1)*n + (i_off+0)][col_ret] += 1
        col_ret += 1
        
        # Term 6: M6
        # U: (1,0), (0,0) -> +1, -1
        U[(i_off+1)*m + (j_off+0)][col_ret] += 1
        U[(i_off+0)*m + (j_off+0)][col_ret] -= 1
        # V: (0,0), (1,0) -> +1
        V[(j_off+0)*p + (k_off+0)][col_ret] += 1
        V[(j_off+1)*p + (k_off+0)][col_ret] += 1
        # W: (0,1) -> +1 (C10 part)
        # M6 = (A10-A00)(B00+B10).
        # Contributes to C10 (i=1, k=0).
        W[(k_off+0)*n + (i_off+1)][col_ret] += 1
        col_ret += 1
        
        # Term 7: M7
        # U: (0,1), (1,1) -> +1, -1
        U[(i_off+0)*m + (j_off+1)][col_ret] += 1
        U[(i_off+1)*m + (j_off+1)][col_ret] -= 1
        # V: (0,1), (1,1) -> +1
        V[(j_off+0)*p + (k_off+1)][col_ret] += 1
        V[(j_off+1)*p + (k_off+1)][col_ret] += 1
        # W: (0,0), (1,1) -> -1, +1
        # M7 = (A01-A11)(B01+B11).
        # Contributes to C00 (i=0, k=0) and C11 (i=1, k=1).
        W[(k_off+0)*n + (i_off+0)][col_ret] -= 1
        W[(k_off+1)*n + (i_off+1)][col_ret] += 1
        col_ret += 1
        
        return col_ret

    # 8 blocks of 2x2x2 for j in 0..3, k in 0..3
    # i_off in 0, 2
    # j_off in 0, 2
    # k_off in 0, 2
    for i_off in [0, 2]:
        for j_off in [0, 2]:
            for k_off in [0, 2]:
                col = add_strassen_2x2x2(i_off, j_off, k_off, col)
                
    # Part 2: 4x4x1 for k=4, j in 0..3
    # 4 blocks of 2x2x1
    # Rank 4 each.
    for i_off in [0, 2]:
        for j_off in [0, 2]:
            # k is fixed at 4
            # C_{i, 4} = sum_j A_{ij} B_{j, 4}
            # For 2x2x1:
            # C00 = A00B00 + A01B10
            # C10 = A10B00 + A11B10
            # 4 terms
            # 1. A00B00
            U[(i_off+0)*m + (j_off+0)][col] += 1
            V[(j_off+0)*p + 4][col] += 1
            W[4*n + (i_off+0)][col] += 1
            col += 1
            # 2. A01B10
            U[(i_off+0)*m + (j_off+1)][col] += 1
            V[(j_off+1)*p + 4][col] += 1
            W[4*n + (i_off+0)][col] += 1
            col += 1
            # 3. A10B00
            U[(i_off+1)*m + (j_off+0)][col] += 1
            V[(j_off+0)*p + 4][col] += 1
            W[4*n + (i_off+1)][col] += 1
            col += 1
            # 4. A11B10
            U[(i_off+1)*m + (j_off+1)][col] += 1
            V[(j_off+1)*p + 4][col] += 1
            W[4*n + (i_off+1)][col] += 1
            col += 1
            
    # Part 3: 4x1x5 for j=4
    # Rank 1
    # C_{i, k} = A_{i, 4} B_{4, k}
    for i in range(4):
        U[i*m + 4][col] += 1
    for k in range(5):
        V[4*p + k][col] += 1
        for i in range(4):
            W[k*n + i][col] += 1 # Wait, W needs to be A_i4 * B_4k
            # My previous logic: U=A, V=1, W=B
            # Here U is A_i4. V is 1. W is B_4k.
            # So W[k*n + i] should be B_4k?
            # But W is fixed.
            # If I set U[i, 4, r] = A_i4, V[4, k, r] = 1, W[k, i, r] = B_4k
            # Then sum_r A_i4 * 1 * B_4k = A_i4 B_4k.
            # So W[k*n + i] = B_4k.
            # But B_4k is not 1.
            # So I cannot use Rank 1 with integer coefficients unless B_4k is 1?
            # No, the decomposition must be exact for ANY A, B.
            # The tensor entries are 1.
            # T_{i, 4, k} = 1.
            # So C_{ik} = A_{i4} B_{4k}.
            # We need U_{i, 4, r} V_{4, k, r} W_{k, i, r} = 1 for all i, k.
            # This implies the matrix $M_{ik} = \sum_r U_{i, 4, r} V_{4, k, r} W_{k, i, r}$ must be all 1s.
            # This is a rank-1 matrix? No, it's a matrix of 1s.
            # Rank of all-1 matrix is 1.
            # So we can do it with 1 term.
            # Set U_{i, 4, 0} = 1.
            # Set V_{4, k, 0} = 1.
            # Set W_{k, i, 0} = 1.
            # Then sum is 1.
            # Correct.
            pass
    
    # Fix Part 3
    for i in range(4):
        U[i*m + 4][col] = 1
    for k in range(5):
        V[4*p + k][col] = 1
        for i in range(4):
            W[k*n + i][col] = 1
    col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
