"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=7
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    U = [[0] * 81 for _ in range(nm)]
    V = [[0] * 81 for _ in range(mp)]
    W = [[0] * 81 for _ in range(pn)]
    
    rank_offset = 0
    
    # Part 1: 4x4x4 (k in 0..3) using Strassen squared (rank 49)
    # Strassen 2x2x2 (rank 7)
    # A: 2x2, B: 2x2, C: 2x2
    # Indices i,j,k in {0,1}
    # We need to map 2x2x2 to 4x4x4 via tensor product.
    # U_4x4x4[i1*2+i2, j1*2+j2, r1*7+r2] = U_2x2x2[i1, j1, r1] * U_2x2x2[i2, j2, r2]
    # Wait, U is for A. A is 4x4.
    # A[i,j] = A[i1,i2, j1,j2]
    # U_A[i, r] = sum U_2[i1, r1] U_2[i2, r2] ?
    # The tensor product of decompositions:
    # T_4 = T_2 \otimes T_2
    # U_4[i, r] = U_2[i1, r1] * U_2[i2, r2]
    # V_4[j, r] = V_2[j1, r1] * V_2[j2, r2]
    # W_4[k, r] = W_2[k1, r1] * W_2[k2, r2]
    
    # Strassen 2x2x2 tensors
    # U2[i, r], V2[j, r], W2[k, r]
    # r in 0..6
    U2 = [[0]*7 for _ in range(2)]
    V2 = [[0]*7 for _ in range(2)]
    W2 = [[0]*7 for _ in range(2)]
    
    # M1 = (A11+A22)(B11+B22)
    U2[0][0] = 1; U2[1][0] = 1
    V2[0][0] = 1; V2[1][0] = 1
    W2[0][0] = 1; W2[1][0] = 1 # C11, C22 get +1
    # Actually W2[k, r] is coeff for C[k].
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Let's define the 7 terms
    # Term 0: M1
    U2[0][0] = 1; U2[1][0] = 1
    V2[0][0] = 1; V2[1][0] = 1
    W2[0][0] = 1; W2[1][0] = 1
    
    # Term 1: M2 = (A21+A22)B11
    U2[1][1] = 1 # A21, A22? No, A21 and A22 are row 2 (index 1).
    # A21 is i=1, j=0. A22 is i=1, j=1.
    U2[1][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1 # C21 gets M2
    
    # Term 2: M3 = A11(B12-B22)
    U2[0][2] = 1
    V2[1][2] = 1
    V2[0][2] = -1 # B12 is j=0? No. B is 2x2. B12 is j=0, k=1. B22 is j=1, k=1.
    # V2[j, r]. B12 corresponds to j=0, k=1.
    # Wait, V is for B. B[j, k].
    # M3 = A11 * (B12 - B22).
    # A11: i=0, j=0. U2[0][2]=1.
    # B12: j=0, k=1. V2[0][2]=1.
    # B22: j=1, k=1. V2[1][2]=-1.
    # C12: i=0, k=1. W2[0][2]=1.
    
    # Let's restart Strassen definition carefully.
    U2 = [[0]*7 for _ in range(2)]
    V2 = [[0]*7 for _ in range(2)]
    W2 = [[0]*7 for _ in range(2)]
    
    # r=0: M1 = (A11+A22)(B11+B22)
    U2[0][0] = 1; U2[1][0] = 1
    V2[0][0] = 1; V2[1][0] = 1
    W2[0][0] = 1; W2[1][0] = 1
    
    # r=1: M2 = (A21+A22)B11
    U2[1][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1
    
    # r=2: M3 = A11(B12-B22)
    U2[0][2] = 1
    V2[0][2] = 1 # B12 (j=0, k=1) -> V2[0]
    V2[1][2] = -1 # B22 (j=1, k=1) -> V2[1]
    W2[0][2] = 1 # C12 (i=0, k=1) -> W2[0]
    
    # r=3: M4 = A22(B21-B11)
    U2[1][3] = 1
    V2[1][3] = 1 # B21 (j=1, k=0)
    V2[0][3] = -1 # B11 (j=0, k=0)
    W2[1][3] = 1 # C21 (i=1, k=0)
    
    # r=4: M5 = (A11+A12)B22
    U2[0][4] = 1; U2[1][4] = 1
    V2[1][4] = 1
    W2[0][4] = -1 # C11 gets -M5
    W2[1][4] = 1  # C22 gets +M5? No.
    # C11 = M1 + M4 - M5 + M7
    # C22 = M1 - M2 + M3 + M6
    # M5 is (A11+A12)B22.
    # C11 term: -M5.
    # C22 term: 0?
    # Wait, Strassen formulas:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # So M5 contributes to C11 (-1) and C12 (+1).
    W2[0][4] = -1
    W2[0][4] = 1 # C12? No, W2[k, r]. C12 is i=0, k=1.
    # My W2 index is k.
    # C11: i=0, k=0.
    # C12: i=0, k=1.
    # C21: i=1, k=0.
    # C22: i=1, k=1.
    # W2[k, r] is coeff for C[i, k]?
    # In tensor decomposition T[i,j,k] = sum U[i,r] V[j,r] W[k,r].
    # Here i is row of A, k is col of B.
    # So W2[k, r] is coeff for C[i, k].
    # But W2 is size 2. k in {0,1}.
    # So W2[0] is for k=0 (C11, C21).
    # W2[1] is for k=1 (C12, C22).
    # But C11 and C21 have different i.
    # The tensor T is 3D.
    # U[i,r], V[j,r], W[k,r].
    # T[i,j,k] = sum U[i,r] V[j,r] W[k,r].
    # This implies W[k,r] is independent of i.
    # This is correct for matrix multiplication tensor.
    
    # So for M5:
    # C11 (i=0, k=0) gets -1.
    # C12 (i=0, k=1) gets +1.
    # But W2[k,r] cannot depend on i.
    # Contradiction?
    # No, Strassen works because the i-dependence is in U.
    # M5 = (A11+A12)B22.
    # U[0, r] = 1 (A11), U[1, r] = 1 (A12).
    # V[1, r] = 1 (B22).
    # W[0, r] = -1 (for C11? No, C11 is i=0, k=0).
    # If W[0, r] = -1, then for i=1, k=0 (C21), it also gets -1 * U[1,r] V[j,r].
    # U[1,r] is 1. So C21 gets -1.
    # But C21 should not get M5.
    # So Strassen is NOT a simple tensor product of U, V, W in this way?
    # It IS a tensor decomposition.
    # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r].
    # If U[0,r]=1, U[1,r]=1, V[1,r]=1, W[0,r]=-1, W[1,r]=1.
    # Then:
    # i=0, j=1, k=0: U[0,r]V[1,r]W[0,r] = 1*1*(-1) = -1. (C11 part)
    # i=1, j=1, k=0: U[1,r]V[1,r]W[0,r] = 1*1*(-1) = -1. (C21 part)
    # But C21 should not have M5.
    # So this specific M5 term is not a rank-1 tensor in the standard sense?
    # No, Strassen's M5 is a bilinear form.
    # The decomposition is valid.
    # I will hardcode the final 49 rank tensors for 4x4x4 to avoid derivation errors.
    
    # Given the complexity, I will use the 112 rank schoolbook if I cannot guarantee correctness of the optimized versions quickly.
    # However, 81 is a strong improvement.
    # I will implement the 81 rank solution with careful mapping.
    
    # For now, return schoolbook to ensure valid output.
    rank = 112
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
