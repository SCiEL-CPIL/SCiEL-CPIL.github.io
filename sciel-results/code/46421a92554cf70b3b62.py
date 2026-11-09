import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def _strassen_2x2():
    """
    Strassen decomposition for 2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) where U indexes [i*2+j], V indexes [j*2+k], W indexes [k*2+i].
    Strassen uses 7 multiplications.
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B01-B11)
    M4 = A11(B10-B00)
    M5 = (A00+A10)B11
    M6 = (A11-A00)(B00+B10)
    M7 = (A01-A11)(B01+B11)
    C00 = M1+M4-M5-M7
    C01 = M3+M5
    C10 = M2+M4
    C11 = M1-M2+M3+M7
    """
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    # 7 multiplications
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1: (A00+A11)(B00+B11)
    # U[0][0]=1, U[3][0]=1 (A00, A11)
    # V[0][0]=1, V[3][0]=1 (B00, B11)
    # W[0][0]=1 (C00 += M1), W[3][0]=1 (C11 += M1)
    U[0][0] += 1; U[3][0] += 1
    V[0][0] += 1; V[3][0] += 1
    W[0][0] += 1; W[3][0] += 1
    
    # M2: (A10+A11)B00
    # U[2][1]=1, U[3][1]=1 (A10, A11)
    # V[0][1]=1 (B00)
    # W[2][1]=1 (C10 += M2)
    U[2][1] += 1; U[3][1] += 1
    V[0][1] += 1
    W[2][1] += 1
    
    # M3: A00(B01-B11)
    # U[0][2]=1 (A00)
    # V[1][2]=1, V[3][2]=-1 (B01, B11)
    # W[1][2]=1 (C01 += M3), W[3][2]=1 (C11 += M3)
    U[0][2] += 1
    V[1][2] += 1; V[3][2] -= 1
    W[1][2] += 1; W[3][2] += 1
    
    # M4: A11(B10-B00)
    # U[3][3]=1 (A11)
    # V[2][3]=1, V[0][3]=-1 (B10, B00)
    # W[0][3]=1 (C00 += M4), W[2][3]=1 (C10 += M4)
    U[3][3] += 1
    V[2][3] += 1; V[0][3] -= 1
    W[0][3] += 1; W[2][3] += 1
    
    # M5: (A00+A10)B11
    # U[0][4]=1, U[2][4]=1 (A00, A10)
    # V[3][4]=1 (B11)
    # W[0][4]=-1 (C00 -= M5), W[1][4]=1 (C01 += M5)
    U[0][4] += 1; U[2][4] += 1
    V[3][4] += 1
    W[0][4] -= 1; W[1][4] += 1
    
    # M6: (A11-A00)(B00+B10)
    # U[3][5]=1, U[0][5]=-1 (A11, A00)
    # V[0][5]=1, V[2][5]=1 (B00, B10)
    # W[3][5]=-1 (C11 -= M6)... wait, C11 = M1-M2+M3+M7, no M6 in C11
    # Let me recheck Strassen formulas:
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M7
    # M6 is (A11-A00)(B00+B10) but it's not used in the final C? 
    # Actually, standard Strassen uses:
    # P1 = (A11 - A00)(B00 + B10)
    # P2 = (A00 + A11)(B00 + B11)
    # P3 = (A10 + A11) B00
    # P4 = (A00 + A10) B11
    # P5 = (A01 - A11)(B01 + B11)
    # P6 = A00 (B01 - B11)
    # P7 = A11 (B10 - B00)
    # C00 = P2 + P4 - P7 - P5
    # C01 = P1 + P5
    # C10 = P3 - P1
    # C11 = P2 - P4 + P6 + P1
    # Let me redo this properly.
    
    # Let me use the standard formulation:
    # p1 = (A11 - A00)(B00 + B10)
    # p2 = (A00 + A11)(B00 + B11)
    # p3 = (A10 + A11) B00
    # p4 = (A00 + A10) B11
    # p5 = (A01 - A11)(B01 + B11)
    # p6 = A00 (B01 - B11)
    # p7 = A11 (B10 - B00)
    # C00 = p2 + p4 - p7 - p5
    # C01 = p1 + p5
    # C10 = p3 - p1
    # C11 = p2 - p4 + p6 + p1
    
    # Reset and redo
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # p1 = (A11 - A00)(B00 + B10)
    U[3][0] = 1; U[0][0] = -1
    V[0][0] = 1; V[2][0] = 1
    W[1][0] = 1  # C01 += p1
    W[2][0] = -1 # C10 -= p1
    W[3][0] = 1  # C11 += p1
    
    # p2 = (A00 + A11)(B00 + B11)
    U[0][1] = 1; U[3][1] = 1
    V[0][1] = 1; V[3][1] = 1
    W[0][1] = 1  # C00 += p2
    W[3][1] = 1  # C11 += p2
    
    # p3 = (A10 + A11) B00
    U[2][2] = 1; U[3][2] = 1
    V[0][2] = 1
    W[2][2] = 1  # C10 += p3
    
    # p4 = (A00 + A10) B11
    U[0][3] = 1; U[2][3] = 1
    V[3][3] = 1
    W[0][3] = 1  # C00 += p4
    W[3][3] = -1 # C11 -= p4
    
    # p5 = (A01 - A11)(B01 + B11)
    U[1][4] = 1; U[3][4] = -1
    V[1][4] = 1; V[3][4] = 1
    W[0][4] = -1 # C00 -= p5
    W[1][4] = 1  # C01 += p5
    
    # p6 = A00 (B01 - B11)
    U[0][5] = 1
    V[1][5] = 1; V[3][5] = -1
    W[3][5] = 1  # C11 += p6
    
    # p7 = A11 (B10 - B00)
    U[3][6] = 1
    V[2][6] = 1; V[0][6] = -1
    W[0][6] = -1 # C00 -= p7
    
    return U, V, W

def _kron_strassen_4x4x4():
    """
    Build 4x4x4 decomposition using Kronecker product of Strassen 2x2 with itself.
    Rank = 7 * 7 = 49.
    """
    U2, V2, W2 = _strassen_2x2()
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            u_idx = (i1*2+i2)*4 + (j1*2+j2)
                            val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                            if val != 0:
                                U[u_idx][r] = val
                            
                            v_idx = (j1*2+j2)*4 + (j1*2+j2)  # wrong, V indexes j,k
                            # V[j*4+k, r] where j = j1*2+j2, k = j2*2+k2... no
                            # For 4x4x4: U[i*4+j], V[j*4+k], W[k*4+i]
                            # Kronecker: j = j1*2+j2, k = k1*2+k2, i = i1*2+i2
                            # U[i*4+j] = U2[i1*2+i2, r1] * U2[i2*2+j2, r2]... 
                            # Actually for Kronecker of matrix mult:
                            # A = A1 ⊗ A2, B = B1 ⊗ B2, C = C1 ⊗ C2
                            # A[i1,i2][j1,j2] * B[j1,j2][k1,k2] = C[i1,i2][k1,k2]
                            # This decomposes as sum over r1,r2 of U2[i1,j1;r1]*U2[i2,j2;r2] * V2[j1,k1;r1]*V2[j2,k2;r2] * W2[k1,i1;r1]*W2[k2,i2;r2]
                            
    # Let me redo this properly
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    u_val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                    v_val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                    w_val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                    
                                    i_idx = i1*4 + i2
                                    j_idx = j1*4 + j2
                                    k_idx = k1*4 + k2
                                    
                                    U[i_idx*4 + j_idx][r] = u_val
                                    V[j_idx*4 + k_idx][r] = v_val
                                    W[k_idx*4 + i_idx][r] = w_val
                                    break
                        break
                    break
                break
            break
    
    # This is getting complicated. Let me use a cleaner approach.
    # For Kronecker product of two matrix multiplications:
    # (A1 ⊗ A2)(B1 ⊗ B2) = (A1B1) ⊗ (A2B2)
    # If A1B1 = sum_r1 u1_r1 * v1_r1 * w1_r1 and A2B2 = sum_r2 u2_r2 * v2_r2 * w2_r2
    # then (A1⊗A2)(B1⊗B2) = sum_{r1,r2} (u1_r1⊗u2_r2)(v1_r1⊗v2_r2)(w1_r1⊗w2_r2)
    
    # For 4x4x4: A is 4x4 = (2x2)⊗(2x2), B is 4x4, C is 4x4
    # A[i1,i2][j1,j2] = A1[i1][j1] * A2[i2][j2]
    # B[j1,j2][k1,k2] = B1[j1][k1] * B2[j2][k2]
    # C[i1,i2][k1,k2] = C1[i1][k1] * C2[i2][k2]
    
    # U[i*4+j, r] where i=i1*2+i2, j=j1*2+j2
    # V[j*4+k, r] where j=j1*2+j2, k=k1*2+k2
    # W[k*4+i, r] where k=k1*2+k2, i=i1*2+i2
    
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for k1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                for k2 in range(2):
                                    u_val = U2[i1*2+j1][r1] * U2[i2*2+j2][r2]
                                    v_val = V2[j1*2+k1][r1] * V2[j2*2+k2][r2]
                                    w_val = W2[k1*2+i1][r1] * W2[k2*2+i2][r2]
                                    
                                    i_idx = i1*2 + i2
                                    j_idx = j1*2 + j2
                                    k_idx = k1*2 + k2
                                    
                                    U[i_idx*4 + j_idx][r] = u_val
                                    V[j_idx*4 + k_idx][r] = v_val
                                    W[k_idx*4 + i_idx][r] = w_val
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, sample=True):
    """
    Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 if j==l else 0
    Wait, the multiplication tensor for A(m×n) * B(n×p) = C(m×p):
    C[i,k] = sum_j A[i,j] * B[j,k]
    The tensor T[i,j,k] = 1 (scalar), and we want T = sum_r u_r ⊗ v_r ⊗ w_r
    where u_r[i,j] = U[i*m+j, r]... 
    Actually, for matrix multiplication A (n×m) × B (m×p) = C (n×p):
    C[i,k] = sum_j A[i,j] B[j,k]
    The multiplication tensor has entries T[i,j,k] such that C[i,k] = sum_{j} A[i,j] B[j,k]
    As a tensor: C = sum_{i,j,k} C[i,k] A[i,j] B[j,k]... 
    
    Actually, the standard formulation: The matrix multiplication tensor T of format (n,m,p)
    has T[i,j,k] = delta_{j,l}... no.
    
    Let me think again. We have A (n×m), B (m×p), C (n×p).
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The "multiplication tensor" is a 3-index object where T[i,j,k] represents the coefficient
    such that C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k].
    So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
    
    Actually, the tensor decomposition of matrix multiplication:
    We want to write the bilinear map (A, B) -> C as a sum of rank-1 terms.
    C[i,k] = sum_r u_r[i,j] * v_r[j,k] * w_r[k,i]... 
    
    Hmm, let me look at the interface again:
    U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    The decomposition says: for each r, the term is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    So the tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    And we need T[i,j,k] = 1 if... 
    
    For matrix multiplication: C[i,k] = sum_j A[i,j] B[j,k]
    The tensor should satisfy: sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = delta(j, j') ... 
    
    I think the correct condition is:
    sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k*n+i, r] = 1 if j == j' else 0
    
    This makes sense because:
    C[i,k] = sum_{j,j'} A[i,j] * (sum_r U[i*m+j,r] V[j'*p+k,r] W[k*n+i,r]) * B[j',k]
    = sum_{j,j'} A[i,j] delta(j,j') B[j',k]
    = sum_j A[i,j] B[j,k]
    
    Yes! That's correct.
    """
    rank = len(U[0])
    errors = 0
    total = 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    expected = 1 if j == jp else 0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i*m+j][r] * V[jp*p+k][r] * W[k*n+i][r]
                    if abs(actual - expected) > 1e-9:
                        errors += 1
                    total += 1
                    if sample and total > 10000:
                        break
                if sample and total > 10000:
                    break
            if sample and total > 10000:
                break
        if sample and total > 10000:
            break
    return errors == 0

def _schoolbook(n, m, p):
    """Build schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n, m, p = 4, 4, 9
    
    # Build 4x4x4 Strassen (rank 49)
    U1, V1, W1 = _kron_strassen_4x4x4()
    rank1 = len(U1[0])
    
    # Build 4x4x5 schoolbook (rank 80)
    U2, V2, W2 = _schoolbook(4, 4, 5)
    rank2 = len(U2[0])
    
    # Combine into 4x4x9
    total_rank = rank1 + rank2
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # Copy first part (4x4x4)
    for i in range(4):
        for j in range(4):
            for r in range(rank1):
                U[i*4+j][r] = U1[i*4+j][r]
        for j in range(4):
            for k in range(4):
                for r in range(rank1):
                    V[j*4+k][r] = V1[j*4+k][r]
        for k in range(4):
            for i in range(4):
                for r in range(rank1):
                    W[k*4+i][r] = W1[k*4+i][r]
    
    # Copy second part (4x4x5) offset by rank1
    for i in range(4):
        for j in range(4):
            for r in range(rank2):
                U[i*4+j][rank1+r] = U2[i*4+j][r]
        for j in range(4):
            for k in range(5):
                for r in range(rank2):
                    V[j*4+(4+k)][rank1+r] = V2[j*4+k][r]
        for k in range(5):
            for i in range(4):
                for r in range(rank2):
                    W[(4+k)*4+i][rank1+r] = W2[k*4+i][r]
    
    # Verify
    if time.time() < deadline - 5:
        if not _verify_decomposition(U, V, W, n, m, p, sample=True):
            # Fallback to schoolbook
            U, V, W = _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
