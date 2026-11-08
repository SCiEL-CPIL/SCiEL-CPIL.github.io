import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k; 0 otherwise
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Strategy: Decompose 2x4x6 as Kronecker product of 2x2x2 (Strassen, rank 7)
    # and 1x2x3 (trivial, rank 6)
    # 
    # For 2x2x2 Strassen decomposition:
    # The 7 rank-1 terms for 2x2x2 matrix multiplication
    # Let's define the Strassen decomposition for 2x2x2:
    # A (2x2) * B (2x2) = C (2x2)
    # Strassen uses 7 multiplications:
    # m1 = (A00 + A11)(B00 + B11)
    # m2 = (A10 + A11)B00
    # m3 = A00(B10 - B11)
    # m4 = A11(B01 - B00)
    # m5 = (A00 + A01)B11
    # m6 = (A10 - A00)(B00 + B01)
    # m7 = (A01 - A11)(B10 + B11)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    #
    # For the 1x2x3 part (which is just a linear map from R^2 to R^3):
    # This has rank 2 (or we can use rank 6 as trivial)
    # Actually, for 1x2x3, the tensor is T[i,j,k] where i in {0}, j in {0,1}, k in {0,1,2}
    # T[0, j, k] = 1 if j == k else 0
    # This is a diagonal-like structure with rank 2 (since j ranges over 2 values)
    # Wait, let me reconsider. For 1x2x3:
    # T[0, j, k] = 1 if j == k
    # j in {0,1}, k in {0,1,2}
    # So T[0,0,0]=1, T[0,1,1]=1, all others 0
    # This has rank 2.
    #
    # But the Kronecker product approach: if we decompose 2x4x6 as (2x2x2) ⊗ (1x2x3),
    # the rank would be rank(2x2x2) * rank(1x2x3) = 7 * 2 = 14
    # But wait, 4 = 2*2 and 6 = 2*3, so this decomposition makes sense.
    #
    # Let me verify: n=2=2*1, m=4=2*2, p=6=2*3
    # So we're decomposing as (n1=2, m1=2, p1=2) ⊗ (n2=1, m2=2, p2=3)
    #
    # The tensor indices:
    # For the full problem: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # Decompose: i = i1*n2 + i2, j = j1*m2 + j2, k = k1*p2 + k2
    # where i1 in {0,1}, i2 in {0}, j1 in {0,1}, j2 in {0,1}, k1 in {0,1}, k2 in {0,1,2}
    #
    # T_full[i*m+j, j*p+k, k*n+i] = 1
    # = T_2x2x2[i1*m1+j1, j1*p1+k1, k1*n1+i1] * T_1x2x3[i2*m2+j2, j2*p2+k2, k2*n2+i2]
    #
    # Let's build the factors using Kronecker products.
    
    # Strassen decomposition for 2x2x2
    # U_strassen has shape (2*2, 7) = (4, 7)
    # V_strassen has shape (2*2, 7) = (4, 7)  
    # W_strassen has shape (2*2, 7) = (4, 7)
    
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # m1 = (A00 + A11)(B00 + B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1  # A00, A11
    V_s[0, 0] = 1; V_s[3, 0] = 1  # B00, B11
    W_s[0, 0] = 1; W_s[3, 0] = 1  # C00, C11
    
    # m2 = (A10 + A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1  # A10, A11
    V_s[0, 1] = 1  # B00
    W_s[2, 1] = 1; W_s[3, 1] = 1  # C10, C11
    
    # m3 = A00(B10 - B11)
    U_s[0, 2] = 1  # A00
    V_s[2, 2] = 1; V_s[3, 2] = -1  # B10, -B11
    W_s[1, 2] = 1; W_s[3, 2] = 1  # C01, C11
    
    # m4 = A11(B01 - B00)
    U_s[3, 3] = 1  # A11
    V_s[1, 3] = 1; V_s[0, 3] = -1  # B01, -B00
    W_s[0, 3] = 1; W_s[2, 3] = 1  # C00, C10
    
    # m5 = (A00 + A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1  # A00, A01
    V_s[3, 4] = 1  # B11
    W_s[0, 4] = -1; W_s[1, 4] = 1  # -C00, C01
    
    # m6 = (A10 - A00)(B00 + B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1  # A10, -A00
    V_s[0, 5] = 1; V_s[1, 5] = 1  # B00, B01
    W_s[3, 5] = 1  # C11
    
    # m7 = (A01 - A11)(B10 + B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1  # A01, -A11
    V_s[2, 6] = 1; V_s[3, 6] = 1  # B10, B11
    W_s[0, 6] = 1  # C00
    
    # Verify Strassen decomposition
    T_s = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
    T_s_expected = np.zeros((4, 4, 4))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T_s_expected[i*2+j, j*2+k, k*2+i] = 1.0
    
    if not np.allclose(T_s, T_s_expected):
        # Fallback to schoolbook
        return _schoolbook_solution(n, m, p)
    
    # 1x2x3 decomposition (rank 2)
    # T_123[0, j, k] = 1 if j == k else 0, j in {0,1}, k in {0,1,2}
    # So T_123[0,0,0] = 1, T_123[0,1,1] = 1
    # This can be decomposed as rank 2:
    # Term 1: U[0]=1, V[0]=1, W[0]=1 (for j=0,k=0)
    # Term 2: U[0]=1, V[1]=1, W[1]=1 (for j=1,k=1)
    
    U_123 = np.zeros((1*2, 2))  # shape (2, 2)
    V_123 = np.zeros((2*3, 2))  # shape (6, 2)
    W_123 = np.zeros((3*1, 2))  # shape (3, 2)
    
    # Term 0: j=0, k=0
    U_123[0, 0] = 1  # i2=0
    V_123[0, 0] = 1  # j2=0, k2=0 -> index 0*3+0=0
    W_123[0, 0] = 1  # k2=0, i2=0 -> index 0*1+0=0
    
    # Term 1: j=1, k=1
    U_123[1, 1] = 1  # i2=0 (but indexed as 0*1+0=0? Wait, shape is (m2, rank) = (2, 2))
    # Actually U_123 has shape (n2*m2, rank) = (1*2, 2) = (2, 2)
    # Index is i2*m2 + j2 = 0*2 + j2 = j2
    # So U_123[j2, r]
    U_123[0, 1] = 1  # j2=0 for term 1? No, j2=1 for term 1
    U_123[1, 1] = 1  # j2=1
    
    V_123[1*3+1, 1] = 1  # j2=1, k2=1 -> index 1*3+1=4
    W_123[1*1+0, 1] = 1  # k2=1, i2=0 -> index 1*1+0=1
    
    # Verify 1x2x3 decomposition
    T_123 = np.einsum('ar,br,cr->abc', U_123, V_123, W_123)
    T_123_expected = np.zeros((2, 6, 3))
    for j in range(2):
        for k in range(3):
            if j == k:
                T_123_expected[0*2+j, j*3+k, k*1+0] = 1.0
    
    if not np.allclose(T_123, T_123_expected):
        return _schoolbook_solution(n, m, p)
    
    # Now combine using Kronecker product structure
    # Full rank = 7 * 2 = 14
    # But wait, we need to be careful about the index mapping.
    # 
    # Full indices:
    # a = i*m + j = i*4 + j, where i = i1*1 + i2, j = j1*2 + j2
    #   = i1*4 + i2*4 + j1*2 + j2 = i1*4 + j1*2 + i2*4 + j2
    #   Hmm, this doesn't factor cleanly. Let me reconsider.
    #
    # Actually, for the Kronecker product to work, we need:
    # n = n1 * n2, m = m1 * m2, p = p1 * p2
    # Here n=2=2*1, m=4=2*2, p=6=2*3
    # 
    # The index mapping should be:
    # i = i1 + n2*i2 = i1 + 1*i2 = i1 + i2 (but i2 is always 0 since n2=1)
    # j = j1 + m2*j2 = j1 + 2*j2
    # k = k1 + p2*k2 = k1 + 3*k2
    #
    # For the U factor (shape n*m, rank):
    # Index a = i*m + j = i*4 + j
    # With i = i1 + i2 (i2=0), j = j1 + 2*j2
    # a = i1*4 + j1 + 2*j2
    #
    # The Kronecker product structure for U:
    # U_full[a, r1*r2+r2_idx] = U_s[i1*m1+j1, r1] * U_123[i2*m2+j2, r2]
    # But i2=0 always, so i2*m2+j2 = j2
    # And a = i1*4 + j1 + 2*j2
    # 
    # Let me just build it directly by iterating.
    
    total_rank = 7 * 2  # 14
    U_full = np.zeros((n*m, total_rank))
    V_full = np.zeros((m*p, total_rank))
    W_full = np.zeros((p*n, total_rank))
    
    for r1 in range(7):
        for r2 in range(2):
            r = r1 * 2 + r2
            for i1 in range(2):
                for i2 in range(1):
                    i = i1 + i2  # i1*1 + i2
                    for j1 in range(2):
                        for j2 in range(2):
                            j = j1 + 2*j2  # j1 + 2*j2
                            a = i * m + j
                            U_full[a, r] = U_s[i1*2+j1, r1] * U_123[j2, r2]
                        
                        for k1 in range(2):
                            for k2 in range(3):
                                k = k1 + 3*k2
                                b = j * p + k
                                V_full[b, r] = V_s[j1*2+k1, r1] * V_123[j2*3+k2, r2]
                            
                            for i1w in range(2):
                                for i2w in range(1):
                                    iw = i1w + i2w
                                    c = k * n + iw
                                    W_full[c, r] = W_s[k1*2+i1w, r1] * W_123[k2*1+i2w, r2]
    
    # Verify
    T_comp = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
    if np.allclose(T_comp, T, atol=1e-10):
        best_rank = total_rank
        best_U = U_full
        best_V = V_full
        best_W = W_full
    else:
        # Fallback to schoolbook
        return _schoolbook_solution(n, m, p)
    
    # Convert to required format (list of lists)
    rank = best_rank
    U_out = [[0.0] * rank for _ in range(n * m)]
    V_out = [[0.0] * rank for _ in range(m * p)]
    W_out = [[0.0] * rank for _ in range(p * n)]
    
    for a in range(n * m):
        for r in range(rank):
            U_out[a][r] = float(best_U[a, r])
    for b in range(m * p):
        for r in range(rank):
            V_out[b][r] = float(best_V[b, r])
    for c in range(p * n):
        for r in range(rank):
            W_out[c][r] = float(best_W[c, r])
    
    return {"U": U_out, "V": V_out, "W": W_out}

def _schoolbook_solution(n, m, p):
    """Fallback schoolbook solution"""
    schoolbook_rank = n * m * p
    U_sb = np.zeros((n*m, schoolbook_rank))
    V_sb = np.zeros((m*p, schoolbook_rank))
    W_sb = np.zeros((p*n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    rank = schoolbook_rank
    U_out = [[0.0] * rank for _ in range(n * m)]
    V_out = [[0.0] * rank for _ in range(m * p)]
    W_out = [[0.0] * rank for _ in range(p * n)]
    
    for a in range(n * m):
        for r in range(rank):
            U_out[a][r] = float(U_sb[a, r])
    for b in range(m * p):
        for r in range(rank):
            V_out[b][r] = float(V_sb[b, r])
    for c in range(p * n):
        for r in range(rank):
            W_out[c][r] = float(W_sb[c, r])
    
    return {"U": U_out, "V": V_out, "W": W_out}
