"""Matrix multiplication 4x4x8 tensor decomposition."""
import time

DIMENSIONS = [4, 4, 8]

def _strassen_2x2():
    """Return rank-7 Strassen decomposition for 2x2x2.
    U: (4,7), V: (4,7), W: (4,7)
    Convention: C[k*2+i, r] = sum_s U[i*2+j, s] * V[j*2+k, s] * W[k*2+i, s]
    where A is 2x2 (i rows, j cols), B is 2x2 (j rows, k cols), C is 2x2 (k rows, i cols)
    """
    # Strassen's 7 products for 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U[i*2+j, s]: coefficient of A[i,j] in product s
    # V[j*2+k, s]: coefficient of B[j,k] in product s
    # W[k*2+i, s]: coefficient of C[k,i] in product s
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Product 0: M1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # A00, A11
    V[0][0] = 1; V[3][0] = 1  # B00, B11
    W[0][0] = 1; W[3][0] = 1  # C00, C11 (k=0,i=0 and k=1,i=1)
    
    # Product 1: M2 = (A10+A11)B00
    U[2][1] = 1; U[3][1] = 1  # A10, A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10 (k=1,i=0)
    
    # Product 2: M3 = A00(B01-B11)
    U[0][2] = 1  # A00
    V[1][2] = 1; V[3][2] = -1  # B01, -B11
    W[1][2] = 1  # C01 (k=0,i=1)
    
    # Product 3: M4 = A11(B10-B00)
    U[3][3] = 1  # A11
    V[2][3] = 1; V[0][3] = -1  # B10, -B00
    W[0][3] = 1; W[2][3] = 1  # C00, C10
    
    # Product 4: M5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1  # A00, A01
    V[3][4] = 1  # B11
    W[0][4] = -1; W[1][4] = 1  # -C00, C01
    
    # Product 5: M6 = (A10-A00)(B00+B01)
    U[2][5] = 1; U[0][5] = -1  # A10, -A00
    V[0][5] = 1; V[1][5] = 1  # B00, B01
    W[3][5] = 1  # C11 (k=1,i=1)
    
    # Product 6: M7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1  # A01, -A11
    V[2][6] = 1; V[3][6] = 1  # B10, B11
    W[0][6] = 1; W[3][6] = 1  # C00, C11
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    start_time = time.time()
    
    # Strategy: Split 4x4x8 into two 4x4x4 blocks sharing A
    # Block 1: A(4x4) * B_left(4x4) -> C_left(4x4)
    # Block 2: A(4x4) * B_right(4x4) -> C_right(4x4)
    # Use rank-49 decomposition for each 4x4x4 (tensor product of two Strassen 2x2x2)
    
    # Get Strassen 2x2x2
    U2, V2, W2 = _strassen_2x2()
    
    # Build 4x4x4 decomposition by tensor product of two 2x2x2
    # A is 4x4: A[i*2+i2, j*2+j2] where i,i2 in {0,1}, j,j2 in {0,1}
    # B is 4x4: B[j*2+j2, k*2+k2]
    # C is 4x4: C[k*2+k2, i*2+i2]
    # Product index: s = s1*7 + s2 where s1,s2 in {0,...,6}
    # U4[i*4+j, s] = U2[i*2+j2, s1] * U2[i2*2+j, s2] ... 
    
    # Actually, tensor product: 
    # For 4x4x4 = (2x2x2) ⊗ (2x2x2):
    # U4[(i,i2),(j,j2), (s1,s2)] = U2[i,j,s1] * U2[i2,j2,s2]
    # V4[(j,j2),(k,k2), (s1,s2)] = V2[j,k,s1] * V2[j2,k2,s2]
    # W4[(k,k2),(i,i2), (s1,s2)] = W2[k,i,s1] * W2[k2,i2,s2]
    
    rank_444 = 7 * 7  # 49
    
    def build_444():
        U4 = [[0]*rank_444 for _ in range(16)]
        V4 = [[0]*rank_444 for _ in range(16)]
        W4 = [[0]*rank_444 for _ in range(16)]
        
        for i in range(2):
            for i2 in range(2):
                for j in range(2):
                    for j2 in range(2):
                        for k in range(2):
                            for k2 in range(2):
                                for s1 in range(7):
                                    for s2 in range(7):
                                        s = s1 * 7 + s2
                                        u_val = U2[i*2+j][s1] * U2[i2*2+j2][s2]
                                        v_val = V2[j*2+k][s1] * V2[j2*2+k2][s2]
                                        w_val = W2[k*2+i][s1] * W2[k2*2+i2][s2]
                                        
                                        U4[(i*2+i2)*4 + (j*2+j2)][s] += u_val
                                        V4[(j*2+j2)*4 + (k*2+k2)][s] += v_val
                                        W4[(k*2+k2)*4 + (i*2+i2)][s] += w_val
        return U4, V4, W4
    
    U4, V4, W4 = build_444()
    
    # Now build 4x4x8:
    # A is 4x4 (16 entries)
    # B is 4x8 (32 entries): B_left = B[:,0:4], B_right = B[:,4:8]
    # C is 8x4 (32 entries): C_left = C[0:4,:], C_right = C[4:8,:]
    # W convention: W[k*n+i, r] where k in 0..7, i in 0..3
    
    total_rank = 2 * rank_444  # 98
    
    U = [[0]*total_rank for _ in range(16)]   # 4*4 = 16
    V = [[0]*total_rank for _ in range(32)]   # 4*8 = 32
    W = [[0]*total_rank for _ in range(32)]   # 8*4 = 32
    
    # Block 1: C_left = A * B_left
    # B_left[j, k] for j in 0..3, k in 0..3 -> V4[j*4+k, s]
    # C_left[k, i] for k in 0..3, i in 0..3 -> W4[k*4+i, s]
    for row in range(16):
        for s in range(rank_444):
            U[row][s] = U4[row][s]
    
    for j in range(4):
        for k in range(4):
            for s in range(rank_444):
                V[j*4 + k][s] = V4[j*4 + k][s]
    
    for k in range(4):
        for i in range(4):
            for s in range(rank_444):
                W[k*4 + i][s] = W4[k*4 + i][s]
    
    # Block 2: C_right = A * B_right
    # B_right[j, k] for j in 0..3, k in 0..3 -> B[j, 4+k]
    # C_right[k, i] for k in 0..3, i in 0..3 -> C[4+k, i]
    offset = rank_444
    for row in range(16):
        for s in range(rank_444):
            U[row][s + offset] = U4[row][s]
    
    for j in range(4):
        for k in range(4):
            for s in range(rank_444):
                V[j*4 + (4 + k)][s + offset] = V4[j*4 + k][s]
    
    for k in range(4):
        for i in range(4):
            for s in range(rank_444):
                W[(4 + k)*4 + i][s + offset] = W4[k*4 + i][s]
    
    elapsed = time.time() - start_time
    if elapsed > 110:
        # Safety: return what we have
        pass
    
    return {"U": U, "V": V, "W": W}
