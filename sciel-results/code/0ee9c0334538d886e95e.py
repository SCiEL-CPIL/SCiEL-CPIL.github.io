def construct():
    n, m, p = 2, 3, 5
    rank = 20
    
    U = [[0.0] * rank for _ in range(n * m)]  # 6 x 20
    V = [[0.0] * rank for _ in range(m * p)]  # 15 x 20
    W = [[0.0] * rank for _ in range(p * n)]  # 10 x 20
    
    col = 0
    # Block 1: A1 * B1[:,0:2] where A1 = A[0:2,0:2], B1 = B[0:2,0:2]
    # Strassen 2x2x2
    # A indices: i=0,1; j=0,1
    # B indices: j=0,1; k=0,1
    # C indices: i=0,1; k=0,1
    
    # M1 = (A00+A11)*(B00+B11)
    U[0*2+0][col] = 1; U[1*2+1][col] = 1
    V[0*5+0][col] = 1; V[1*5+1][col] = 1
    W[0*2+0][col] = 1; W[1*2+0][col] = -1; W[0*2+1][col] = 1; W[1*2+1][col] = 1
    # C00 += M1, C01 += 0, C10 += 0, C11 += M1... wait
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    # In 0-indexed: C[0,0]=M1+M4-M5+M7, C[0,1]=M3+M5, C[1,0]=M2+M4, C[1,1]=M1-M2+M3+M6
    
    col += 1
    # M2 = (A10+A11)*B00
    U[1*2+0][col] = 1; U[1*2+1][col] = 1
    V[0*5+0][col] = 1
    W[1*2+0][col] = 1
    col += 1
    # M3 = A00*(B01-B11)
    U[0*2+0][col] = 1
    V[0*5+1][col] = 1; V[1*5+1][col] = -1
    W[0*2+1][col] = 1
    col += 1
    # M4 = A11*(B10-B00)
    U[1*2+1][col] = 1
    V[1*5+0][col] = 1; V[0*5+0][col] = -1
    W[0*2+0][col] = 1; W[1*2+0][col] = 1
    col += 1
    # M5 = (A00+A01)*B11
    U[0*2+0][col] = 1; U[0*2+1][col] = 1
    V[1*5+1][col] = 1
    W[0*2+0][col] = -1; W[1*2+1][col] = 1
    col += 1
    # M6 = (A10-A00)*(B00+B01)
    U[1*2+0][col] = 1; U[0*2+0][col] = -1
    V[0*5+0][col] = 1; V[0*5+1][col] = 1
    W[1*2+1][col] = 1
    col += 1
    # M7 = (A01-A11)*(B10+B11)
    U[0*2+1][col] = 1; U[1*2+1][col] = -1
    V[1*5+0][col] = 1; V[1*5+1][col] = 1
    W[0*2+0][col] = 1
    col += 1
    # col is now 7
    
    # Block 2: A1 * B1[:,2:4], same Strassen but B columns offset by 2
    for dk in range(2):  # dk=0,1 correspond to k=2,3
        pass
    # Actually let me just replicate the Strassen with B column offsets
    
    # M1' = (A00+A11)*(B0,2+B1,3)
    U[0*2+0][col] = 1; U[1*2+1][col] = 1
    V[0*5+2][col] = 1; V[1*5+3][col] = 1
    W[0*2+2][col] = 1; W[1*2+2][col] = -1; W[0*2+3][col] = 1; W[1*2+3][col] = 1
    col += 1
    # M2' = (A10+A11)*B0,2
    U[1*2+0][col] = 1; U[1*2+1][col] = 1
    V[0*5+2][col] = 1
    W[1*2+2][col] = 1
    col += 1
    # M3' = A00*(B0,3-B1,3)
    U[0*2+0][col] = 1
    V[0*5+3][col] = 1; V[1*5+3][col] = -1
    W[0*2+3][col] = 1
    col += 1
    # M4' = A11*(B1,2-B0,2)
    U[1*2+1][col] = 1
    V[1*5+2][col] = 1; V[0*5+2][col] = -1
    W[0*2+2][col] = 1; W[1*2+2][col] = 1
    col += 1
    # M5' = (A00+A01)*B1,3
    U[0*2+0][col] = 1; U[0*2+1][col] = 1
    V[1*5+3][col] = 1
    W[0*2+2][col] = -1; W[1*2+3][col] = 1
    col += 1
    # M6' = (A10-A00)*(B0,2+B0,3)
    U[1*2+0][col] = 1; U[0*2+0][col] = -1
    V[0*5+2][col] = 1; V[0*5+3][col] = 1
    W[1*2+3][col] = 1
    col += 1
    # M7' = (A01-A11)*(B1,2+B1,3)
    U[0*2+1][col] = 1; U[1*2+1][col] = -1
    V[1*5+2][col] = 1; V[1*5+3][col] = 1
    W[0*2+2][col] = 1
    col += 1
    # col is now 14
    
    # Block 3: A1 * B1[:,4:5] (2x2 x 2x1), rank 4
    # C[i,4] = A[i,0]*B[0,4] + A[i,1]*B[1,4]
    # 4 terms
    for i in range(2):
        for j in range(2):
            U[i*2+j][col] = 1
            V[j*5+4][col] = 1
            W[4*2+i][col] = 1
            col += 1
    # col is now 18
    
    # Block 4: A2 * B2 (2x1 x 1x5), rank 2
    # C[i,4] += A[i,2]*B[2,k] for k=0..4
    # Wait, B2 is row 2 of B, which is B[2,0..4]
    # C[i,k] += A[i,2] * B[2,k]
    # This is outer product: for each i, C[i,:] += A[i,2] * B[2,:]
    # Rank 2: one term for i=0, one for i=1
    # Term for i=0: U[0*2+2]=1, V[2*5+k]=1 for all k, W[k*2+0]=1 for all k
    # But that's 5 rank-1 terms if done naively. 
    # Actually the tensor T[i,2,k] = 1 for i=0,1; k=0..4. 
    # This factors as (e0+e1) in i-space ⊗ e2 in j-space ⊗ (sum of e_k) in k-space
    # So it's rank 1! Wait no, it's a single rank-1 tensor.
    
    # Hmm, let me reconsider. The sub-tensor for A2*B2 has T[i,j,k] nonzero only when j=2.
    # T[i,2,k] = 1 for all i,k.
    # This is indeed rank 1: it's u ⊗ v where u[i] = 1 for i=0,1 (and 0 otherwise in the full space),
    # v[j] = 1 for j=2, and w[k] = 1 for all k.
    # So just 1 rank-1 term!
    
    U[0*2+2][col] = 1; U[1*2+2][col] = 1
    for k in range(5):
        V[2*5+k][col] = 1
    for k in range(5):
        W[k*2+0][col] = 1; W[k*2+1][col] = 1
    col += 1
    # col is now 19
    
    # Wait, that's only 19. Let me recount.
    # Block 1: 7 terms (cols 0-6)
    # Block 2: 7 terms (cols 7-13)  
    # Block 3: 4 terms (cols 14-17)
    # Block 4: 1 term (col 18)
    # Total: 19
    
    # But I set rank=20. Let me fix.
    return {"U": U, "V": V, "W": W}
