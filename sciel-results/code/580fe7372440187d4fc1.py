import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 6 x 22
    V = [[0] * rank for _ in range(m * p)]  # 18 x 22
    W = [[0] * rank for _ in range(p * n)]  # 12 x 22
    
    col = 0
    
    # Part 1: A[:,0] * B[0,:] -> C[:,k] for all k
    # This is a 2x1 x 1x6 outer product, rank 1
    # T[i,0,k] = A[i,0]*B[0,k], need T[i,0,k]=1 for all i,k
    # Use: U[i,0]=1, V[0,0]=1, W[k,0]=1
    for i in range(n):
        U[i][col] = 1
    V[0][col] = 1  # V[0] corresponds to B[0,0], B[0,1], ..., B[0,5] are V[0..5]
    # Actually V[j*p + c] for B[j,c]. So B[0,k] is V[k].
    # We need V[j, col] to pick out B[0,k] for all k. 
    # But V is indexed by (j,c) -> j*p+c. For j=0, c=k: V[k].
    # We need V[j, col] * W[c, col] to give the right pattern.
    # For this term: U[i,col] * V[j,col] * W[c,col] should equal 1 if j=0, else 0.
    # Set V[j,col] = 1 if j=0 else 0, W[c,col] = 1 for all c.
    # Then U[i,col] * 1 * 1 = U[i,col]. Set U[i,col]=1.
    for j in range(m):
        for c in range(p):
            if j == 0:
                V[c][col] = 1  # V[j*p+c] = V[0*6+c] = V[c]
    for c in range(p):
        for i in range(n):
            W[c*n + i][col] = 1  # W[k*n+i] for C[i,k], so W index = k*n + i
    
    col += 1  # col is now 1
    
    # Part 2: A[:,1:3] * B[1:3,:] -> C[:,k] for k=0..5
    # Split into 3 blocks of 2 columns each
    # A[:,1:3] is 2x2: A[0,1], A[0,2], A[1,1], A[1,2]
    # U indices: A[i,j] -> U[i*m + j] = U[i*3 + j]
    # A[0,1] -> U[1], A[0,2] -> U[2], A[1,1] -> U[4], A[1,2] -> U[5]
    
    for x in range(3):
        c_start = 2 * x  # columns 2x, 2x+1 of C and B
        
        # B[1:3, 2x:2x+2] is 2x2:
        # B[1,2x] -> V[1*6+2x] = V[6+2x]
        # B[1,2x+1] -> V[6+2x+1]
        # B[2,2x] -> V[2*6+2x] = V[12+2x]
        # B[2,2x+1] -> V[12+2x+1]
        
        # C[:, 2x:2x+2] is 2x2:
        # C[0,2x] -> W[2x*2+0] = W[4x]
        # C[0,2x+1] -> W[4x+2]
        # C[1,2x] -> W[4x+1]
        # C[1,2x+1] -> W[4x+3]
        
        # Strassen 7-term decomposition for 2x2x2
        # Let a00=A[0,1], a01=A[0,2], a10=A[1,1], a11=A[1,2]
        # Let b00=B[1,2x], b01=B[1,2x+1], b10=B[2,2x], b11=B[2,2x+1]
        
        # M1 = (a00+a11)*(b00+b11)
        U[1][col] = 1; U[5][col] = 1
        V[6+c_start][col] = 1; V[12+c_start+1][col] = 1
        W[4*x][col] = 1; W[4*x+3][col] = 1
        col += 1
        
        # M2 = (a10+a11)*b00
        U[4][col] = 1; U[5][col] = 1
        V[6+c_start][col] = 1
        W[4*x+1][col] = 1; W[4*x+3][col] = -1
        col += 1
        
        # M3 = a00*(b01-b11)
        U[1][col] = 1
        V[6+c_start+1][col] = 1; V[12+c_start+1][col] = -1
        W[4*x+2][col] = 1; W[4*x+3][col] = 1
        col += 1
        
        # M4 = a11*(b10-b00)
        U[5][col] = 1
        V[12+c_start][col] = 1; V[6+c_start][col] = -1
        W[4*x][col] = 1; W[4*x+1][col] = 1
        col += 1
        
        # M5 = (a00+a01)*b11
        U[1][col] = 1; U[2][col] = 1
        V[12+c_start+1][col] = 1
        W[4*x][col] = -1; W[4*x+2][col] = 1
        col += 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[4][col] = 1; U[1][col] = -1
        V[6+c_start][col] = 1; V[6+c_start+1][col] = 1
        W[4*x+3][col] = 1
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[2][col] = 1; U[5][col] = -1
        V[12+c_start][col] = 1; V[12+c_start+1][col] = 1
        W[4*x][col] = 1
        col += 1
    
    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_est = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    T_target = np.zeros((n*m, m*p, p*n))
    for r in range(n):
        for s in range(m):
            for c in range(p):
                T_target[r*m+s, s*p+c, c*n+r] = 1
    if np.max(np.abs(T_est - T_target)) > 1e-9:
        # Fallback to rank 33
        rank = 33
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for k in range(p):
            U[0][col] = 1; V[k][col] = 1; W[2*k][col] = 1; col += 1
            U[3][col] = 1; V[k][col] = 1; W[2*k + 1][col] = 1; col += 1
        for x in range(3):
            cs = 2 * x
            U[1][col]=1; U[5][col]=1; V[6+cs][col]=1; V[12+cs+1][col]=1; W[2*cs][col]=1; W[2*cs+3][col]=1; col+=1
            U[4][col]=1; U[5][col]=1; V[6+cs][col]=1; W[2*cs+1][col]=1; W[2*cs+3][col]=-1; col+=1
            U[1][col]=1; V[6+cs+1][col]=1; V[12+cs+1][col]=-1; W[2*cs+2][col]=1; W[2*cs+3][col]=1; col+=1
            U[5][col]=1; V[12+cs][col]=1; V[6+cs][col]=-1; W[2*cs][col]=1; W[2*cs+1][col]=1; col+=1
            U[1][col]=1; U[2][col]=1; V[12+cs+1][col]=1; W[2*cs][col]=-1; W[2*cs+2][col]=1; col+=1
            U[4][col]=1; U[1][col]=-1; V[6+cs][col]=1; V[6+cs+1][col]=1; W[2*cs+3][col]=1; col+=1
            U[2][col]=1; U[5][col]=-1; V[12+cs][col]=1; V[12+cs+1][col]=1; W[2*cs][col]=1; col+=1
    
    return {"U": U, "V": V, "W": W}
