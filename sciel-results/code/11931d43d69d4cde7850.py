"""Matrix multiplication 2x4x8 with rank-56 Strassen decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Use Strassen's rank-7 decomposition for each 2x2x2 block
    # Split m=4 into 2 blocks of 2 (j0 in {0, 2})
    # Split p=8 into 4 blocks of 2 (k0 in {0, 2, 4, 6})
    # Total: 2 * 4 = 8 blocks, each rank 7, total rank 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # shape (8, 56)
    V = [[0] * rank for _ in range(m * p)]  # shape (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # shape (16, 56)
    
    col = 0
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Strassen terms for 2x2x2 block
            # Local indices: i in {0,1}, j in {j0, j0+1}, k in {k0, k0+1}
            
            # M1 = (A00 + A11)(B00 + B11)
            # U: A[i=0,j=j0]=1, A[i=1,j=j0+1]=1
            # V: B[j=j0,k=k0]=1, B[j=j0+1,k=k0+1]=1
            # W: C[k=k0,i=0]=1, C[k=k0+1,i=1]=1
            U[0 * m + j0][col] += 1
            U[1 * m + (j0 + 1)][col] += 1
            V[j0 * p + k0][col] += 1
            V[(j0 + 1) * p + (k0 + 1)][col] += 1
            W[k0 * n + 0][col] += 1
            W[(k0 + 1) * n + 1][col] += 1
            col += 1
            
            # M2 = (A10 + A11)B00
            # U: A[i=1,j=j0]=1, A[i=1,j=j0+1]=1
            # V: B[j=j0,k=k0]=1
            # W: C[k=k0,i=1]=1, C[k=k0+1,i=1]=-1
            U[1 * m + j0][col] += 1
            U[1 * m + (j0 + 1)][col] += 1
            V[j0 * p + k0][col] += 1
            W[k0 * n + 1][col] += 1
            W[(k0 + 1) * n + 1][col] += -1
            col += 1
            
            # M3 = A00(B01 - B11)
            # U: A[i=0,j=j0]=1
            # V: B[j=j0,k=k0+1]=1, B[j=j0+1,k=k0+1]=-1
            # W: C[k=k0+1,i=0]=1, C[k=k0+1,i=1]=1
            U[0 * m + j0][col] += 1
            V[j0 * p + (k0 + 1)][col] += 1
            V[(j0 + 1) * p + (k0 + 1)][col] += -1
            W[(k0 + 1) * n + 0][col] += 1
            W[(k0 + 1) * n + 1][col] += 1
            col += 1
            
            # M4 = A11(B10 - B00)
            # U: A[i=1,j=j0+1]=1
            # V: B[j=j0+1,k=k0]=1, B[j=j0,k=k0]=-1
            # W: C[k=k0,i=0]=1, C[k=k0,i=1]=1
            U[1 * m + (j0 + 1)][col] += 1
            V[(j0 + 1) * p + k0][col] += 1
            V[j0 * p + k0][col] += -1
            W[k0 * n + 0][col] += 1
            W[k0 * n + 1][col] += 1
            col += 1
            
            # M5 = (A00 + A01)B11
            # U: A[i=0,j=j0]=1, A[i=0,j=j0+1]=1
            # V: B[j=j0+1,k=k0+1]=1
            # W: C[k=k0,i=0]=-1, C[k=k0+1,i=0]=1
            U[0 * m + j0][col] += 1
            U[0 * m + (j0 + 1)][col] += 1
            V[(j0 + 1) * p + (k0 + 1)][col] += 1
            W[k0 * n + 0][col] += -1
            W[(k0 + 1) * n + 0][col] += 1
            col += 1
            
            # M6 = (A10 - A00)(B00 + B01)
            # U: A[i=1,j=j0]=1, A[i=0,j=j0]=-1
            # V: B[j=j0,k=k0]=1, B[j=j0,k=k0+1]=1
            # W: C[k=k0+1,i=1]=1
            U[1 * m + j0][col] += 1
            U[0 * m + j0][col] += -1
            V[j0 * p + k0][col] += 1
            V[j0 * p + (k0 + 1)][col] += 1
            W[(k0 + 1) * n + 1][col] += 1
            col += 1
            
            # M7 = (A01 - A11)(B10 + B11)
            # U: A[i=0,j=j0+1]=1, A[i=1,j=j0+1]=-1
            # V: B[j=j0+1,k=k0]=1, B[j=j0+1,k=k0+1]=1
            # W: C[k=k0,i=0]=1
            U[0 * m + (j0 + 1)][col] += 1
            U[1 * m + (j0 + 1)][col] += -1
            V[(j0 + 1) * p + k0][col] += 1
            V[(j0 + 1) * p + (k0 + 1)][col] += 1
            W[k0 * n + 0][col] += 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
