"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Term 0: M1 = (A11+A22)(B11+B22)
                # U: A[i*2+0,j*2+0]=1, A[i*2+1,j*2+1]=1
                # V: B[j*2+0,k*2+0]=1, B[j*2+1,k*2+1]=1
                # W: C[i*2+0,k*2+0]=+1, C[i*2+1,k*2+1]=+1
                U[i*2+0][col] += 1
                U[i*2+1][col] += 1
                V[j*2+0][col] += 1
                V[j*2+1][col] += 1
                W[(k*2+0)*n + (i*2+0)][col] += 1
                W[(k*2+1)*n + (i*2+1)][col] += 1
                col += 1
                
                # Term 1: M2 = (A21+A22)B11
                # U: A[i*2+1,j*2+0]=1, A[i*2+1,j*2+1]=1
                # V: B[j*2+0,k*2+0]=1
                # W: C[i*2+1,k*2+0]=+1, C[i*2+1,k*2+1]=-1
                U[i*2+1][col] += 1
                U[i*2+1][col] += 1
                V[j*2+0][col] += 1
                W[(k*2+0)*n + (i*2+1)][col] += 1
                W[(k*2+1)*n + (i*2+1)][col] += -1
                col += 1
                
                # Term 2: M3 = A11(B21-B22)
                # U: A[i*2+0,j*2+0]=1
                # V: B[j*2+1,k*2+0]=1, B[j*2+1,k*2+1]=-1
                # W: C[i*2+0,k*2+1]=+1, C[i*2+1,k*2+1]=+1
                U[i*2+0][col] += 1
                V[j*2+1][col] += 1
                V[j*2+1][col] += -1
                W[(k*2+1)*n + (i*2+0)][col] += 1
                W[(k*2+1)*n + (i*2+1)][col] += 1
                col += 1
                
                # Term 3: M4 = A22(B11-B21)
                # U: A[i*2+1,j*2+1]=1
                # V: B[j*2+0,k*2+0]=1, B[j*2+1,k*2+0]=-1
                # W: C[i*2+0,k*2+0]=+1, C[i*2+1,k*2+0]=+1
                U[i*2+1][col] += 1
                V[j*2+0][col] += 1
                V[j*2+1][col] += -1
                W[(k*2+0)*n + (i*2+0)][col] += 1
                W[(k*2+0)*n + (i*2+1)][col] += 1
                col += 1
                
                # Term 4: M5 = (A11+A12)B22
                # U: A[i*2+0,j*2+0]=1, A[i*2+0,j*2+1]=1
                # V: B[j*2+1,k*2+1]=1
                # W: C[i*2+0,k*2+0]=-1, C[i*2+0,k*2+1]=+1
                U[i*2+0][col] += 1
                U[i*2+0][col] += 1
                V[j*2+1][col] += 1
                W[(k*2+0)*n + (i*2+0)][col] += -1
                W[(k*2+1)*n + (i*2+0)][col] += 1
                col += 1
                
                # Term 5: M6 = (A21-A11)(B11+B12)
                # U: A[i*2+1,j*2+0]=1, A[i*2+0,j*2+0]=-1
                # V: B[j*2+0,k*2+0]=1, B[j*2+0,k*2+1]=1
                # W: C[i*2+1,k*2+1]=+1
                U[i*2+1][col] += 1
                U[i*2+0][col] += -1
                V[j*2+0][col] += 1
                V[j*2+0][col] += 1
                W[(k*2+1)*n + (i*2+1)][col] += 1
                col += 1
                
                # Term 6: M7 = (A12-A22)(B21+B22)
                # U: A[i*2+0,j*2+1]=1, A[i*2+1,j*2+1]=-1
                # V: B[j*2+1,k*2+0]=1, B[j*2+1,k*2+1]=1
                # W: C[i*2+0,k*2+0]=+1, C[i*2+1,k*2+1]=+1
                U[i*2+0][col] += 1
                U[i*2+1][col] += -1
                V[j*2+1][col] += 1
                V[j*2+1][col] += 1
                W[(k*2+0)*n + (i*2+0)][col] += 1
                W[(k*2+1)*n + (i*2+1)][col] += 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
