"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for Jb in range(2):
        for Kb in range(3):
            j0 = Jb * 2
            j1 = Jb * 2 + 1
            k0 = Kb * 2
            k1 = Kb * 2 + 1
            
            # M1 = (A[0,j0]+A[1,j1])*(B[j0,k0]+B[j1,k1])
            U[0*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # M2 = (A[1,j0]+A[1,j1])*B[j0,k0]
            U[1*m + j0][col] += 1
            U[1*m + j1][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 1][col] += 1
            W[k1*n + 1][col] -= 1
            col += 1
            
            # M3 = A[0,j0]*(B[j0,k1]-B[j1,k1])
            U[0*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            V[j1*p + k1][col] -= 1
            W[k1*n + 0][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # M4 = A[1,j1]*(B[j1,k0]-B[j0,k0])
            U[1*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            V[j0*p + k0][col] -= 1
            W[k0*n + 0][col] += 1
            W[k0*n + 1][col] += 1
            col += 1
            
            # M5 = (A[0,j0]+A[0,j1])*B[j1,k1]
            U[0*m + j0][col] += 1
            U[0*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] -= 1
            W[k1*n + 0][col] += 1
            col += 1
            
            # M6 = (A[1,j0]-A[0,j0])*(B[j0,k0]+B[j0,k1])
            U[1*m + j0][col] += 1
            U[0*m + j0][col] -= 1
            V[j0*p + k0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 1][col] += 1
            col += 1
            
            # M7 = (A[0,j1]-A[1,j1])*(B[j1,k0]+B[j1,k1])
            U[0*m + j1][col] += 1
            U[1*m + j1][col] -= 1
            V[j1*p + k0][col] += 1
            V[j1*p + k1][col] += 1
            W[k0*n + 0][col] += 1
            col += 1
            
            # S1 = A[2,j0]*B[j0,k0]
            U[2*m + j0][col] += 1
            V[j0*p + k0][col] += 1
            W[k0*n + 2][col] += 1
            col += 1
            
            # S2 = A[2,j0]*B[j0,k1]
            U[2*m + j0][col] += 1
            V[j0*p + k1][col] += 1
            W[k1*n + 2][col] += 1
            col += 1
            
            # S3 = A[2,j1]*B[j1,k0]
            U[2*m + j1][col] += 1
            V[j1*p + k0][col] += 1
            W[k0*n + 2][col] += 1
            col += 1
            
            # S4 = A[2,j1]*B[j1,k1]
            U[2*m + j1][col] += 1
            V[j1*p + k1][col] += 1
            W[k1*n + 2][col] += 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
