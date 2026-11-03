"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Rank 55 decomposition
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, :]
    # Split B into 5 blocks of width 2
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen Multiplications for 2x2 * 2x2
        # M1 = (a+d)(e+h)
        # U: A[0,0]=1, A[1,1]=1
        # V: B[0,k0]=1, B[1,k1]=1
        # W: C[0,k0]=1, C[0,k1]=1, C[1,k0]=1, C[1,k1]=1 (with signs)
        
        # We need to define the 7 terms for this block
        # Term 1: (A00+A11)(B0k0+B1k1)
        U[0*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*10+k0][col] = 1
        V[1*10+k1][col] = 1
        # Contributions to C:
        # C0k0 += 1, C0k1 += 1, C1k0 += 1, C1k1 += 1
        W[k0*2+0][col] = 1
        W[k1*2+0][col] = 1
        W[k0*2+1][col] = 1
        W[k1*2+1][col] = 1
        col += 1
        
        # Term 2: (A10+A11)B0k0
        U[1*3+0][col] = 1
        U[1*3+1][col] = 1
        V[0*10+k0][col] = 1
        # C1k0 += 1
        W[k0*2+1][col] = 1
        col += 1
        
        # Term 3: A00(B0k1-B1k1)
        U[0*3+0][col] = 1
        V[0*10+k1][col] = 1
        V[1*10+k1][col] = -1
        # C0k1 += 1
        W[k1*2+0][col] = 1
        col += 1
        
        # Term 4: A11(B1k0-B0k0)
        U[1*3+1][col] = 1
        V[1*10+k0][col] = 1
        V[0*10+k0][col] = -1
        # C0k0 += 1, C1k0 += 1
        W[k0*2+0][col] = 1
        W[k0*2+1][col] = 1
        col += 1
        
        # Term 5: (A00+A01)B1k1
        U[0*3+0][col] = 1
        U[0*3+1][col] = 1
        V[1*10+k1][col] = 1
        # C0k0 -= 1, C0k1 += 1
        W[k0*2+0][col] = -1
        W[k1*2+0][col] = 1
        col += 1
        
        # Term 6: (A10-A00)(B0k0+B0k1)
        U[1*3+0][col] = 1
        U[0*3+0][col] = -1
        V[0*10+k0][col] = 1
        V[0*10+k1][col] = 1
        # C1k1 += 1
        W[k1*2+1][col] = 1
        col += 1
        
        # Term 7: (A01-A11)(B1k0+B1k1)
        U[0*3+1][col] = 1
        U[1*3+1][col] = -1
        V[1*10+k0][col] = 1
        V[1*10+k1][col] = 1
        # C0k0 += 1, C1k1 += 1
        W[k0*2+0][col] = 1
        W[k1*2+1][col] = 1
        col += 1

    # Part 2: A[0:2, 2] * B[2, :]
    # Schoolbook for 2x1 * 1x10
    for i in range(2):
        for k in range(10):
            U[i*3+2][col] = 1
            V[2*10+k][col] = 1
            W[k*2+i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
