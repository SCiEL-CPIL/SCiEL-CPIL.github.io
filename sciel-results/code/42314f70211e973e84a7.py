"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    # Target rank: 55
    # 5 Strassen blocks (rank 7 each) for j in {0,1}, k in pairs
    # 20 Schoolbook terms for j=2
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen decomposition for M_{2,2,2}
    # Indices: i in {0,1}, j in {0,1}, k in {0,1}
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b01-b11)
    # m4 = a11(b10-b00)
    # m5 = (a00+a10)b11
    # m6 = (a01-a00)(b00+b10)
    # m7 = (a11-a10)(b01+b11)
    
    # Vectors for U (indices i, j)
    # Vectors for V (indices j, k)
    # Vectors for W (indices k, i)
    
    # We will generate the 7 terms for each block
    # Block b uses j in {0,1} and k in {2b, 2b+1}
    
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Term 1: (a00+a11)(b00+b11)
        # U: a00 (i=0,j=0) + a11 (i=1,j=1) -> U[0]=1, U[3]=1
        # V: b00 (j=0,k0) + b11 (j=1,k1) -> V[k0]=1, V[m+k1]=1 (m=3)
        # W: c00 (k0,i=0) + c11 (k1,i=1) -> W[k0*n+0]=1, W[k1*n+1]=1
        U[0][col] += 1; U[3][col] += 1
        V[k0][col] += 1; V[m + k1][col] += 1
        W[k0 * n + 0][col] += 1; W[k1 * n + 1][col] += 1
        col += 1
        
        # Term 2: (a10+a11)b00
        # U: a10 (i=1,j=0) + a11 (i=1,j=1) -> U[2]=1, U[3]=1
        # V: b00 (j=0,k0) -> V[k0]=1
        # W: c10 (k0,i=1) + c11 (k1,i=1) -> W[k0*n+1]=1, W[k1*n+1]=1
        U[2][col] += 1; U[3][col] += 1
        V[k0][col] += 1
        W[k0 * n + 1][col] += 1; W[k1 * n + 1][col] += 1
        col += 1
        
        # Term 3: a00(b01-b11)
        # U: a00 (i=0,j=0) -> U[0]=1
        # V: b01 (j=0,k1) - b11 (j=1,k1) -> V[k1]=1, V[m+k1]=-1
        # W: c01 (k1,i=0) - c11 (k1,i=1) -> W[k1*n+0]=1, W[k1*n+1]=-1
        U[0][col] += 1
        V[k1][col] += 1; V[m + k1][col] -= 1
        W[k1 * n + 0][col] += 1; W[k1 * n + 1][col] -= 1
        col += 1
        
        # Term 4: a11(b10-b00)
        # U: a11 (i=1,j=1) -> U[3]=1
        # V: b10 (j=1,k0) - b00 (j=0,k0) -> V[m+k0]=1, V[k0]=-1
        # W: c10 (k0,i=1) - c00 (k0,i=0) -> W[k0*n+1]=1, W[k0*n+0]=-1
        U[3][col] += 1
        V[m + k0][col] += 1; V[k0][col] -= 1
        W[k0 * n + 1][col] += 1; W[k0 * n + 0][col] -= 1
        col += 1
        
        # Term 5: (a00+a10)b11
        # U: a00 (i=0,j=0) + a10 (i=1,j=0) -> U[0]=1, U[2]=1
        # V: b11 (j=1,k1) -> V[m+k1]=1
        # W: c01 (k1,i=0) + c11 (k1,i=1) -> W[k1*n+0]=1, W[k1*n+1]=1
        U[0][col] += 1; U[2][col] += 1
        V[m + k1][col] += 1
        W[k1 * n + 0][col] += 1; W[k1 * n + 1][col] += 1
        col += 1
        
        # Term 6: (a01-a00)(b00+b10)
        # U: a01 (i=0,j=1) - a00 (i=0,j=0) -> U[1]=1, U[0]=-1
        # V: b00 (j=0,k0) + b10 (j=1,k0) -> V[k0]=1, V[m+k0]=1
        # W: c00 (k0,i=0) + c10 (k0,i=1) -> W[k0*n+0]=1, W[k0*n+1]=1
        U[1][col] += 1; U[0][col] -= 1
        V[k0][col] += 1; V[m + k0][col] += 1
        W[k0 * n + 0][col] += 1; W[k0 * n + 1][col] += 1
        col += 1
        
        # Term 7: (a11-a10)(b01+b11)
        # U: a11 (i=1,j=1) - a10 (i=1,j=0) -> U[3]=1, U[2]=-1
        # V: b01 (j=0,k1) + b11 (j=1,k1) -> V[k1]=1, V[m+k1]=1
        # W: c01 (k1,i=0) + c11 (k1,i=1) -> W[k1*n+0]=1, W[k1*n+1]=1
        U[3][col] += 1; U[2][col] -= 1
        V[k1][col] += 1; V[m + k1][col] += 1
        W[k1 * n + 0][col] += 1; W[k1 * n + 1][col] += 1
        col += 1

    # Schoolbook for j=2
    # i in {0,1}, k in {0..9}
    for i in range(n):
        for k in range(p):
            # U index: i*m + 2
            # V index: 2*p + k
            # W index: k*n + i
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
