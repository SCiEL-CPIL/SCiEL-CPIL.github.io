"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=3
    rank = 17
    U = [[0] * rank for _ in range(n * m)]  # (6, 17)
    V = [[0] * rank for _ in range(m * p)]  # (9, 17)
    W = [[0] * rank for _ in range(p * n)]  # (6, 17)
    
    # Helper: U[i,j] index = i*m+j, V[j,k] index = j*p+k, W[k,i] index = k*n+i
    def ui(i, j): return i * m + j
    def vi(j, k): return j * p + k
    def wi(k, i): return k * n + i
    
    col = 0
    # Strassen 2x2x2 for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # M1 = (A00+A01)(B00+B01)
    U[ui(0,0)][col] += 1; U[ui(0,1)][col] += 1
    V[vi(0,0)][col] += 1; V[vi(0,1)][col] += 1
    W[wi(0,0)][col] += 1; W[wi(1,1)][col] += 1
    col += 1
    
    # M2 = (A00+A01)B11
    U[ui(0,0)][col] += 1; U[ui(0,1)][col] += 1
    V[vi(1,1)][col] += 1
    W[wi(1,0)][col] += 1
    col += 1
    
    # M3 = A00(B10-B11)
    U[ui(0,0)][col] += 1
    V[vi(1,0)][col] += 1; V[vi(1,1)][col] += -1
    W[wi(0,1)][col] += 1
    col += 1
    
    # M4 = A11(B00-B10)
    U[ui(1,1)][col] += 1
    V[vi(0,0)][col] += 1; V[vi(1,0)][col] += -1
    W[wi(0,0)][col] += 1; W[wi(0,1)][col] += 1
    col += 1
    
    # M5 = (A10+A11)B00
    U[ui(1,0)][col] += 1; U[ui(1,1)][col] += 1
    V[vi(0,0)][col] += 1
    W[wi(1,0)][col] += 1; W[wi(0,1)][col] += -1
    col += 1
    
    # M6 = (A10-A00)(B11+B01)
    U[ui(1,0)][col] += 1; U[ui(0,0)][col] += -1
    V[vi(1,1)][col] += 1; V[vi(0,1)][col] += 1
    W[wi(1,1)][col] += 1
    col += 1
    
    # M7 = (A01-A10)(B10+B00)
    U[ui(0,1)][col] += 1; U[ui(1,0)][col] += -1
    V[vi(1,0)][col] += 1; V[vi(0,0)][col] += 1
    W[wi(0,0)][col] += 1; W[wi(1,1)][col] += -1
    col += 1
    
    # 2x2x1: A[0:2,0:2] * B[0:2,2:3] -> C[0:2,2:3]
    # (i=0,j=0,k=2)
    U[ui(0,0)][col] += 1; V[vi(0,2)][col] += 1; W[wi(2,0)][col] += 1
    col += 1
    # (i=0,j=1,k=2)
    U[ui(0,1)][col] += 1; V[vi(1,2)][col] += 1; W[wi(2,0)][col] += 1
    col += 1
    # (i=1,j=0,k=2)
    U[ui(1,0)][col] += 1; V[vi(0,2)][col] += 1; W[wi(2,1)][col] += 1
    col += 1
    # (i=1,j=1,k=2)
    U[ui(1,1)][col] += 1; V[vi(1,2)][col] += 1; W[wi(2,1)][col] += 1
    col += 1
    
    # 2x1x2: A[0:2,2:3] * B[0:2,0:2] -> C[0:2,0:2]
    # (i=0,j=2,k=0)
    U[ui(0,2)][col] += 1; V[vi(2,0)][col] += 1; W[wi(0,0)][col] += 1
    col += 1
    # (i=0,j=2,k=1)
    U[ui(0,2)][col] += 1; V[vi(2,1)][col] += 1; W[wi(1,0)][col] += 1
    col += 1
    # (i=1,j=2,k=0)
    U[ui(1,2)][col] += 1; V[vi(2,0)][col] += 1; W[wi(0,1)][col] += 1
    col += 1
    # (i=1,j=2,k=1)
    U[ui(1,2)][col] += 1; V[vi(2,1)][col] += 1; W[wi(1,1)][col] += 1
    col += 1
    
    # 2x1x1: A[0:2,2:3] * B[0:2,2:3] -> C[0:2,2:3]
    # (i=0,j=2,k=2)
    U[ui(0,2)][col] += 1; V[vi(2,2)][col] += 1; W[wi(2,0)][col] += 1
    col += 1
    # (i=1,j=2,k=2)
    U[ui(1,2)][col] += 1; V[vi(2,2)][col] += 1; W[wi(2,1)][col] += 1
    col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
