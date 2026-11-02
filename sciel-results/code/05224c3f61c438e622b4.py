"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split 11 columns into 5 pairs + 1 single
    pairs = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    single = 10
    
    rank = len(pairs) * 7 + 4  # 5*7 + 4 = 39
    U = [[0] * rank for _ in range(n * m)]  # 4 x 39
    V = [[0] * rank for _ in range(m * p)]  # 22 x 39
    W = [[0] * rank for _ in range(p * n)]  # 22 x 39
    
    col = 0
    for (k1, k2) in pairs:
        # Strassen for 2x2 x 2x2 (columns k1, k2)
        # a=A[0,0]=U[0], b=A[0,1]=U[1], c=A[1,0]=U[2], d=A[1,1]=U[3]
        # e=B[0,k1]=V[k1], f=B[0,k2]=V[k2], g=B[1,k1]=V[11+k1], h=B[1,k2]=V[11+k2]
        # C[0,k1]=W[2*k1], C[0,k2]=W[2*k2], C[1,k1]=W[2*k1+1], C[1,k2]=W[2*k2+1]
        
        # M1 = (a+d)(e+h)
        U[0][col] += 1
        U[3][col] += 1
        V[k1][col] += 1
        V[11+k2][col] += 1
        W[2*k1][col] += 1    # C[0,k1] += M1
        W[2*k2][col] += 1    # C[0,k2] += M1
        W[2*k1+1][col] += 1  # C[1,k1] += M1
        W[2*k2+1][col] -= 1  # C[1,k2] -= M1
        col += 1
        
        # M2 = (c+d)e
        U[2][col] += 1
        U[3][col] += 1
        V[k1][col] += 1
        W[2*k1+1][col] += 1  # C[1,k1] += M2
        W[2*k2+1][col] -= 1  # C[1,k2] -= M2
        col += 1
        
        # M3 = a(f-h)
        U[0][col] += 1
        V[k2][col] += 1
        V[11+k2][col] -= 1
        W[2*k2][col] += 1    # C[0,k2] += M3
        W[2*k2+1][col] += 1  # C[1,k2] += M3
        col += 1
        
        # M4 = d(g-e)
        U[3][col] += 1
        V[11+k1][col] += 1
        V[k1][col] -= 1
        W[2*k1][col] += 1    # C[0,k1] += M4
        W[2*k1+1][col] += 1  # C[1,k1] += M4
        col += 1
        
        # M5 = (a+b)h
        U[0][col] += 1
        U[1][col] += 1
        V[11+k2][col] += 1
        W[2*k1][col] -= 1    # C[0,k1] -= M5
        W[2*k2][col] += 1    # C[0,k2] += M5
        col += 1
        
        # M6 = (c-a)(e+f)
        U[2][col] += 1
        U[0][col] -= 1
        V[k1][col] += 1
        V[k2][col] += 1
        W[2*k2+1][col] += 1  # C[1,k2] += M6
        col += 1
        
        # M7 = (b-d)(g+h)
        U[1][col] += 1
        U[3][col] -= 1
        V[11+k1][col] += 1
        V[11+k2][col] += 1
        W[2*k1][col] += 1    # C[0,k1] += M7
        col += 1
    
    # Single column 10: schoolbook 2x2 x 2x1
    # C[0,10] = A[0,0]*B[0,10] + A[0,1]*B[1,10]
    # C[1,10] = A[1,0]*B[0,10] + A[1,1]*B[1,10]
    
    # Term 1: A[0,0]*B[0,10]
    U[0][col] += 1
    V[10][col] += 1
    W[2*10][col] += 1  # C[0,10]
    col += 1
    
    # Term 2: A[0,1]*B[1,10]
    U[1][col] += 1
    V[11+10][col] += 1
    W[2*10][col] += 1  # C[0,10]
    col += 1
    
    # Term 3: A[1,0]*B[0,10]
    U[2][col] += 1
    V[10][col] += 1
    W[2*10+1][col] += 1  # C[1,10]
    col += 1
    
    # Term 4: A[1,1]*B[1,10]
    U[3][col] += 1
    V[11+10][col] += 1
    W[2*10+1][col] += 1  # C[1,10]
    col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
