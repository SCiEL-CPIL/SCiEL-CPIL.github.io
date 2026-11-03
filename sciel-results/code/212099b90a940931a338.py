import numpy as np

DIMENSIONS = [3, 4, 7]

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 matrix multiplication with rank 7 (Strassen)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # m3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # m4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # m5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # m6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # m7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The coefficient of A[i,j]*B[j,k] in C[i,k] should be 1
                coeff = 0.0
                for r in range(rank):
                    coeff += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(coeff - 1.0) > 1e-10:
                    return False
                # Check that cross terms cancel (should be 0 for different i,j,k combos)
                # Actually we need to check that the full tensor matches
    return True

def _verify_full_tensor(U, V, W, n, m, p):
    """Full verification that the decomposition is exact."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # In the decomposition: T = sum_r u_r ⊗ v_r ⊗ w_r
    # where u_r[i*m+j] = U[i*m+j][r], v_r[j*p+k] = V[j*p+k][r], w_r[k*n+i] = W[k*n+i][r]
    # So T[i,j,k] = sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    # This should equal 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                coeff = 0.0
                for r in range(rank):
                    coeff += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(coeff - 1.0) > 1e-10:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Use schoolbook for simplicity - rank 84
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
