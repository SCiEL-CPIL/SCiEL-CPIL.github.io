"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 6
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add(u_i, u_j, v_j, v_k, w_k, w_i, u_val=1, v_val=1, w_val=1):
        nonlocal col
        U[u_i * m + u_j][col] = u_val
        V[v_j * p + v_k][col] = v_val
        W[w_k * n + w_i][col] = w_val
        col += 1
    
    # M1 = (A11+A22)(B11+B22)
    # X1[i][j] = A[i][j] + A[i+2][j+2]
    # Y1[j][k] = B[j][k] + B[j+2][k]
    # C11[i][k] += M1[i][k] = sum_j X1[i][j]*Y1[j][k]
    
    # For i=0,j=0,k=0: X1[0][0]=A[0][0]+A[2][2], Y1[0][0]=B[0][0]+B[2][0]
    # This gives 4 terms: (A00,B00), (A00,B20), (A22,B00), (A22,B20)
    # But we can only have 1 term per (i,j,k) in schoolbook!
    
    # I think the issue is that Strassen doesn't directly translate to a
    # tensor decomposition in the way I'm thinking.
    
    # Let me just use schoolbook rank 96.
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
