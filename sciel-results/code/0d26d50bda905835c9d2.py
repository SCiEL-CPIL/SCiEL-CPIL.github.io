"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 10
    # For 2x3x1, rank 3 suffices. For 2x3x10, rank 3*10 = 30.
    # Construction: for each k, use 3 terms:
    #   Term (k,0): U[3*0+0]=1, U[3*1+0]=1, V[0]=1, W[k*2+0]=1, W[k*2+1]=1
    #   Term (k,1): U[3*0+1]=1, U[3*1+1]=1, V[1]=1, W[k*2+0]=1, W[k*2+1]=1
    #   Term (k,2): U[3*0+2]=1, U[3*1+2]=1, V[2]=1, W[k*2+0]=1, W[k*2+1]=1
    # This gives T[i,j,k] = sum_c U[3i+j,c]*V[j,c]*W[2k+i,c] = 1 for all i,j,k.
    
    rank = 3 * p  # 30
    U = [[0] * rank for _ in range(n * m)]  # 6 x 30
    V = [[0] * rank for _ in range(m * p)]  # 30 x 30
    W = [[0] * rank for _ in range(p * n)]  # 20 x 30
    
    col = 0
    for k in range(p):
        for j in range(m):
            # U[0*m+j, col] = 1, U[1*m+j, col] = 1
            U[0 * m + j][col] = 1
            U[1 * m + j][col] = 1
            # V[j*p + k, col] = 1
            V[j * p + k][col] = 1
            # W[k*n + 0, col] = 1, W[k*n + 1, col] = 1
            W[k * n + 0][col] = 1
            W[k * n + 1][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
