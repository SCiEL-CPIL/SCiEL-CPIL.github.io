"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=11
    rank = 39  # 5 Strassen blocks (rank 7 each) + 1 schoolbook block (rank 4)
    
    U = [[0] * rank for _ in range(n * m)]      # shape (4, rank)
    V = [[0] * rank for _ in range(m * p)]      # shape (22, rank)
    W = [[0] * rank for _ in range(p * n)]      # shape (22, rank)
    
    col = 0
    
    # Process 5 pairs of columns (0..9)
    for k in range(5):
        if time.time() > deadline:
            break
        # b11 = B[0, 2k] -> V idx 2k
        # b12 = B[0, 2k+1] -> V idx 2k+1
        # b21 = B[1, 2k] -> V idx 11+2k
        # b22 = B[1, 2k+1] -> V idx 11+2k+1
        
        # W indices: C[i, k] -> W[k*n + i]
        # C[0, 2k] -> W[2k*2 + 0] = W[4k]
        # C[1, 2k] -> W[2k*2 + 1] = W[4k+1]
        # C[0, 2k+1] -> W[(2k+1)*2 + 0] = W[4k+2]
        # C[1, 2k+1] -> W[(2k+1)*2 + 1] = W[4k+3]
        
        # M1: (A00 + A11) * (B0,2k + B1,2k+1)
        # Contributes to: C[0,2k] and C[1,2k+1]
        U[0][col] += 1
        U[3][col] += 1
        V[2*k][col] += 1
        V[11 + 2*k + 1][col] += 1
        W[4*k][col] += 1      # C[0, 2k]
        W[4*k + 3][col] += 1  # C[1, 2k+1]
        col += 1
        
        # M2: (A10 + A11) * B0,2k
        # Contributes to: C[1,2k] and -C[1,2k+1]
        U[2][col] += 1
        U[3][col] += 1
        V[2*k][col] += 1
        W[4*k + 1][col] += 1    # C[1, 2k]
        W[4*k + 3][col] += -1   # C[1, 2k+1]
        col += 1
        
        # M3: A00 * (B0,2k+1 - B1,2k+1)
        # Contributes to: C[0,2k+1] and C[1,2k+1]
        U[0][col] += 1
        V[2*k + 1][col] += 1
        V[11 + 2*k + 1][col] += -1
        W[4*k + 2][col] += 1    # C[0, 2k+1]
        W[4*k + 3][col] += 1    # C[1, 2k+1]
        col += 1
        
        # M4: A11 * (B1,2k - B0,2k)
        # Contributes to: C[0,2k] and C[1,2k]
        U[3][col] += 1
        V[11 + 2*k][col] += 1
        V[2*k][col] += -1
        W[4*k][col] += 1      # C[0, 2k]
        W[4*k + 1][col] += 1  # C[1, 2k]
        col += 1
        
        # M5: (A00 + A01) * B1,2k+1
        # Contributes to: -C[0,2k] and C[0,2k+1]
        U[0][col] += 1
        U[1][col] += 1
        V[11 + 2*k + 1][col] += 1
        W[4*k][col] += -1      # C[0, 2k]
        W[4*k + 2][col] += 1   # C[0, 2k+1]
        col += 1
        
        # M6: (A10 - A00) * (B0,2k + B0,2k+1)
        # Contributes to: C[1,2k+1]
        U[2][col] += 1
        U[0][col] += -1
        V[2*k][col] += 1
        V[2*k + 1][col] += 1
        W[4*k + 3][col] += 1   # C[1, 2k+1]
        col += 1
        
        # M7: (A01 - A11) * (B1,2k + B1,2k+1)
        # Contributes to: C[0,2k]
        U[1][col] += 1
        U[3][col] += -1
        V[11 + 2*k][col] += 1
        V[11 + 2*k + 1][col] += 1
        W[4*k][col] += 1       # C[0, 2k]
        col += 1
    
    # Process last column (index 10) - schoolbook
    if col < rank and time.time() < deadline:
        # M8: A00 * B0,10
        U[0][col] += 1
        V[10][col] += 1
        W[20][col] += 1  # C[0, 10]
        col += 1
        
        # M9: A01 * B1,10
        U[1][col] += 1
        V[21][col] += 1
        W[20][col] += 1  # C[0, 10]
        col += 1
        
        # M10: A10 * B0,10
        U[2][col] += 1
        V[10][col] += 1
        W[21][col] += 1  # C[1, 10]
        col += 1
        
        # M11: A11 * B1,10
        U[3][col] += 1
        V[21][col] += 1
        W[21][col] += 1  # C[1, 10]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
