"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # WFTA approach for 2x3x10 matrix multiplication
    # Use transform matrices to diagonalize the tensor
    # FA for n=2, FB for m=3
    # This creates 6 independent 1x1x10 problems
    
    # FA = [[1, 1], [1, -1]] (2-point transform)
    # FB = [[1, 1, 1], [1, 0, -1], [1, -1, 0]] (3-point transform)
    
    # The transformed tensor has rank = sum of ranks of independent problems
    # Each 1x1x10 problem has rank 10
    # Total: 6 * 10 = 60, but we can do better
    
    # Actually, let's use a different approach:
    # Decompose into:
    # 1. j=0: 2x1x10 -> rank 20
    # 2. j=1,2: 2x2x10 -> use Strassen on 2x2, split into 5 blocks of 2x2
    #    Each 2x2x2 block uses 7 multiplications (Strassen)
    #    Total: 5 * 7 = 35
    # Grand total: 20 + 35 = 55
    
    # To improve to 40, we need a better decomposition
    
    # Let's try: use WFTA properly
    # The key insight: for 2x3x10, we can use the following:
    # - Transform A using a 2x2 matrix
    # - Transform B using a 3x3 matrix
    # - This gives 6 independent 1x1x10 convolutions
    # - Each convolution can be done in 10 multiplications
    # - Total: 60
    
    # But we can reduce this by noting that some convolutions are simpler
    
    # For now, let's implement a local search to find a lower-rank decomposition
    
    # Start with the current best (55) and try to improve
    
    # Actually, let's just return the current solution for now
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part A: j=0 (first column of A, first row of B)
    # A[i,0] * B[0,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # A2 is 2x2, B2 is 2x10
    # Split B2 into 5 blocks of 2x2
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2
        
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1
        
        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
