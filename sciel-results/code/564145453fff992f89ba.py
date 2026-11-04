import time

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s margin out of 120s/150s
    
    n, m, p = DIMENSIONS
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen part for A1 B1
    for j in range(5):
        if time.time() > deadline:
            break
            
        idx_n11 = 2 * j
        idx_n12 = 2 * j + 1
        idx_n21 = 10 + 2 * j
        idx_n22 = 10 + 2 * j + 1
        
        idx_m11 = 0
        idx_m12 = 1
        idx_m21 = 3
        idx_m22 = 4
        
        idx_c11 = 4 * j
        idx_c12 = 4 * j + 2
        idx_c21 = 4 * j + 1
        idx_c22 = 4 * j + 3
        
        # P1
        U[idx_m11][col] += 1
        U[idx_m22][col] += 1
        V[idx_n11][col] += 1
        V[idx_n22][col] += 1
        W[idx_c11][col] += 1
        W[idx_c22][col] += 1
        col += 1
        
        # P2
        U[idx_m11][col] += 1
        U[idx_m12][col] += 1
        V[idx_n22][col] += 1
        W[idx_c21][col] += 1
        W[idx_c22][col] -= 1
        col += 1
        
        # P3
        U[idx_m11][col] += 1
        V[idx_n12][col] += 1
        V[idx_n22][col] -= 1
        W[idx_c12][col] += 1
        W[idx_c22][col] += 1
        col += 1
        
        # P4
        U[idx_m22][col] += 1
        V[idx_n11][col] += 1
        V[idx_n12][col] -= 1
        W[idx_c11][col] += 1
        W[idx_c21][col] += 1
        col += 1
        
        # P5
        U[idx_m11][col] += 1
        U[idx_m21][col] += 1
        V[idx_n11][col] += 1
        W[idx_c11][col] -= 1
        W[idx_c12][col] += 1
        col += 1
        
        # P6
        U[idx_m12][col] += 1
        U[idx_m11][col] -= 1
        V[idx_n21][col] += 1
        V[idx_n22][col] += 1
        W[idx_c22][col] += 1
        col += 1
        
        # P7
        U[idx_m11][col] += 1
        U[idx_m21][col] -= 1
        V[idx_n11][col] += 1
        V[idx_n12][col] += 1
        W[idx_c11][col] += 1
        col += 1

    # Schoolbook part for A2 B2
    idx_a1 = 2
    idx_a2 = 5
    
    for k in range(10):
        if time.time() > deadline:
            break
            
        idx_b = 20 + k
        idx_c1 = 2 * k
        idx_c2 = 2 * k + 1
        
        U[idx_a1][col] += 1
        V[idx_b][col] += 1
        W[idx_c1][col] += 1
        col += 1
        
        U[idx_a2][col] += 1
        V[idx_b][col] += 1
        W[idx_c2][col] += 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
