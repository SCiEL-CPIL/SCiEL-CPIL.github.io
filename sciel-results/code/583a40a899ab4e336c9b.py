"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    n, m, p = 2, 3, 10
    
    # Rank 55 decomposition:
    # - Strassen on 2x2 blocks (j=0,1) for 5 pairs of output columns: 5*7 = 35
    # - Schoolbook for j=2: 2*10 = 20
    # Total rank = 55
    
    rank = 55
    U_mat = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V_mat = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W_mat = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    idx = 0
    
    # Strassen blocks: j=0,1 over five 2-column pairs of B
    for l in range(5):
        if time.time() > deadline:
            break
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices
        a00 = 0 * 3 + 0  # A[0,0]
        a01 = 0 * 3 + 1  # A[0,1]
        a10 = 1 * 3 + 0  # A[1,0]
        a11 = 1 * 3 + 1  # A[1,1]
        
        # B indices
        b0k0 = 0 * 10 + k0
        b0k1 = 0 * 10 + k1
        b1k0 = 1 * 10 + k0
        b1k1 = 1 * 10 + k1
        
        # W indices
        c0k0 = k0 * 2 + 0
        c0k1 = k1 * 2 + 0
        c1k0 = k0 * 2 + 1
        c1k1 = k1 * 2 + 1
        
        # M1 = (a00 + a11) * (b0k0 + b1k1)
        U_mat[a00][idx] = 1
        U_mat[a11][idx] = 1
        V_mat[b0k0][idx] = 1
        V_mat[b1k1][idx] = 1
        W_mat[c0k0][idx] = 1
        W_mat[c1k1][idx] = 1
        idx += 1
        
        # M2 = (a10 + a11) * b0k0
        U_mat[a10][idx] = 1
        U_mat[a11][idx] = 1
        V_mat[b0k0][idx] = 1
        W_mat[c1k0][idx] = 1
        W_mat[c1k1][idx] = -1
        idx += 1
        
        # M3 = a00 * (b0k1 - b1k1)
        U_mat[a00][idx] = 1
        V_mat[b0k1][idx] = 1
        V_mat[b1k1][idx] = -1
        W_mat[c0k1][idx] = 1
        W_mat[c1k1][idx] = 1
        idx += 1
        
        # M4 = a11 * (b1k0 - b0k0)
        U_mat[a11][idx] = 1
        V_mat[b1k0][idx] = 1
        V_mat[b0k0][idx] = -1
        W_mat[c0k0][idx] = 1
        W_mat[c1k0][idx] = 1
        idx += 1
        
        # M5 = (a00 + a01) * b1k1
        U_mat[a00][idx] = 1
        U_mat[a01][idx] = 1
        V_mat[b1k1][idx] = 1
        W_mat[c0k0][idx] = -1
        W_mat[c0k1][idx] = 1
        idx += 1
        
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U_mat[a10][idx] = 1
        U_mat[a00][idx] = -1
        V_mat[b0k0][idx] = 1
        V_mat[b0k1][idx] = 1
        W_mat[c1k1][idx] = 1
        idx += 1
        
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U_mat[a01][idx] = 1
        U_mat[a11][idx] = -1
        V_mat[b1k0][idx] = 1
        V_mat[b1k1][idx] = 1
        W_mat[c0k0][idx] = 1
        idx += 1
    
    # Schoolbook for j=2
    for i in range(2):
        for k in range(10):
            if time.time() > deadline:
                break
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            
            U_mat[u_idx][idx] = 1
            V_mat[v_idx][idx] = 1
            W_mat[w_idx][idx] = 1
            idx += 1
    
    assert idx == rank, f"Expected rank {rank}, got {idx}"
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
