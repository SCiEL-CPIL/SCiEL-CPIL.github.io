"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 10
    # Rank 55 decomposition
    # Term 1: A[:2] @ B[:2, :] using Strassen on 5 blocks of 2x2 (rank 35)
    # Term 2: A[:, 2] @ B[2, :] using schoolbook (rank 20)
    
    rank = 55
    U_mat = [[0] * rank for _ in range(n * m)] # 6 x 55
    V_mat = [[0] * rank for _ in range(m * p)] # 30 x 55
    W_mat = [[0] * rank for _ in range(p * n)] # 20 x 55
    
    idx = 0
    
    # Term 1: Strassen blocks
    # A is 2x3, we use cols 0,1. Indices in U: i*3 + j
    # B is 3x10, we use rows 0,1. Indices in V: j*10 + k
    # C is 2x10, indices in W: k*2 + i
    
    for l in range(5):
        k0 = 2 * l
        k1 = 2 * l + 1
        
        # A indices
        a00 = 0 * 3 + 0 # 0
        a01 = 0 * 3 + 1 # 1
        a10 = 1 * 3 + 0 # 3
        a11 = 1 * 3 + 1 # 4
        
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
        U_mat[a00][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M2 = (a10 + a11) * b0k0
        U_mat[a10][idx] += 1
        U_mat[a11][idx] += 1
        V_mat[b0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        W_mat[c1k1][idx] -= 1
        idx += 1
        
        # M3 = a00 * (b0k1 - b1k1)
        U_mat[a00][idx] += 1
        V_mat[b0k1][idx] += 1
        V_mat[b1k1][idx] -= 1
        W_mat[c0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M4 = a11 * (b1k0 - b0k0)
        U_mat[a11][idx] += 1
        V_mat[b1k0][idx] += 1
        V_mat[b0k0][idx] -= 1
        W_mat[c0k0][idx] += 1
        W_mat[c1k0][idx] += 1
        idx += 1
        
        # M5 = (a00 + a01) * b1k1
        U_mat[a00][idx] += 1
        U_mat[a01][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] -= 1
        W_mat[c0k1][idx] += 1
        idx += 1
        
        # M6 = (a10 - a00) * (b0k0 + b0k1)
        U_mat[a10][idx] += 1
        U_mat[a00][idx] -= 1
        V_mat[b0k0][idx] += 1
        V_mat[b0k1][idx] += 1
        W_mat[c1k1][idx] += 1
        idx += 1
        
        # M7 = (a01 - a11) * (b1k0 + b1k1)
        U_mat[a01][idx] += 1
        U_mat[a11][idx] -= 1
        V_mat[b1k0][idx] += 1
        V_mat[b1k1][idx] += 1
        W_mat[c0k0][idx] += 1
        idx += 1
        
    # Term 2: Schoolbook for j=2
    # C[i,k] += A[i,2] * B[2,k]
    for i in range(2):
        for k in range(10):
            u_idx = i * 3 + 2
            v_idx = 2 * 10 + k
            w_idx = k * 2 + i
            
            U_mat[u_idx][idx] += 1
            V_mat[v_idx][idx] += 1
            W_mat[w_idx][idx] += 1
            idx += 1
            
    # The construction is deterministic and completes instantly.
    # idx should be exactly 55.
    assert idx == rank, f"Expected rank {rank}, got {idx}"
    
    return {'U': U_mat, 'V': V_mat, 'W': W_mat}
# EVOLVE-BLOCK-END
