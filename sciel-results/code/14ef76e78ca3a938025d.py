"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=8
    
    # Strassen 2x2x2 rank-7 decomposition
    # U_strassen: (4, 7), index i*2+j
    # V_strassen: (4, 7), index j*2+k
    # W_strassen: (4, 7), index k*2+i
    U_s = [
        [1, 0, 1, 1, 1, 0, 1],  # A[0,0]
        [0, 1, 1, 0, 1, 0, 0],  # A[0,1]
        [0, 1, 0, 0, 0, 1, 0],  # A[1,0]
        [1, 1, 0, 0, 0, 0, 1],  # A[1,1]
    ]
    V_s = [
        [1, 1, 0, 1, 0, 1, 0],  # B[0,0]
        [0, 1, 1, 0, 1, 0, 0],  # B[0,1]
        [0, 0, 1, 0, 0, 1, 1],  # B[1,0]
        [1, 0, 1, 0, 1, 0, 1],  # B[1,1]
    ]
    W_s = [
        [1, 0, 0, 0, 1, 0, 0],    # C[0,0]: k=0,i=0 -> idx 0
        [0, 0, 1, 0, 1, 0, 0],    # C[0,1]: k=1,i=0 -> idx 2
        [0, 1, 0, 0, 0, 1, 0],    # C[1,0]: k=0,i=1 -> idx 1
        [1, -1, 0, 0, 0, -1, 1],  # C[1,1]: k=1,i=1 -> idx 3
    ]
    
    # Verify Strassen decomposition
    # T[a,b,c] = sum_r U_s[a,r]*V_s[b,r]*W_s[c,r] should equal 1 if valid (i,j,k)
    # a = i*2+j, b = j*2+k, c = k*2+i
    for i in range(2):
        for j in range(2):
            for k in range(2):
                a_idx = i*2 + j
                b_idx = j*2 + k
                c_idx = k*2 + i
                total = sum(U_s[a_idx][r] * V_s[b_idx][r] * W_s[c_idx][r] for r in range(7))
                assert total == 1, f"Strassen verification failed at i={i},j={j},k={k}: got {total}"
    
    # Tile to 4x4 x 4x8
    # n=4, m=4, p=8
    # U shape: (16, 112), V shape: (32, 112), W shape: (32, 112)
    # 16 blocks: (a, b, c) where a in {0,1}, b in {0,1}, c in {0,1,2,3}
    # Each block has 7 Strassen terms
    rank = 16 * 7  # 112
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 112
    V = [[0] * rank for _ in range(m * p)]  # 32 x 112
    W = [[0] * rank for _ in range(p * n)]  # 32 x 112
    
    for a in range(2):
        for b in range(2):
            for c in range(4):
                blk = a * 8 + b * 4 + c
                for r in range(7):
                    global_r = blk * 7 + r
                    for i_prime in range(2):
                        for j_prime in range(2):
                            # A index: i = 2*a + i_prime, j = 2*b + j_prime
                            a_idx = i_prime * 2 + j_prime
                            U_idx = (2*a + i_prime) * 4 + (2*b + j_prime)
                            val = U_s[a_idx][r]
                            if val != 0:
                                U[U_idx][global_r] += val
                    
                    for j_prime in range(2):
                        for k_prime in range(2):
                            # B index: j = 2*b + j_prime, k = 2*c + k_prime
                            b_idx = j_prime * 2 + k_prime
                            V_idx = (2*b + j_prime) * 8 + (2*c + k_prime)
                            val = V_s[b_idx][r]
                            if val != 0:
                                V[V_idx][global_r] += val
                    
                    for i_prime in range(2):
                        for k_prime in range(2):
                            # C index: i = 2*a + i_prime, k = 2*c + k_prime
                            c_idx = k_prime * 2 + i_prime
                            W_idx = (2*c + k_prime) * 4 + (2*a + i_prime)
                            val = W_s[c_idx][r]
                            if val != 0:
                                W[W_idx][global_r] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
