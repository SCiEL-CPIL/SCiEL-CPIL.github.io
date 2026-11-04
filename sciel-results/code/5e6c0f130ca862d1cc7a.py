# EVOLVE-BLOCK-START
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

_DEADLINE = time.time() + 140

def _make_laderman_3x3():
    """
    Returns U, V, W for Laderman 3x3 decomposition (rank 23).
    Returns None if construction fails verification.
    """
    s = 3
    rank = 23
    U = np.zeros((s*s, rank))
    V = np.zeros((s*s, rank))
    W = np.zeros((s*s, rank))
    
    def au(i, j, c, v): U[i*s+j, c] += v
    def av(j, k, c, v): V[j*s+k, c] += v
    def aw(k, i, c, v): W[k*s+i, c] += v
    
    # Direct terms
    au(0,0,0,1); av(0,0,0,1); aw(0,0,0,1)
    au(0,1,1,1); av(1,0,1,1); aw(0,0,1,1)
    au(0,2,2,1); av(2,0,2,1); aw(0,0,2,1)
    au(1,0,3,1); av(0,1,3,1); aw(1,0,3,1)
    au(1,1,4,1); av(1,1,4,1); aw(1,1,4,1)
    au(1,2,5,1); av(2,1,5,1); aw(1,1,5,1)
    au(2,0,6,1); av(0,2,6,1); aw(2,0,6,1)
    au(2,1,7,1); av(1,2,7,1); aw(2,1,7,1)
    au(2,2,8,1); av(2,2,8,1); aw(2,2,8,1)
    
    # Diag pairs
    # (A00+A11)(B00+B11)
    au(0,0,9,1); au(1,1,9,1)
    av(0,0,9,1); av(1,1,9,1)
    aw(1,0,9,1) # C[0,1] += A00B11
    aw(0,1,9,1) # C[1,0] += A11B00
    
    # (A00+A22)(B00+B22)
    au(0,0,10,1); au(2,2,10,1)
    av(0,0,10,1); av(2,2,10,1)
    aw(2,0,10,1) # C[0,2] += A00B22
    aw(0,2,10,1) # C[2,0] += A22B00
    
    # (A11+A22)(B11+B22)
    au(1,1,11,1); au(2,2,11,1)
    av(1,1,11,1); av(2,2,11,1)
    aw(2,1,11,1) # C[1,2] += A11B22
    aw(1,2,11,1) # C[2,1] += A22B11
    
    # Row/Col sums
    # (A00+A01+A02)(B10+B11+B12)
    au(0,0,12,1); au(0,1,12,1); au(0,2,12,1)
    av(1,0,12,1); av(1,1,12,1); av(1,2,12,1)
    # This term is Row0(A) * Row1(B).
    # It contributes to C[0][k]? No.
    # This set of coefficients might be for a different convention or I am missing the W coefficients.
    # The W coefficients for these sum terms are crucial.
    
    # Given the high risk of incorrect Laderman coefficients, I will NOT include the unverified Laderman.
    # I will stick to the working Strassen(2x2) x Schoolbook(3x3) which is Rank 189.
    # The current program already achieves this.
    # To improve, I need a better inner decomposition.
    # Since I cannot guarantee Laderman, I will return None.
    return None

def construct():
    start_time = time.time()
    n, m, p = 6, 6, 6
    
    def _verify_decomposition(U, V, W, n, m, p):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                u_row = U_np[i * m + j, :]
                for k in range(p):
                    v_row = V_np[j * p + k, :]
                    w_row = W_np[k * n + i, :]
                    s = np.dot(u_row, v_row * w_row)
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err < 1e-10

    def _hybrid_strassen_schoolbook():
        block_size = 3
        dim = 6
        rank_inner = block_size ** 3  # 27
        rank_outer = 7
        total_rank = rank_inner * rank_outer
        
        U = np.zeros((dim*dim, total_rank))
        V = np.zeros((dim*dim, total_rank))
        W = np.zeros((dim*dim, total_rank))
        
        col = 0
        steps = [
            ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
            ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
            ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
            ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
            ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
            ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
            ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
        ]
        
        for A_coeffs, B_coeffs, C_coeffs in steps:
            for u in range(block_size):
                for v in range(block_size):
                    for w in range(block_size):
                        for (I, J), ca in A_coeffs.items():
                            i_g = I * block_size + u
                            j_g = J * block_size + v
                            U[i_g*dim+j_g, col] += ca
                        for (J, K), cb in B_coeffs.items():
                            j_g = J * block_size + v
                            k_g = K * block_size + w
                            V[j_g*dim+k_g, col] += cb
                        for (I, K), cc in C_coeffs.items():
                            i_g = I * block_size + u
                            k_g = K * block_size + w
                            W[k_g*dim+i_g, col] += cc
                        col += 1
        
        return U, V, W, total_rank

    U6, V6, W6, rank6 = _hybrid_strassen_schoolbook()
    
    if _verify_decomposition(U6, V6, W6, n, m, p):
        return {"U": U6.tolist(), "V": V6.tolist(), "W": W6.tolist()}
    
    # Fallback
    rank = n*m*p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
