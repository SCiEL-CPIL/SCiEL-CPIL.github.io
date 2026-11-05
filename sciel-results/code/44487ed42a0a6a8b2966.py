import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin before the 120s kill
    
    n, m, p = DIMENSIONS
    
    # We will construct a rank-55 decomposition using the strategy from the inspirations:
    # Split A (2x3) into A_left (2x2) and A_right (2x1).
    # Split B (3x10) into B_top (2x10) and B_bottom (1x10).
    # C = A_left @ B_top + A_right @ B_bottom
    #
    # Part 1: A_left (2x2) @ B_top (2x10)
    #   Split B_top into 5 blocks of size 2x2.
    #   Use Strassen for each 2x2x2 block (rank 7 each).
    #   Total rank for Part 1: 5 * 7 = 35.
    #
    # Part 2: A_right (2x1) @ B_bottom (1x10)
    #   This is a rank-20 tensor (schoolbook for 2x1x10).
    #   Total rank for Part 2: 20.
    #
    # Total Rank: 35 + 20 = 55.
    
    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55
    
    # Indexing conventions:
    # U row: i*m + j, where i in [0,1] (row of A), j in [0,1,2] (col of A)
    # V row: j*p + k, where j in [0,1,2] (row of B), k in [0..9] (col of B)
    # W row: k*n + i, where k in [0..9] (col of C), i in [0,1] (row of C)
    
    r = 0  # Current rank index
    
    # --- Part 1: A_left (2x2) @ B_top (2x10) ---
    # A_left entries:
    #   A[0,0] -> U index 0*3+0 = 0
    #   A[0,1] -> U index 0*3+1 = 1
    #   A[1,0] -> U index 1*3+0 = 3
    #   A[1,1] -> U index 1*3+1 = 4
    # B_top entries:
    #   B[0,k] -> V index 0*10+k = k
    #   B[1,k] -> V index 1*10+k = 10+k
    
    for t in range(5):
        k0 = 2 * t
        k1 = 2 * t + 1
        
        # Strassen 7 products for A_left * B_block(t)
        # B_block(t) is 2x2:
        #   [[B[0,k0], B[0,k1]],
        #    [B[1,k0], B[1,k1]]]
        # V indices:
        #   B[0,k0] -> V[k0]
        #   B[0,k1] -> V[k1]
        #   B[1,k0] -> V[10+k0]
        #   B[1,k1] -> V[10+k1]
        
        # M1 = (A00 + A11) * (B0_k0 + B1_k1)
        # U: A00(1), A11(1)
        # V: B0_k0(1), B1_k1(1)
        # W: C0_k0(1), C1_k1(1)
        U[0][r] += 1
        U[4][r] += 1
        V[k0][r] += 1
        V[10+k1][r] += 1
        W[k0*2+0][r] += 1
        W[k1*2+1][r] += 1
        r += 1
        
        # M2 = (A10 + A11) * B0_k0
        # U: A10(1), A11(1)
        # V: B0_k0(1)
        # W: C1_k0(1), C1_k1(-1)
        U[3][r] += 1
        U[4][r] += 1
        V[k0][r] += 1
        W[k0*2+1][r] += 1
        W[k1*2+1][r] -= 1
        r += 1
        
        # M3 = A00 * (B0_k1 - B1_k1)
        # U: A00(1)
        # V: B0_k1(1), B1_k1(-1)
        # W: C0_k1(1), C1_k1(1)
        U[0][r] += 1
        V[k1][r] += 1
        V[10+k1][r] -= 1
        W[k1*2+0][r] += 1
        W[k1*2+1][r] += 1
        r += 1
        
        # M4 = A11 * (B1_k0 - B0_k0)
        # U: A11(1)
        # V: B1_k0(1), B0_k0(-1)
        # W: C0_k0(1), C1_k0(1)
        U[4][r] += 1
        V[10+k0][r] += 1
        V[k0][r] -= 1
        W[k0*2+0][r] += 1
        W[k0*2+1][r] += 1
        r += 1
        
        # M5 = (A00 + A01) * B1_k1
        # U: A00(1), A01(1)
        # V: B1_k1(1)
        # W: C0_k0(-1), C0_k1(1)
        U[0][r] += 1
        U[1][r] += 1
        V[10+k1][r] += 1
        W[k0*2+0][r] -= 1
        W[k1*2+0][r] += 1
        r += 1
        
        # M6 = (A10 - A00) * (B0_k0 + B0_k1)
        # U: A10(1), A00(-1)
        # V: B0_k0(1), B0_k1(1)
        # W: C1_k1(1)
        U[3][r] += 1
        U[0][r] -= 1
        V[k0][r] += 1
        V[k1][r] += 1
        W[k1*2+1][r] += 1
        r += 1
        
        # M7 = (A01 - A11) * (B1_k0 + B1_k1)
        # U: A01(1), A11(-1)
        # V: B1_k0(1), B1_k1(1)
        # W: C0_k0(1)
        U[1][r] += 1
        U[4][r] -= 1
        V[10+k0][r] += 1
        V[10+k1][r] += 1
        W[k0*2+0][r] += 1
        r += 1

    # --- Part 2: A_right (2x1) @ B_bottom (1x10) ---
    # A_right entries:
    #   A[0,2] -> U index 0*3+2 = 2
    #   A[1,2] -> U index 1*3+2 = 5
    # B_bottom entries:
    #   B[2,k] -> V index 2*10+k = 20+k
    
    for k in range(10):
        # Term for C[0,k] = A[0,2] * B[2,k]
        U[2][r] += 1
        V[20+k][r] += 1
        W[k*2+0][r] += 1
        r += 1
        
        # Term for C[1,k] = A[1,2] * B[2,k]
        U[5][r] += 1
        V[20+k][r] += 1
        W[k*2+1][r] += 1
        r += 1

    # Verify r == rank
    if r != rank:
        # Fallback to trivial if something went wrong (should not happen)
        return _trivial_decomposition(n, m, p)
    
    # Quick validation
    if time.time() < deadline:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check a few random terms or all terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if not np.isclose(lhs, 1.0, atol=1e-9):
                        # Fallback to trivial rank-60
                        return _trivial_decomposition(n, m, p)

    return {"U": U, "V": V, "W": W}

def _trivial_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][r] = 1
                V[j*p+k][r] = 1
                W[k*n+i][r] = 1
                r += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
