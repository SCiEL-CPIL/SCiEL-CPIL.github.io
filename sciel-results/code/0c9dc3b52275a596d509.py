import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill

    # Rank 55 decomposition for 2x3 x 3x10
    # Strategy:
    # Split A (2x3) into A1 (2x2, cols 0,1) and A2 (2x1, col 2)
    # Split B (3x10) into B1 (2x10, rows 0,1) and B2 (1x10, row 2)
    # C = A1 B1 + A2 B2
    # A1 B1: 2x2 * 2x10. Split B1 into 5 blocks of 2x2.
    # Each 2x2 * 2x2 block uses Strassen (rank 7).
    # 5 * 7 = 35 terms.
    # A2 B2: 2x1 * 1x10.
    # C[i,k] += A[i,2] * B[2,k].
    # 2 * 10 = 20 terms.
    # Total rank = 35 + 20 = 55.

    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0

    # Part 1: A1 B1 (Strassen on 2x2 blocks)
    # A1 entries: A[0,0], A[0,1], A[1,0], A[1,1]
    # U indices: i*m + j
    # A[0,0] -> U[0]
    # A[0,1] -> U[1]
    # A[1,0] -> U[3]
    # A[1,1] -> U[4]
    
    # B1 entries: B[0,k], B[1,k]
    # V indices: j*p + k
    # B[0,k] -> V[k]
    # B[1,k] -> V[10+k]
    
    # W indices: k*n + i
    # C[0,k] -> W[2k]
    # C[1,k] -> W[2k+1]

    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1
        
        # Indices for this block
        u_a00 = 0 * m + 0 # 0
        u_a01 = 0 * m + 1 # 1
        u_a10 = 1 * m + 0 # 3
        u_a11 = 1 * m + 1 # 4
        
        v_b00 = 0 * p + k0 # k0
        v_b01 = 0 * p + k1 # k1
        v_b10 = 1 * p + k0 # 10+k0
        v_b11 = 1 * p + k1 # 10+k1
        
        w_c00 = k0 * n + 0 # 2*k0
        w_c01 = k1 * n + 0 # 2*k1
        w_c10 = k0 * n + 1 # 2*k0+1
        w_c11 = k1 * n + 1 # 2*k1+1

        # Strassen's 7 products for 2x2 matrix multiplication
        # M1 = (a00 + a11)(b00 + b11)
        U[u_a00][col] += 1
        U[u_a11][col] += 1
        V[v_b00][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M2 = (a10 + a11)b00
        U[u_a10][col] += 1
        U[u_a11][col] += 1
        V[v_b00][col] += 1
        W[w_c10][col] += 1
        W[w_c11][col] += -1
        col += 1
        
        # M3 = a00(b01 - b11)
        U[u_a00][col] += 1
        V[v_b01][col] += 1
        V[v_b11][col] += -1
        W[w_c01][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M4 = a11(b10 - b00)
        U[u_a11][col] += 1
        V[v_b10][col] += 1
        V[v_b00][col] += -1
        W[w_c00][col] += 1
        W[w_c10][col] += 1
        col += 1
        
        # M5 = (a00 + a01)b11
        U[u_a00][col] += 1
        U[u_a01][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += -1
        W[w_c01][col] += 1
        col += 1
        
        # M6 = (a10 - a00)(b00 + b01)
        U[u_a10][col] += 1
        U[u_a00][col] += -1
        V[v_b00][col] += 1
        V[v_b01][col] += 1
        W[w_c11][col] += 1
        col += 1
        
        # M7 = (a01 - a11)(b10 + b11)
        U[u_a01][col] += 1
        U[u_a11][col] += -1
        V[v_b10][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        col += 1

    # Part 2: A2 B2 (Naive)
    # A2 is A[:, 2], B2 is B[2, :]
    # A[i, 2] * B[2, k] -> C[i, k]
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    if col != rank:
        # Fallback to schoolbook if construction fails (should not happen)
        rank_schoolbook = n * m * p
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Verification step (optional but good for sanity, skip if slow)
    # Given the deterministic construction, verification is likely redundant but safe.
    # We'll skip heavy verification to save time, as the logic is standard Strassen.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
