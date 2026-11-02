import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # The optimal known rank for M_{2,4,7} is 50.
    # This is achieved by splitting m=4 into two blocks of 2.
    # Each block is a 2x2x7 multiplication.
    # 2x2x7 is decomposed as 3*(2x2x2) + 1*(2x2x1).
    # Rank = 3*7 + 4 = 25 per block. Total = 50.
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Strassen decomposition for 2x2x2
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10, -C11
    # M3 = A00(B10-B11) -> C01, C11
    # M4 = A11(B01-B00) -> C00, C10
    # M5 = (A00+A01)B11 -> -C00, C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # Indices
        a00 = i0 * m + j0
        a01 = i0 * m + j0 + 1
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + j0 + 1
        
        b00 = j0 * p + k0
        b01 = j0 * p + k0 + 1
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + k0 + 1
        
        w00 = k0 * n + i0
        w01 = (k0 + 1) * n + i0
        w10 = k0 * n + i0 + 1
        w11 = (k0 + 1) * n + i0 + 1
        
        # M1
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[w00, col] += 1; W[w11, col] += 1
        col += 1
        
        # M2
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[w10, col] += 1; W[w11, col] -= 1
        col += 1
        
        # M3
        U[a00, col] += 1
        V[b10, col] += 1; V[b11, col] -= 1
        W[w01, col] += 1; W[w11, col] += 1
        col += 1
        
        # M4
        U[a11, col] += 1
        V[b01, col] += 1; V[b00, col] -= 1
        W[w00, col] += 1; W[w10, col] += 1
        col += 1
        
        # M5
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[w00, col] -= 1; W[w01, col] += 1
        col += 1
        
        # M6
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M7
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[w00, col] += 1
        col += 1

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        a00 = i0 * m + j0
        a01 = i0 * m + j0 + 1
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + j0 + 1
        
        b00 = j0 * p + k0
        b10 = (j0 + 1) * p + k0
        
        w00 = k0 * n + i0
        w10 = k0 * n + i0 + 1
        
        # P1 = A00 B00
        U[a00, col] += 1; V[b00, col] += 1; W[w00, col] += 1
        col += 1
        # P2 = A01 B10
        U[a01, col] += 1; V[b10, col] += 1; W[w00, col] += 1
        col += 1
        # P3 = A10 B00
        U[a10, col] += 1; V[b00, col] += 1; W[w10, col] += 1
        col += 1
        # P4 = A11 B10
        U[a11, col] += 1; V[b10, col] += 1; W[w10, col] += 1
        col += 1

    # Construct the decomposition
    # Two blocks for j in {0,1} and {2,3}
    for j_start in [0, 2]:
        if time.time() > deadline:
            break
        # Three Strassen blocks for k pairs (0,1), (2,3), (4,5)
        for k_start in [0, 2, 4]:
            add_strassen(0, j_start, k_start)
        # One Schoolbook block for k=6
        add_schoolbook_221(0, j_start, 6)
        
    # Verify rank
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")

    # Verification
    def verify(U_mat, V_mat, W_mat, tol=1e-6):
        r = U_mat.shape[1]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0.0
                        for c in range(r):
                            s += U_mat[i*m + j, c] * V_mat[jp*p + k, c] * W_mat[k*n + i, c]
                        expected = 1.0 if j == jp else 0.0
                        if abs(s - expected) > tol:
                            return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook (rank 56) if something went wrong
        rank_fb = n * m * p
        U = np.zeros((n * m, rank_fb), dtype=float)
        V = np.zeros((m * p, rank_fb), dtype=float)
        W = np.zeros((p * n, rank_fb), dtype=float)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, idx] = 1
                    V[j*p + k, idx] = 1
                    W[k*n + i, idx] = 1
                    idx += 1
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
