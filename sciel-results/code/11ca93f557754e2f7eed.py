import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Block decomposition:
    # Split m=4 into two blocks of 2: j in {0,1} and j in {2,3}
    # For each 2x2 block of A, multiply by corresponding 2x7 block of B
    # For 2x2 x 2x7: split 7 into three 2s and one 1
    #   - Three 2x2x2 subproblems: Strassen (rank 7 each) = 21
    #   - One 2x2x1 subproblem: schoolbook (rank 4) = 4
    #   - Total per j-block: 25
    # Grand total: 2 * 25 = 50
    
    rank = 50
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        """Add Strassen decomposition for C[i0:i0+2, k0:k0+2] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]"""
        nonlocal col
        # A indices (flattened: i*m + j)
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        # B indices (flattened: j*p + k)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        
        # W indices (flattened: k*n + i)
        w00 = k0 * n + i0
        w01 = (k0 + 1) * n + i0
        w10 = k0 * n + (i0 + 1)
        w11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h) -> contributes to C[0,0] and C[1,1]
        U[a, col] += 1; U[d, col] += 1
        V[e, col] += 1; V[h, col] += 1
        W[w00, col] += 1; W[w11, col] += 1
        col += 1
        
        # M2 = (c+d)e -> contributes to C[1,0] and -C[1,1]
        U[c, col] += 1; U[d, col] += 1
        V[e, col] += 1
        W[w10, col] += 1; W[w11, col] -= 1
        col += 1
        
        # M3 = a(f-h) -> contributes to C[0,1] and C[1,1]
        U[a, col] += 1
        V[f, col] += 1; V[h, col] -= 1
        W[w01, col] += 1; W[w11, col] += 1
        col += 1
        
        # M4 = d(g-e) -> contributes to C[0,0] and C[1,0]
        U[d, col] += 1
        V[g, col] += 1; V[e, col] -= 1
        W[w00, col] += 1; W[w10, col] += 1
        col += 1
        
        # M5 = (a+b)h -> contributes to -C[0,0] and C[0,1]
        U[a, col] += 1; U[b, col] += 1
        V[h, col] += 1
        W[w00, col] -= 1; W[w01, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f) -> contributes to C[1,1]
        U[c, col] += 1; U[a, col] -= 1
        V[e, col] += 1; V[f, col] += 1
        W[w11, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h) -> contributes to C[0,0]
        U[b, col] += 1; U[d, col] -= 1
        V[g, col] += 1; V[h, col] += 1
        W[w00, col] += 1
        col += 1

    def add_schoolbook_221(i0, j0, k0):
        """Add schoolbook for C[i0:i0+2, k0] = A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0]"""
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        
        w0 = k0 * n + i0
        w1 = k0 * n + (i0 + 1)
        
        # P1 = a*e -> C[0, k0]
        U[a, col] += 1; V[e, col] += 1; W[w0, col] += 1
        col += 1
        # P2 = b*g -> C[0, k0]
        U[b, col] += 1; V[g, col] += 1; W[w0, col] += 1
        col += 1
        # P3 = c*e -> C[1, k0]
        U[c, col] += 1; V[e, col] += 1; W[w1, col] += 1
        col += 1
        # P4 = d*g -> C[1, k0]
        U[d, col] += 1; V[g, col] += 1; W[w1, col] += 1
        col += 1

    # Process two j-blocks: j in {0,1} and j in {2,3}
    for j_start in [0, 2]:
        if time.time() > deadline:
            break
        # Three 2x2x2 Strassen blocks for k in {0,1}, {2,3}, {4,5}
        for k_start in [0, 2, 4]:
            add_strassen_222(0, j_start, k_start)
        # One 2x2x1 schoolbook block for k=6
        add_schoolbook_221(0, j_start, 6)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Verify the tensor decomposition
    def verify(U_mat, V_mat, W_mat, tol=1e-9):
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
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
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
# EVOLVE-BLOCK-END
