import numpy as np
import time

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill

    # Fallback to schoolbook if heuristic fails or exceeds time
    def get_schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    try:
        # We construct a rank 50 solution based on block decomposition.
        # Split m=4 into two blocks of size 2: j in {0,1} and j in {2,3}.
        # For each block, split p=7 into three blocks of size 2 (k in {0,1}, {2,3}, {4,5}) 
        # and one block of size 1 (k=6).
        # For 2x2 x 2x2 subproblems, use Strassen (rank 7).
        # For 2x2 x 2x1 subproblems, use schoolbook (rank 4).
        # Total rank = 2 * (3 * 7 + 1 * 4) = 2 * (21 + 4) = 50.
        
        rank = 50
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        
        col = 0
        
        def add_strassen_block(i0, j0, k0):
            nonlocal col
            # A indices: A[i0:i0+2, j0:j0+2]
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            
            # B indices: B[j0:j0+2, k0:k0+2]
            e_idx = j0 * p + k0
            f_idx = j0 * p + (k0 + 1)
            g_idx = (j0 + 1) * p + k0
            h_idx = (j0 + 1) * p + (k0 + 1)
            
            # W indices: W[k*n+i, r]
            c00_idx = k0 * n + i0
            c01_idx = (k0 + 1) * n + i0
            c10_idx = k0 * n + (i0 + 1)
            c11_idx = (k0 + 1) * n + (i0 + 1)
            
            # M1 = (a+d)(e+h)
            U[a_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1; V[h_idx, col] += 1
            W[c00_idx, col] += 1; W[c11_idx, col] += 1
            col += 1
            
            # M2 = (c+d)e
            U[c_idx, col] += 1; U[d_idx, col] += 1
            V[e_idx, col] += 1
            W[c10_idx, col] += 1; W[c11_idx, col] -= 1
            col += 1
            
            # M3 = a(f-h)
            U[a_idx, col] += 1
            V[f_idx, col] += 1; V[h_idx, col] -= 1
            W[c01_idx, col] += 1; W[c11_idx, col] += 1
            col += 1
            
            # M4 = d(g-e)
            U[d_idx, col] += 1
            V[g_idx, col] += 1; V[e_idx, col] -= 1
            W[c00_idx, col] += 1; W[c10_idx, col] += 1
            col += 1
            
            # M5 = (a+b)h
            U[a_idx, col] += 1; U[b_idx, col] += 1
            V[h_idx, col] += 1
            W[c00_idx, col] -= 1; W[c01_idx, col] += 1
            col += 1
            
            # M6 = (c-a)(e+f)
            U[c_idx, col] += 1; U[a_idx, col] -= 1
            V[e_idx, col] += 1; V[f_idx, col] += 1
            W[c11_idx, col] += 1
            col += 1
            
            # M7 = (b-d)(g+h)
            U[b_idx, col] += 1; U[d_idx, col] -= 1
            V[g_idx, col] += 1; V[h_idx, col] += 1
            W[c00_idx, col] += 1
            col += 1

        def add_schoolbook_221_block(i0, j0, k0):
            nonlocal col
            a_idx = i0 * m + j0
            b_idx = i0 * m + (j0 + 1)
            c_idx = (i0 + 1) * m + j0
            d_idx = (i0 + 1) * m + (j0 + 1)
            
            e_idx = j0 * p + k0
            g_idx = (j0 + 1) * p + k0
            
            c00_idx = k0 * n + i0
            c10_idx = k0 * n + (i0 + 1)
            
            U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1; col += 1
            U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1; col += 1
            U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1; col += 1
            U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1; col += 1

        # Process the two j-blocks
        for j_start in [0, 2]:
            for k_start in [0, 2, 4]:
                add_strassen_block(0, j_start, k_start)
            add_schoolbook_221_block(0, j_start, 6)

        assert col == rank, f"Expected rank {rank}, got {col}"
        
        # Verify the tensor decomposition
        def verify_fast(U, V, W):
            r = U.shape[1]
            for i in range(n):
                for k in range(p):
                    S = np.zeros((m, m))
                    for r_idx in range(r):
                        u_col = U[:, r_idx]
                        w_val = W[k * n + i, r_idx]
                        v_col = V[:, r_idx]
                        for j in range(m):
                            for jp in range(m):
                                S[j, jp] += u_col[i * m + j] * w_val * v_col[jp * p + k]
                    if not np.allclose(S, np.eye(m), atol=1e-9):
                        return False
            return True

        if not verify_fast(U, V, W):
            raise ValueError("Verification failed")
            
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    except Exception as e:
        # Fallback to schoolbook if anything goes wrong
        return get_schoolbook()
# EVOLVE-BLOCK-END
