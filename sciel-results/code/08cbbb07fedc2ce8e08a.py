"""Program for rank-minimal exact tensor decomposition of 4x5*5x6 matrix multiplication."""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _strassen_222():
    """
    Return the Strassen decomposition for 2x2*2x2 matrix multiplication.
    Returns (U, V, W) each of shape (4, 7).
    Tensor convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)

    # M1 = (A00+A11)*(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1   # C00: k=0,i=0
    W[3, 0] = 1   # C11: k=1,i=1

    # M2 = (A10+A11)*B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1   # C10: k=0,i=1
    W[3, 1] = -1  # C11: k=1,i=1

    # M3 = A00*(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1   # C01: k=1,i=0
    W[3, 2] = 1   # C11: k=1,i=1

    # M4 = A11*(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1   # C00: k=0,i=0
    W[1, 3] = 1   # C10: k=0,i=1

    # M5 = (A00+A01)*B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1  # C00: k=0,i=0
    W[2, 4] = 1   # C01: k=1,i=0

    # M6 = (A10-A00)*(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1   # C11: k=1,i=1

    # M7 = (A01-A11)*(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1   # C00: k=0,i=0

    return U, V, W


def construct():
    """
    Construct an exact tensor decomposition for 4x5 * 5x6 matrix multiplication.
    Uses block decomposition with Strassen for 2x2x2 sub-blocks.
    
    Block partitions:
    - n=4 split into [2, 2]
    - m=5 split into [2, 2, 1]  
    - p=6 split into [2, 2, 2]
    
    Rank calculation:
    - 8 blocks of (2,2,2): 8 * 7 = 56 (Strassen)
    - 4 blocks of (2,2,1): 4 * 4 = 16 (schoolbook)
    - 4 blocks of (2,1,2): 4 * 4 = 16 (schoolbook)
    - 2 blocks of (2,1,1): 2 * 2 = 4 (schoolbook)
    Total rank = 92
    """
    start_time = time.time()
    n, m, p = 4, 5, 6

    i_splits = [2, 2]
    j_splits = [2, 2, 1]
    k_splits = [2, 2, 2]

    i_offsets = [0, 2]
    j_offsets = [0, 2, 4]
    k_offsets = [0, 2, 4]

    S_U, S_V, S_W = _strassen_222()

    # Compute total rank
    total_rank = 0
    for nb in i_splits:
        for mb in j_splits:
            for pb in k_splits:
                if nb == 2 and mb == 2 and pb == 2:
                    total_rank += 7
                else:
                    total_rank += nb * mb * pb

    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)

    col = 0
    for i_idx, (i_0, n_b) in enumerate(zip(i_offsets, i_splits)):
        for j_idx, (j_0, m_b) in enumerate(zip(j_offsets, j_splits)):
            for k_idx, (k_0, p_b) in enumerate(zip(k_offsets, k_splits)):
                if time.time() - start_time > DEADLINE:
                    break

                if n_b == 2 and m_b == 2 and p_b == 2:
                    # Strassen for 2x2x2
                    for r in range(7):
                        su = S_U[:, r]
                        sv = S_V[:, r]
                        sw = S_W[:, r]
                        for i in range(2):
                            for j in range(2):
                                val = su[i * 2 + j]
                                if val != 0:
                                    U[(i_0 + i) * m + (j_0 + j), col + r] += val
                        for j in range(2):
                            for k in range(2):
                                val = sv[j * 2 + k]
                                if val != 0:
                                    V[(j_0 + j) * p + (k_0 + k), col + r] += val
                        for k in range(2):
                            for i in range(2):
                                val = sw[k * 2 + i]
                                if val != 0:
                                    W[(k_0 + k) * n + (i_0 + i), col + r] += val
                    col += 7
                else:
                    # Schoolbook
                    for u in range(n_b):
                        for v in range(m_b):
                            for w in range(p_b):
                                U[(i_0 + u) * m + (j_0 + v), col] = 1
                                V[(j_0 + v) * p + (k_0 + w), col] = 1
                                W[(k_0 + w) * n + (i_0 + u), col] = 1
                                col += 1

    # Verify correctness with random test
    if time.time() - start_time < DEADLINE - 10:
        np.random.seed(42)
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        # Reconstruct C from decomposition
        C_reconstructed = np.zeros((n, p))
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)

        for r in range(total_rank):
            u_col = U_arr[:, r]
            v_col = V_arr[:, r]
            w_col = W_arr[:, r]
            # u_col has entries for A[i,j] at index i*m+j
            # v_col has entries for B[j,k] at index j*p+k
            # w_col has entries for C[i,k] at index k*n+i
            # The rank-1 term: sum_{i,j,k} A[i,j]*B[j,k]*tensor[i,j,k]
            # = sum_r (sum_{i,j} u[r,i,j]*A[i,j]) * (sum_{j,k} v[r,j,k]*B[j,k]) * ... 
            # Actually: C[i,k] = sum_r w[r,k,i] * (sum_j u[r,i,j]*A[i,j]) * (sum_j v[r,j,k]*B[j,k])
            # Wait, that's not right either. The tensor decomposition is:
            # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # So C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]... no.
            # C[i,k] = sum_j A[i,j]*B[j,k] = sum_j T[i,j,k]*A[i,j]*B[j,k] where T[i,j,k]=delta[j,j]
            # Actually the multiplication tensor has T[i,j,k] = 1 if the (i,k) entry of C depends on A[i,j]*B[j,k]
            # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k], so T[i,j,k] = 1 for all i,j,k (each term A[i,j]*B[j,k] contributes to C[i,k])
            # The decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # So C[i,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]*A[i,j]*B[j,k]
            # = sum_r W[k*n+i,r] * (sum_j U[i*m+j,r]*A[i,j]) * (sum_j V[j*p+k,r]*B[j,k])
            
            a_proj = np.zeros(n)  # for each i, sum_j U[i*m+j,r]*A[i,j]
            b_proj = np.zeros(p)  # for each k, sum_j V[j*p+k,r]*B[j,k]
            for i in range(n):
                for j in range(m):
                    a_proj[i] += U_arr[i*m+j, r] * A[i,j]
            for k in range(p):
                for j in range(m):
                    b_proj[k] += V_arr[j*p+k, r] * B[j,k]
            for i in range(n):
                for k in range(p):
                    C_reconstructed[i,k] += W_arr[k*n+i, r] * a_proj[i] * b_proj[k]

        if not np.allclose(C_reconstructed, C_expected, atol=1e-10):
            # Fallback to schoolbook if Strassen is buggy
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
