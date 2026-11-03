import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact rank-189 decomposition for 6x6 matrix multiplication.
    Uses hierarchical Strassen: 2x2 blocks of size 3x3.
    The inner 3x3 multiplications use the schoolbook (rank-27) algorithm.
    Total rank = 7 * 27 = 189.
    """
    n, m, p = DIMENSIONS
    assert n == 6 and m == 6 and p == 6

    deadline = time.time() + 120

    bs = 3
    rank3 = 27
    rank = 7 * rank3

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # Strassen definitions for 2x2 block multiplication
    # Each entry: list of (block_index, coefficient)
    # Block indices: 0=A11, 1=A12, 2=A21, 3=A22
    strassen_defs = [
        {'A': [(0, 1.0), (3, 1.0)], 'B': [(0, 1.0), (3, 1.0)]},
        {'A': [(2, 1.0), (3, 1.0)], 'B': [(0, 1.0)]},
        {'A': [(0, 1.0)], 'B': [(1, 1.0), (3, -1.0)]},
        {'A': [(3, 1.0)], 'B': [(2, 1.0), (0, -1.0)]},
        {'A': [(0, 1.0), (1, 1.0)], 'B': [(3, 1.0)]},
        {'A': [(2, 1.0), (0, -1.0)], 'B': [(0, 1.0), (1, 1.0)]},
        {'A': [(1, 1.0), (3, -1.0)], 'B': [(2, 1.0), (3, 1.0)]},
    ]

    # Reconstruction coefficients: which Strassen products contribute to each output block
    # recon_coeffs[output_block][strassen_idx] = coefficient
    # Output blocks: 0=C11, 1=C12, 2=C21, 3=C22
    recon_coeffs = [
        [(0, 1.0), (3, 1.0), (4, -1.0), (6, 1.0)],   # C11
        [(2, 1.0), (4, 1.0)],                         # C12
        [(1, 1.0), (3, 1.0)],                         # C21
        [(0, 1.0), (1, -1.0), (2, 1.0), (5, 1.0)],   # C22
    ]

    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        for u in range(bs):
            for v in range(bs):
                for j in range(bs):
                    # Fill U entries
                    for block_idx, coeff_a in s_def['A']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        i_glob = br * bs + u
                        k_glob = bc * bs + v
                        u_row = i_glob * m + k_glob
                        U[u_row][r_idx] += coeff_a

                    # Fill V entries
                    for block_idx, coeff_b in s_def['B']:
                        br = block_idx // 2
                        bc = block_idx % 2
                        k_glob = br * bs + v
                        j_glob = bc * bs + j
                        v_row = k_glob * p + j_glob
                        V[v_row][r_idx] += coeff_b

                    # Fill W entries
                    for c_block_idx in range(4):
                        for (s_idx, g) in recon_coeffs[c_block_idx]:
                            if s_idx == s:
                                br = c_block_idx // 2
                                bc = c_block_idx % 2
                                i_glob = br * bs + u
                                j_glob = bc * bs + j
                                w_row = j_glob * n + i_glob
                                W[w_row][r_idx] += g
                                break

                    r_idx += 1

    assert r_idx == rank

    # Quick sanity check: verify with random matrices
    if time.time() < deadline:
        rng = np.random.RandomState(42)
        A_test = rng.randint(-5, 5, (n, m)).astype(float)
        B_test = rng.randint(-5, 5, (m, p)).astype(float)
        C_expected = A_test @ B_test

        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)

        # C[i,j] = sum_k A[i,k] * B[k,j]
        # Using tensor decomposition: C[i,j] = sum_r U[i*m+k... wait
        # U has shape (n*m, rank), indexed by (i, k) -> row i*m+k
        # V has shape (m*p, rank), indexed by (k, j) -> row k*p+j
        # W has shape (p*n, rank), indexed by (j, i) -> row j*n+i (trace convention W[k*n+i, r] where k=j, i=i)
        
        # C[i,j] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r] ... 
        # Wait, that's not right. Let me think about the tensor structure.
        # The multiplication tensor T[i,j,k] = A[i,k]*B[k,j]
        # Decomposition: T[i,j,k] = sum_r U[i*m+k, r] * V[k*p+j, r] * W[j*n+i, r]
        # So C[i,j] = sum_k T[i,j,k] = sum_k sum_r U[i*m+k,r]*V[k*p+j,r]*W[j*n+i,r]
        # = sum_r W[j*n+i,r] * (sum_k U[i*m+k,r]*V[k*p+j,r])
        
        # Let's compute properly
        C_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                total = 0.0
                for k in range(m):
                    for r in range(rank):
                        total += U[i*m+k][r] * V[k*p+j][r] * W[j*n+i][r]
                C_computed[i, j] = total

        max_err = np.max(np.abs(C_computed - C_expected))
        if max_err > 1e-10:
            raise ValueError(f"Decomposition verification failed: max error = {max_err}")

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
