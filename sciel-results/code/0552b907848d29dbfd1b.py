"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS

    # Build rank-35 decomposition using 5 independent Strassen blocks
    # A is 2x2, B is 2x10, C is 2x10
    # For each column block t (0..4), we decompose the 2x2 sub-product
    # C[:, 2t:2t+2] = A @ B[:, 2t:2t+2] using Strassen (rank 7)
    #
    # Total rank = 5 * 7 = 35
    #
    # The tensor T for C = A*B has entries T[i,j,k] = delta_{i,j} ... no wait.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T_{i,k,j} = A[i,j] * B[j,k] -> but in bilinear form:
    # C[i,k] = sum_{j} sum_{r} U[i,j, r] * W[j,k,r] * V[r]
    # Actually the standard decomposition:
    # C[i,k] = sum_r (sum_j A[i,j]*U[j,r]) * (sum_k' B[k',k]*V[k',r]) * W[i,k,r]
    #
    # Let me use the conventional formulation:
    # C = A*B, C[i,k] = sum_j A[i,j] B[j,k]
    # Bilinear: C[i,k] = sum_r M_r(i) * M_r(j) * M_r(k) ... no.
    #
    # The matrix multiplication tensor T has T[i,j,k] = delta_{i,j}... no.
    # T_{i,j,k} represents the coefficient in C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor is T with T[i,j,k] = 1 if we think of it as A[i,j]*B[j,k]
    #
    # In the factorization: T = sum_r U_r ⊗ V_r ⊗ W_r
    # where U_r is (n*m), V_r is (m*p), W_r is (p*n)
    # U_r[i*m+j] corresponds to A[i,j]
    # V_r[j*p+k] corresponds to B[j,k]
    # W_r[k*n+i] corresponds to C[i,k]
    #
    # For Strassen on block t (columns 2t, 2t+1 of B, and columns 2t, 2t+1 of C):
    # The 2x2 block: C_block = A * B_block
    # where B_block = B[:, 2t:2t+2]
    #
    # Strassen terms for 2x2*2x2 -> 2x2:
    # A_vec index: [A[0,0], A[0,1], A[1,0], A[1,1]] -> flat: [0,1,2,3]
    # B_vec index: [B[0,0], B[0,1], B[1,0], B[1,1]] -> in our V: 
    #   B[0,2t] -> V[0*p+2t], B[0,2t+1] -> V[0*p+2t+1]
    #   B[1,2t] -> V[1*p+2t], B[1,2t+1] -> V[1*p+2t+1]
    # C_vec: [C[0,0], C[1,0], C[0,1], C[1,1]] in local 2x2
    #   C[0,2t]   -> W[2t*n+0] = W[4t]
    #   C[1,2t]   -> W[2t*n+1] = W[4t+1]
    #   C[0,2t+1] -> W[(2t+1)*n+0] = W[4t+2]
    #   C[1,2t+1] -> W[(2t+1)*n+1] = W[4t+3]
    #
    # Strassen rank-7 terms (a_vec for A, b_vec for B, c_vec for C):
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6

    strassen = [
        # (A_vec, B_vec, C_vec)
        # A_vec: [a00, a01, a10, a11]
        # B_vec: [b00, b01, b10, b11]
        # C_vec: [c00, c10, c01, c11]
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 1, 0, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]

    rank = 35
    U = np.zeros((n * m, rank), dtype=np.int32)
    V = np.zeros((m * p, rank), dtype=np.int32)
    W = np.zeros((p * n, rank), dtype=np.int32)

    col = 0
    for t in range(p // 2):
        if time.time() > deadline:
            break
        for a_vec, b_vec, c_vec in strassen:
            # Fill U
            for idx in range(n * m):
                U[idx, col] = a_vec[idx]
            # Fill V
            V[0 * p + 2 * t, col] = b_vec[0]
            V[0 * p + 2 * t + 1, col] = b_vec[1]
            V[1 * p + 2 * t, col] = b_vec[2]
            V[1 * p + 2 * t + 1, col] = b_vec[3]
            # Fill W
            W[2 * t * n + 0, col] = c_vec[0]
            W[2 * t * n + 1, col] = c_vec[1]
            W[(2 * t + 1) * n + 0, col] = c_vec[2]
            W[(2 * t + 1) * n + 1, col] = c_vec[3]
            col += 1

    # Verify the decomposition: for random A (2x2) and B (2x10),
    # check that sum_r U_r^T A * V_r^T B * W_r == A@B
    if time.time() < deadline - 5:
        rng = np.random.default_rng(42)
        for trial in range(100):
            A = rng.integers(-10, 11, size=(n, m)).astype(float)
            B = rng.integers(-10, 11, size=(m, p)).astype(float)
            C_expected = A @ B
            C_reconstructed = np.zeros((n, p))
            for r in range(rank):
                # U[r] is (n*m), reshape to (n,m)
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)  # W[k*n+i] -> reshape to (n,p)? 
                # W[k*n+i] means W is indexed as [k*n+i], so W.reshape(n, p) gives W[i,k]?
                # Actually W[k*n+i] -> if we reshape W to (p, n) we get W[k, i] = W[k*n+i]
                # But C[i,k] should get contribution from Wr
                # Let me reconsider: the bilinear form is
                # C[i,k] += Ur[i,m_idx] * Vr[m_idx,k] * Wr[k*n+i]
                # So Wr is indexed as [k*n+i] for output C[i,k]
                # Wr.reshape(p, n) gives Wr[k, i] = W[k*n + i]
                Wr_mat = Wr.reshape(p, n)
                # C[i,k] contribution = sum_j Ur[i,j] * Vr[j,k] * Wr[k,i]
                tmp = Ur @ Vr  # (n, p)
                C_reconstructed += tmp * Wr_mat.T  # Wr_mat.T[i,k] = Wr[k,i] = W[k*n+i]
            if not np.allclose(C_reconstructed, C_expected, atol=1e-9):
                # Decomposition is wrong; fall back to schoolbook rank 40
                U = np.zeros((n * m, 40), dtype=np.int32)
                V = np.zeros((m * p, 40), dtype=np.int32)
                W = np.zeros((p * n, 40), dtype=np.int32)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            r = i * m * p + j * p + k
                            U[i * m + j, r] = 1
                            V[j * p + k, r] = 1
                            W[k * n + i, r] = 1
                break

    # Convert to lists of lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
