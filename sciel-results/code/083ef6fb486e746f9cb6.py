"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    start_time = time.time()
    deadline = start_time + 115.0

    # Strassen-based rank-84 decomposition for 4x4 x 4x6 -> 4x6
    # Split into 2x2x2 sub-blocks: (4/2)*(4/2)*(6/2) = 2*2*3 = 12 blocks
    # Each 2x2x2 block: 8 schoolbook -> 7 Strassen terms
    # Total rank: 12 * 7 = 84
    #
    # For each 2x2x2 sub-block at (i_g, j_g, k_g):
    #   i0,i1 = 2*i_g, 2*i_g+1  (row indices of A and C)
    #   j0,j1 = 2*j_g, 2*j_g+1  (col indices of A, row indices of B)
    #   k0,k1 = 2*k_g, 2*k_g+1  (col indices of B and C)
    #
    # The 7 Strassen products for C = A*B in 2x2x2:
    #   M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    #   M2 = (A[i0,j1]+A[i1,j1]) * B[j0,k0]
    #   M3 = A[i0,j0] * (B[j1,k1]-B[j0,k0])
    #   M4 = A[i1,j0] * (B[j1,k0]-B[j0,k0])
    #   M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    #   M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
    #   M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
    #
    # C[i0,k0] = M1 - M2 + M3 + M6
    # C[i0,k1] = M1 - M5 + M7
    # C[i1,k0] = M1 - M3 + M4
    # C[i1,k1] = M2 + M5 - M7
    #
    # Each M_t is a rank-1 tensor: product of a linear combo of A entries
    # and a linear combo of B entries, contributing to specific C entries.
    #
    # For term t, the rank-1 tensor has:
    #   U[i*m+j, t] = coefficient of A[i,j] in the A-part of M_t
    #   V[j*p+k, t] = coefficient of B[j,k] in the B-part of M_t
    #   W[k*n+i, t] = coefficient in the C-part (how M_t contributes to C[i,k])

    rank = 84
    U = np.zeros((n * m, rank))  # 16 x 84
    V = np.zeros((m * p, rank))  # 24 x 84
    W = np.zeros((p * n, rank))  # 24 x 84

    col = 0
    for i_g in range(2):
        for j_g in range(2):
            for k_g in range(3):
                i0, i1 = 2 * i_g, 2 * i_g + 1
                j0, j1 = 2 * j_g, 2 * j_g + 1
                k0, k1 = 2 * k_g, 2 * k_g + 1

                # Define the 7 Strassen terms for this block
                # Each term: (a_coeffs, b_coeffs, c_coeffs)
                # a_coeffs: dict mapping (i,j) -> coeff for A part
                # b_coeffs: dict mapping (j,k) -> coeff for B part
                # c_coeffs: dict mapping (i,k) -> coeff for C part

                terms = [
                    # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
                    # C[i0,k0]+=M1, C[i0,k1]+=M1, C[i1,k0]+=M1, C[i1,k1]+=M1
                    # But with signs from the reconstruction formulas:
                    # C[i0,k0] = M1 - M2 + M3 + M6
                    # C[i0,k1] = M1 - M5 + M7
                    # C[i1,k0] = M1 - M3 + M4
                    # C[i1,k1] = M2 + M5 - M7
                    # So M1 contributes +1 to all four C entries
                    (
                        {(i0, j0): 1, (i1, j1): 1},  # A part
                        {(j0, k0): 1, (j1, k1): 1},  # B part
                        {(i0, k0): 1, (i0, k1): 1, (i1, k0): 1, (i1, k1): 1}  # C part
                    ),
                    # M2 = (A[i0,j1]+A[i1,j1]) * B[j0,k0]
                    # C[i0,k0] -= M2, C[i1,k1] += M2
                    (
                        {(i0, j1): 1, (i1, j1): 1},
                        {(j0, k0): 1},
                        {(i0, k0): -1, (i1, k1): 1}
                    ),
                    # M3 = A[i0,j0] * (B[j1,k1]-B[j0,k0])
                    # C[i0,k0] += M3, C[i1,k0] -= M3
                    (
                        {(i0, j0): 1},
                        {(j1, k1): 1, (j0, k0): -1},
                        {(i0, k0): 1, (i1, k0): -1}
                    ),
                    # M4 = A[i1,j0] * (B[j1,k0]-B[j0,k0])
                    # C[i1,k0] += M4
                    (
                        {(i1, j0): 1},
                        {(j1, k0): 1, (j0, k0): -1},
                        {(i1, k0): 1}
                    ),
                    # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
                    # C[i0,k1] -= M5, C[i1,k1] += M5
                    (
                        {(i0, j0): 1, (i0, j1): 1},
                        {(j1, k1): 1},
                        {(i0, k1): -1, (i1, k1): 1}
                    ),
                    # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
                    # C[i0,k0] += M6
                    (
                        {(i1, j0): 1, (i0, j0): -1},
                        {(j0, k0): 1, (j0, k1): 1},
                        {(i0, k0): 1}
                    ),
                    # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
                    # C[i0,k1] += M7, C[i1,k1] -= M7
                    (
                        {(i0, j1): 1, (i1, j1): -1},
                        {(j1, k0): 1, (j1, k1): 1},
                        {(i0, k1): 1, (i1, k1): -1}
                    ),
                ]

                for a_coeffs, b_coeffs, c_coeffs in terms:
                    # Fill U: U[i*m + j, col] = coeff
                    for (i, j), coeff in a_coeffs.items():
                        U[i * m + j, col] += coeff
                    # Fill V: V[j*p + k, col] = coeff
                    for (j, k), coeff in b_coeffs.items():
                        V[j * p + k, col] += coeff
                    # Fill W: W[k*n + i, col] = coeff
                    for (i, k), coeff in c_coeffs.items():
                        W[k * n + i, col] += coeff
                    col += 1

    # Verify: check that U @ (V * W) reconstructs the matrix multiplication tensor
    # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # should equal 1 if i,j,k form a valid (i,j,k) triple for C[i,k] = sum_j A[i,j]*B[j,k]
    # i.e., T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # and 0 otherwise (but in our formulation, every (i,j,k) is valid)
    
    # Let's verify by checking a few entries
    # Actually, let's do a full verification
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This should equal 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    
    # Quick verification
    T_reconstructed = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_reconstructed[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
    
    # Check if it's all ones
    if not np.allclose(T_reconstructed, 1.0):
        # Fallback to schoolbook rank 96
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, c] = 1
                    V_fb[j * p + k, c] = 1
                    W_fb[k * n + i, c] = 1
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
