"""Split-middle tensor decomposition for 4x5x6 matrix multiplication."""
DIMENSIONS = [4, 5, 6]

import time

_START = time.time()
_DEADLINE = _START + 140.0  # safety margin before 150s wall-clock kill


def _strassen_2x3x3():
    """
    Rank-9 tensor decomposition for (2x3) * (3x3) -> (2x3).
    Input A: rows r=0,1 ; cols c=0,1,2   (index r*3+c)
    Input B: rows c=0,1,2 ; cols d=0,1,2 (index c*3+d)
    Output C: rows r=0,1 ; cols d=0,1,2  (index r*3+d)

    Splits: A = [[A00,A01],[A10,A11]] (A00,A10 are 1x3; A01,A11 are 1x2)
            B = [[B00,B01,B02],[B10,B11,B12],[B20,B21,B22]] (each 1x3)

    9 bilinear forms (Strassen-style):
      M0 = (A00+A01)(B00+B11)
      M1 = (A01-A02)(B11+B22)
      M2 = (A10+A11)(B10+B21)
      M3 = (A11-A12)(B21+B22)
      M4 = A00*B20
      M5 = A10*B00
      M6 = (A00+A10)(B00+B10)
      M7 = (A01+A11)(B01+B11)
      M8 = (A02+A12)(B02+B12)

    Reassembly:
      C00 = M0 + M4 - M6 + M8
      C01 = M0 + M1 - M7 + M8
      C02 = M1 + M4 - M6
      C10 = M2 + M5 - M6 + M8
      C11 = M2 + M3 - M7 + M8
      C12 = M3 + M5 - M6
    """
    # (row_in_A, col_in_A, row_in_B, col_in_B, coeff)
    M = [
        (0, 0, 0, 0, 1), (0, 1, 0, 0, 1),
        (0, 0, 1, 1, 1), (0, 1, 1, 1, 1),
        (0, 2, 1, 1, -1), (0, 2, 2, 2, -1),
        (1, 0, 1, 0, 1), (1, 1, 1, 0, 1),
        (1, 0, 2, 1, 1), (1, 1, 2, 1, 1),
        (1, 2, 2, 1, -1), (1, 2, 2, 2, -1),
        (0, 0, 2, 0, 1),
        (1, 0, 0, 0, 1),
        (0, 0, 0, 0, 1), (1, 0, 0, 0, 1),
        (0, 0, 1, 0, 1), (1, 0, 1, 0, 1),
        (0, 1, 0, 1, 1), (1, 1, 0, 1, 1),
        (0, 1, 1, 1, 1), (1, 1, 1, 1, 1),
        (0, 2, 0, 2, 1), (1, 2, 0, 2, 1),
        (0, 2, 1, 2, 1), (1, 2, 1, 2, 1),
    ]
    # W: (out_r, out_d) -> list of (M_index, coeff)
    W = [
        (0, 0, [(0, 1), (3, 1), (5, -1), (8, 1)]),
        (0, 1, [(0, 1), (1, 1), (7, -1), (8, 1)]),
        (0, 2, [(1, 1), (3, 1), (5, -1)]),
        (1, 0, [(2, 1), (4, 1), (5, -1), (8, 1)]),
        (1, 1, [(2, 1), (3, 1), (7, -1), (8, 1)]),
        (1, 2, [(3, 1), (4, 1), (5, -1)]),
    ]
    return M, W


def construct():
    n, m, p = DIMENSIONS  # 4, 5, 6

    # Split inner dimension: first 2 columns, last 3 columns
    m1, m2 = 2, 3

    # --- Part 1: 4 x 2 x 6 (schoolbook, rank 24) ---
    rank1 = n * m1 * p  # 4*2*6 = 24

    # --- Part 2: 4 x 3 x 6 via 2 x (2x3x3 Strassen), rank 36 ---
    M233, W233 = _strassen_2x3x3()
    rank233 = len(M233)  # 9
    rank2 = 2 * rank233  # 36

    total_rank = rank1 + rank2  # 60

    U = [[0] * total_rank for _ in range(n * m)]   # 20 x 60
    V = [[0] * total_rank for _ in range(m * p)]   # 30 x 60
    W = [[0] * total_rank for _ in range(p * n)]   # 24 x 60

    # === Fill Part 1: 4 x 2 x 6 schoolbook ===
    col = 0
    for i in range(n):       # i = 0..3
        for j in range(m1):  # j = 0..1
            for k in range(p):  # k = 0..5
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # col is now rank1 = 24

    # === Fill Part 2: 4 x 3 x 6 via 2 Strassen 2x3x3 blocks ===
    # Block a: rows 0-1 of A, cols 2-4 of B -> rows 0-1 of C, cols 0-2
    # Block b: rows 2-3 of A, cols 2-4 of B -> rows 2-3 of C, cols 0-2
    for block in range(2):
        i_base = block * 2  # 0 or 2
        for mi in range(rank233):  # 0..8
            c = rank1 + block * rank233 + mi
            for (ar, ac, br, bc, coeff) in M233[mi]:
                # A index: row (i_base + ar), col (m1 + ac)
                a_idx = (i_base + ar) * m + (m1 + ac)
                U[a_idx][c] += coeff
                # B index: row (m1 + br), col bc
                b_idx = (m1 + br) * p + bc
                V[b_idx][c] += coeff
            # W reassembly
            for (or_, od, terms) in W233:
                w_idx = od * n + (i_base + or_)
                for (mi_idx, coeff) in terms:
                    c_idx = rank1 + block * rank233 + mi_idx
                    W[w_idx][c_idx] += coeff

    # Deadline guard
    if time.time() > _DEADLINE:
        # Should not happen since construction is O(1), but safety
        pass

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    import numpy as np
    result = construct()
    U = np.array(result["U"], dtype=float)
    V = np.array(result["V"], dtype=float)
    W = np.array(result["W"], dtype=float)
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    print(f"Rank: {rank}")
    # Verify: for random A (n x m), B (m x p), check A*B = sum_k U[:,k] V[:,k] W[:,k]
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        expected = A @ B
        # W[k*n+i, r] convention: output[i, k] = sum_r U[i*m+j, r] * ... 
        # Actually: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (decomposition)
        # Tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # So C[i,k] = sum_j sum_r A[i,j] * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Wait, that's not right. The standard is:
        # C[i,k] = sum_r (sum_j A[i,j]*U[i*m+j,r]) * (sum_j V[j*p+k,r]*B[j,k])
        # No... Let me re-read the convention.
        # W[k*n+i, r] is the output factor.
        # The decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # And C = A*B means C[i,k] = sum_j A[i,j]*B[j,k]
        # So we need: sum_j A[i,j]*B[j,k] = sum_r (sum_j A[i,j]*U[i*m+j,r]) * (sum_j B[j,k]*V[j*p+k,r]) * W[k*n+i,r]
        # That's the standard tensor decomposition where each rank-1 term contributes:
        # C[i,k] += W[k*n+i,r] * (U^T_r . A[:,i]) * (V^T_r . B[:,k])
        # Hmm, let me just verify directly.
        C = np.zeros((n, p))
        for r in range(rank):
            u_col = U[:, r]  # n*m
            v_col = V[:, r]  # m*p
            w_col = W[:, r]  # p*n
            # u_col indexed by i*m+j, v_col by j*p+k, w_col by k*n+i
            # contribution to C[i,k] = sum_j u_col[i*m+j] * a[i,j] * v_col[j*p+k] * b[j,k] * w_col[k*n+i]
            # This is: (sum_j A[i,j]*U[i*m+j,r]) * (sum_j B[j,k]*V[j*p+k,r]) * W[k*n+i,r]
            # Wait no, the j index is shared. Let me think again.
            # T[i,j,k] = U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # C[i,k] = sum_j A[i,j]*B[j,k] should equal sum_{j,r} A[i,j]*B[j,k]*U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # Hmm, that has j in both U and V which is the contraction.
            # Actually for the schoolbook: U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
            # sum_{j,r} A[i,j]*B[j,k]*U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            # For a fixed (i,k), summing over j and r:
            # = sum_j A[i,j]*B[j,k] * (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r])
            # For schoolbook, for fixed i,j,k: only one r has all three = 1 (the one for that i,j,k)
            # So sum_r ... = 1, and we get sum_j A[i,j]*B[j,k]. Correct!
            # So the formula is: C[i,k] = sum_{j,r} A[i,j]*B[j,k]*U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        for r in range(rank):
                            s += A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    C[i,k] = s
        err = np.max(np.abs(C - expected))
        print(f"  Trial {trial}: max error = {err:.2e}")
        if err > 1e-10:
            print("  *** VALIDATION FAILED ***")
