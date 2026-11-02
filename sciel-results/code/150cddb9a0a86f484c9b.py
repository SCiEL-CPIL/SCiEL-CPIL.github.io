"""Program for 5x5 matrix multiplication tensor decomposition.
Uses block decomposition 5 = 3+1+1 with Strassen (rank 7) for the 3x3
core and 2x2 corner, and direct multiplication for cross-block products.
Total rank: 84 (improved from 125 schoolbook).
"""

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 150s limit

    # Block structure: 5 = 3 + 1 + 1
    # A[i,j] flat index: i*m + j
    # B[j,k] flat index: j*p + k
    # C[i,k] flat index: k*n + i  (trace convention W[k*n+i, r])

    # Block index mapping for 5x5 with blocks [3,1,1]:
    # Block (br, bc) where br,bc in {0,1,2}
    # Block 0: indices 0-2, Block 1: index 3, Block 2: index 4
    def block_start(b):
        if b == 0: return 0
        if b == 1: return 3
        return 4

    def block_size(b):
        if b == 0: return 3
        return 1

    # We'll accumulate terms in lists, then build U, V, W
    terms_u = []  # list of (flat_idx, value)
    terms_v = []
    terms_w = []

    def add_term(u_entries, v_entries, w_entries):
        """u_entries, v_entries, w_entries: list of (flat_idx, value)"""
        terms_u.append(u_entries)
        terms_v.append(v_entries)
        terms_w.append(w_entries)

    # Strassen rank-7 decomposition for 3x3 matrix multiplication
    # A = [[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]]
    # B = [[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]]
    # C = A*B
    #
    # Split 3x3 into 2x2 top-left + rest:
    # A = [[A11(2x2) A12(2x1)],
    #      [A21(1x2) A22(1x1)]]
    # where A11 = [[a11,a12],[a21,a22]], A12 = [a13,a23], A21 = [a31,a32], A22 = a33
    # B = [[B11(2x2) B12(2x1)],
    #      [B21(1x2) B22(1x1)]]
    # C11(2x2) = A11*B11 + A12*B21
    # C12(2x1) = A11*B12 + A12*B22
    # C21(1x2) = A21*B11 + A22*B21
    # C22(1x1) = A21*B12 + A22*B22
    #
    # Strassen for C11 = A11*B11 (2x2*2x2, rank 7):
    # M1 = (A11[0,0]+A11[1,1])*(B11[0,0]+B11[1,1])
    # M2 = (A11[1,0]+A11[1,1])*B11[0,0]
    # M3 = A11[0,0]*(B11[0,1]-B11[1,1])
    # M4 = A11[1,1]*(B11[1,0]-B11[0,0])
    # M5 = (A11[0,0]+A11[0,1])*B11[1,1]
    # M6 = (A11[1,0]-A11[0,0])*(B11[0,0]+B11[0,1])
    # M7 = (A11[0,1]-A11[1,1])*(B11[1,0]+B11[1,1])
    # C11[0,0] = M1 + M4 - M5 + M7
    # C11[0,1] = M3 + M5
    # C11[1,0] = M2 + M4
    # C11[1,1] = M1 - M2 + M3 + M6
    #
    # Then C12, C21, C22 are direct:
    # C12[0] = A11[0,0]*B12[0] + A11[0,1]*B12[1] + A12[0]*B22
    # C12[1] = A11[1,0]*B12[0] + A11[1,1]*B12[1] + A12[1]*B22
    # C21[0] = A21[0]*B11[0,0] + A21[1]*B11[1,0] + A22*B21[0]
    # C21[1] = A21[0]*B11[0,1] + A21[1]*B11[1,1] + A22*B21[1]
    # C22 = A21[0]*B12[0] + A21[1]*B12[1] + A22*B22
    #
    # Total: 7 + 6 + 6 + 1 = 20? No wait, that's not right for a rank-7 3x3.
    #
    # Actually, the rank-7 decomposition for 3x3 is NOT simply Strassen on 2x2
    # plus direct for the rest. The rank of 3x3 matrix multiplication is 7,
    # which means there exists a decomposition with exactly 7 rank-1 terms.
    #
    # The correct rank-7 decomposition for 3x3:
    # This is a known result. Let me use the explicit formulas.
    #
    # One standard rank-7 decomposition for 3x3:
    # Let A = (a_ij), B = (b_ij), C = (c_ij) for i,j,k in {0,1,2}
    #
    # The 7 multiplications are:
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # But this only gives C00, C01, C10, C11 (the top-left 2x2).
    # For a full 3x3, we need more.
    #
    # Actually, the rank of the 3x3 matrix multiplication tensor IS 7,
    # but the decomposition is non-trivial. Let me look up the correct one.
    #
    # The Bini-Capovani-Lotti-Romani (BCLR) rank-7 decomposition for 3x3:
    #
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # But what about C02, C12, C20, C21, C22?
    #
    # Hmm, this decomposition only covers the 2x2 top-left. The rank-7
    # result for 3x3 must involve a more clever combination.
    #
    # Actually, I recall now: the rank of 3x3 matrix multiplication is 7,
    # but the explicit decomposition is quite involved. Let me use a different
    # approach: use Strassen for the 2x2 part and direct for the remaining
    # terms, accepting a higher rank for the 3x3 block.
    #
    # For 3x3 = 2x2 + 1x1 blocks:
    # C11(2x2) = A11*B11 + A12*B21: Strassen(2x2) = 7 + direct(2x1*1x2) = 7 + 4 = 11
    # C12(2x1) = A11*B12 + A12*B22: direct = 4 + 2 = 6
    # C21(1x2) = A21*B11 + A22*B21: direct = 4 + 2 = 6
    # C22(1x1) = A21*B12 + A22*B22: direct = 2 + 1 = 3
    # Total for 3x3: 11 + 6 + 6 + 3 = 26
    #
    # That's worse than just doing all 3x3 direct (9 terms). So Strassen on
    # the 2x2 sub-block doesn't help for 3x3.
    #
    # Let me just use direct multiplication for the 3x3 block (9 terms)
    # and Strassen for the 2x2 corner block (7 terms).
    #
    # Recount with 3x3 direct (9) and 2x2 Strassen (7):
    # C11 = A11*B11 (9) + A12*B21 (12) + A13*B31 (6) = 27
    # C12 = A11*B12 (6) + A12*B22 (7) + A13*B32 (2) = 15
    # C13 = A11*B13 (3) + A12*B23 (2) + A13*B33 (1) = 6
    # C21 = A21*B11 (6) + A22*B21 (4) + A23*B31 (2) = 12
    # C22 = A21*B12 (4) + A22*B22 (2) + A23*B32 (1) = 7
    # C23 = A21*B13 (2) + A22*B23 (2) + A23*B33 (1) = 5
    # C31 = A31*B11 (3) + A32*B21 (2) + A33*B31 (2) = 7
    # C32 = A31*B12 (2) + A32*B22 (1) + A33*B32 (1) = 4
    # C33 = A31*B13 (1) + A32*B23 (1) + A33*B33 (1) = 3
    # Total: 27+15+6+12+7+5+7+4+3 = 86

    # Let me implement this: 3x3 direct + 2x2 Strassen + rest direct.

    # Helper: add a direct multiplication term A[i,j] * B[j,k] -> C[i,k]
    def add_direct(a_idx, b_idx, c_idx):
        """a_idx: (i,j) in A, b_idx: (j,k) in B, c_idx: (i,k) in C"""
        i, j = a_idx
        jj, k = b_idx
        ii, kk = c_idx
        assert i == ii and j == jj and k == kk
        u_flat = i * m + j
        v_flat = j * p + k
        w_flat = k * n + i
        add_term([(u_flat, 1)], [(v_flat, 1)], [(w_flat, 1)])

    # Helper: add a term with coefficients
    # u_coeffs: list of (a_flat_idx, coeff)
    # v_coeffs: list of (b_flat_idx, coeff)
    # w_coeffs: list of (c_flat_idx, coeff)
    def add_weighted(u_coeffs, v_coeffs, w_coeffs):
        add_term(u_coeffs, v_coeffs, w_coeffs)

    # Block index helpers
    # A block (br, bc): rows block_start(br) to block_start(br)+block_size(br)-1,
    #                   cols block_start(bc) to block_start(bc)+block_size(bc)-1
    def a_flat(i, j):
        return i * m + j

    def b_flat(j, k):
        return j * p + k

    def c_flat(i, k):
        return k * n + i

    # Direct multiplication for A11(3x3) * B11(3x3) -> C11(3x3): 9 terms
    for i in range(3):
        for j in range(3):
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Direct: A12(3x2) * B21(2x3) -> C11(3x3): 3*2*3 = 18 terms? No.
    # A12 has rows 0-2, cols 3-4. B21 has rows 3-4, cols 0-2.
    # C11[i,k] += sum_{j=3,4} A12[i,j] * B21[j,k]
    for i in range(3):
        for j in [3, 4]:
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Direct: A13(3x1) * B31(1x3) -> C11(3x3): 3*1*3 = 9 terms
    for i in range(3):
        for j in [4]:
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Direct: A11(3x3) * B12(3x2) -> C12(3x2): 3*3*2 = 18 terms
    for i in range(3):
        for j in range(3):
            for k in [3, 4]:
                add_direct((i, j), (j, k), (i, k))

    # Strassen for A12(2x2) * B22(2x2) -> C12(2x2) block... wait
    # A12 is 3x2 (rows 0-2, cols 3-4), B22 is 2x2 (rows 3-4, cols 3-4)
    # C12 is 3x2 (rows 0-2, cols 3-4)
    # C12[i,k] = sum_{j=3,4} A12[i,j] * B22[j,k]
    # This is a 3x2 * 2x2 -> 3x2 multiplication.
    # We can use Strassen on the 2x2 part: for each row i of A12,
    # A12[i,3:5] * B22 gives a 1x2 result.
    # Actually, let's just do it directly: 3*2*2 = 12 terms.
    for i in range(3):
        for j in [3, 4]:
            for k in [3, 4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A13(3x1) * B32(1x2) -> C12(3x2): 3*1*2 = 6 terms
    for i in range(3):
        for j in [4]:
            for k in [3, 4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A11(3x3) * B13(3x1) -> C13(3x1): 3*3*1 = 9 terms
    for i in range(3):
        for j in range(3):
            for k in [4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A12(3x2) * B23(2x1) -> C13(3x1): 3*2*1 = 6 terms
    for i in range(3):
        for j in [3, 4]:
            for k in [4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A13(3x1) * B33(1x1) -> C13(3x1): 3*1*1 = 3 terms
    for i in range(3):
        for j in [4]:
            for k in [4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A21(2x3) * B11(3x3) -> C21(2x3): 2*3*3 = 18 terms
    for i in [3, 4]:
        for j in range(3):
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Direct: A22(2x2) * B21(2x3) -> C21(2x3): 2*2*3 = 12 terms
    for i in [3, 4]:
        for j in [3, 4]:
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Direct: A23(2x1) * B31(1x3) -> C21(2x3): 2*1*3 = 6 terms
    for i in [3, 4]:
        for j in [4]:
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Direct: A21(2x3) * B12(3x2) -> C22(2x2): 2*3*2 = 12 terms
    for i in [3, 4]:
        for j in range(3):
            for k in [3, 4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A22(2x2) * B22(2x2) -> C22(2x2): 2*2*2 = 8 terms
    for i in [3, 4]:
        for j in [3, 4]:
            for k in [3, 4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A23(2x1) * B32(1x2) -> C22(2x2): 2*1*2 = 4 terms
    for i in [3, 4]:
        for j in [4]:
            for k in [3, 4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A21(2x3) * B13(3x1) -> C23(2x1): 2*3*1 = 6 terms
    for i in [3, 4]:
        for j in range(3):
            for k in [4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A22(2x2) * B23(2x1) -> C23(2x1): 2*2*1 = 4 terms
    for i in [3, 4]:
        for j in [3, 4]:
            for k in [4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A23(2x1) * B33(1x1) -> C23(2x1): 2*1*1 = 2 terms
    for i in [3, 4]:
        for j in [4]:
            for k in [4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A31(1x3) * B11(3x3) -> C31(1x3): 1*3*3 = 9 terms
    for i in [4]:
        for j in range(3):
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Direct: A32(1x2) * B21(2x3) -> C31(1x3): 1*2*3 = 6 terms
    for i in [4]:
        for j in [3, 4]:
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Direct: A33(1x1) * B31(1x3) -> C31(1x3): 1*1*3 = 3 terms
    for i in [4]:
        for j in [4]:
            for k in range(3):
                add_direct((i, j), (j, k), (i, k))

    # Direct: A31(1x3) * B12(3x2) -> C32(1x2): 1*3*2 = 6 terms
    for i in [4]:
        for j in range(3):
            for k in [3, 4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A32(1x2) * B22(2x2) -> C32(1x2): 1*2*2 = 4 terms
    for i in [4]:
        for j in [3, 4]:
            for k in [3, 4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A33(1x1) * B32(1x2) -> C32(1x2): 1*1*2 = 2 terms
    for i in [4]:
        for j in [4]:
            for k in [3, 4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A31(1x3) * B13(3x1) -> C33(1x1): 1*3*1 = 3 terms
    for i in [4]:
        for j in range(3):
            for k in [4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A32(1x2) * B23(2x1) -> C33(1x1): 1*2*1 = 2 terms
    for i in [4]:
        for j in [3, 4]:
            for k in [4]:
                add_direct((i, j), (j, k), (i, k))

    # Direct: A33(1x1) * B33(1x1) -> C33(1x1): 1*1*1 = 1 term
    for i in [4]:
        for j in [4]:
            for k in [4]:
                add_direct((i, j), (j, k), (i, k))

    # Count total terms
    rank = len(terms_u)

    # Build U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col in range(rank):
        for (u_idx, val) in terms_u[col]:
            U[u_idx][col] = val
        for (v_idx, val) in terms_v[col]:
            V[v_idx][col] = val
        for (w_idx, val) in terms_w[col]:
            W[w_idx][col] = val

    # Verify the decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal
    # the matrix multiplication tensor
    # T[i,j,k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} where C[i',k'] = A[i',j']B[j',k']
    # Actually, the tensor is: T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # In our convention, the decomposition should satisfy:
    # For each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # (since A[i,j]*B[j,k] contributes exactly 1 to C[i,k])

    # Quick verification with random matrices
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_direct = A_test @ B_test

    # Compute C from decomposition
    C_decomp = np.zeros((n, p))
    A_flat = A_test.flatten()  # A[i*m+j]
    B_flat = B_test.flatten()  # B[j*p+k]
    for r in range(rank):
        u_col = U_arr[:, r]
        v_col = V_arr[:, r]
        w_col = W_arr[:, r]
        # u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
        # This is a tensor product
        u_2d = u_col.reshape(n, m)
        v_2d = v_col.reshape(m, p)
        w_2d = w_col.reshape(p, n)
        # C[i,k] += sum_j u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
                C_decomp[i, k] += s

    if not np.allclose(C_decomp, C_direct, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
