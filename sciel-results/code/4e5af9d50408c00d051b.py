import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    start = time.time()
    DEADLINE = 110.0  # Safety margin before 120s limit

    n, m, p = DIMENSIONS

    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best = (rank_sb, U_sb, V_sb, W_sb)

    def verify(U, V, W):
        """Check that sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = delta_ij * delta_jk is not right.
        Actually the tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] should equal 1 for all i,j,k.
        The multiplication tensor T_{n,m,p}(i,j,k) = 1 for all (i,j,k) in [n]x[m]x[p].
        """
        Um = np.array(U, dtype=float)
        Vm = np.array(V, dtype=float)
        Wm = np.array(W, dtype=float)
        # Reshape
        Um = Um.reshape(n, m, -1)
        Vm = Vm.reshape(m, p, -1)
        Wm = Wm.reshape(p, n, -1)
        # Compute tensor: T[i,j,k] = sum_c Um[i,j,c] * Vm[j,k,c] * Wm[k,i,c]
        # Efficient way: for each c, outer product
        # T = np.zeros((n,m,p))
        # for c in range(r):
        #     T += Um[:,:,c] * ... but this isn't a simple outer product
        # Actually T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Let's compute via einsum or explicit
        T = np.zeros((n, m, p))
        r = Um.shape[2]
        for c in range(r):
            uc = Um[:, :, c]  # (n, m)
            vc = Vm[:, :, c]  # (m, p)
            wc = Wm[:, :, c]  # (p, n)
            # For each i,j,k: uc[i,j] * vc[j,k] * wc[k,i]
            # This is a 3D outer product with contraction
            # T[i,j,k] += uc[i,j] * vc[j,k] * wc[k,i]
            # Can do: for j in range(m): T[:,j,:] += uc[:,j][:,None] * vc[j,:,None] * wc.T[None,:,None]
            # wc[k,i] -> wc.T[i,k]
            wcT = wc.T  # (n, p), wcT[i,k] = wc[k,i]
            for j in range(m):
                T[:, j, :] += uc[:, j][:, None] * vc[j, :, None] * wcT[None, :, :]
        # Should equal 1 everywhere
        return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

    # Try Strassen-based decomposition for 2x4x6
    # Decompose into 6 blocks of 2x2x2
    # A is 2x4, B is 4x6, C is 2x6
    # Split A into A1 (2x2), A2 (2x2); B into B1 (2x2), B2 (2x2), B3 (2x2); C into C1 (2x2), C2 (2x2), C3 (2x2)
    # C1 = A1*B1 + A2*B2 ... wait no.
    # Actually: A = [A1 | A2], B = [B1; B2; B3] (each 2x2)
    # C = A1*B1 + A2*B2 + ... no, that's not right either.
    # A is 2x4, B is 4x6.
    # A = [A_left | A_right] where A_left is 2x2, A_right is 2x2
    # B = [B_top; B_mid; B_bot] where each is 2x2
    # C = A_left * B_top + A_right * B_mid ... no.
    # C[i,k] = sum_j A[i,j] B[j,k]
    # j ranges 0..3, k ranges 0..5.
    # Let's split j into {0,1} and {2,3}.
    # C = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:]
    # That's just 2 terms, not helpful.
    # Better: use Strassen on 2x2 blocks.
    # Let A1 = A[:,0:2], A2 = A[:,2:4] (both 2x2)
    # Let B1 = B[0:2,:], B2 = B[2:4,:] (both 2x6)
    # C = A1*B1 + A2*B2
    # Now split each 2x6 product into 2x2x2 subproducts.
    # B1 = [B11 | B12 | B13], B2 = [B21 | B22 | B23] (each 2x2)
    # A1*B1 = A1*B11 + A1*B12 + A1*B13 (3 subproducts of 2x2x2)
    # A2*B2 = A2*B21 + A2*B22 + A2*B23 (3 subproducts of 2x2x2)
    # Total 6 subproducts, each can be done with Strassen (rank 7).
    # Total rank = 6*7 = 42.

    # Let's construct this decomposition.
    # Strassen 2x2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = AB
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e), M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # U vectors (for A, flattened as [a,b,c,d]):
    # M1: (a+d)(e+h) -> U=[1,0,0,1], V=[1,0,0,1]
    # M2: (c+d)e -> U=[0,0,1,1], V=[1,0,0,0]
    # M3: a(f-h) -> U=[1,0,0,0], V=[0,1,0,-1]
    # M4: d(g-e) -> U=[0,0,0,1], V=[-1,0,1,0]
    # M5: (a+b)h -> U=[1,1,0,0], V=[0,0,0,1]
    # M6: (c-a)(e+f) -> U=[-1,0,1,0], V=[1,1,0,0]
    # M7: (b-d)(g+h) -> U=[0,1,0,-1], V=[0,1,1,1]

    # W vectors (for C, flattened as [C00,C01,C10,C11]):
    # M1: [1,0,0,1]
    # M2: [0,0,1,-1]
    # M3: [0,1,0,1]
    # M4: [1,0,1,0]
    # M5: [-1,1,0,0]
    # M6: [0,0,0,1]
    # M7: [1,0,0,1]

    U_strassen = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,1,0,-1]
    ]
    V_strassen = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [-1,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,1,1,1]
    ]
    W_strassen = [
        [1,0,0,1],
        [0,0,1,-1],
        [0,1,0,1],
        [1,0,1,0],
        [-1,1,0,0],
        [0,0,0,1],
        [1,0,0,1]
    ]

    # Now build the full 2x4x6 decomposition with 42 rank-1 terms.
    rank_s = 42
    U_s = [[0] * rank_s for _ in range(8)]  # n*m = 2*4 = 8
    V_s = [[0] * rank_s for _ in range(24)]  # m*p = 4*6 = 24
    W_s = [[0] * rank_s for _ in range(12)]  # p*n = 6*2 = 12

    # 6 subproblems: (A_part, B_part) -> C_part
    # A1 = A[:,0:2] (rows 0,1; cols 0,1 in A), A2 = A[:,2:4] (rows 0,1; cols 2,3 in A)
    # B1 = B[0:2,:] (rows 0,1; cols 0,1,2,3,4,5), B2 = B[2:4,:] (rows 2,3; cols 0,1,2,3,4,5)
    # C1 = A1*B1 (2x6), C2 = A2*B2 (2x6)
    # C = C1 + C2

    # Split B1 into B11 (2x2, cols 0,1), B12 (2x2, cols 2,3), B13 (2x2, cols 4,5)
    # Split B2 into B21 (2x2, cols 0,1), B22 (2x2, cols 2,3), B23 (2x2, cols 4,5)

    # Subproblem 1: A1 * B11 -> C11 (cols 0,1 of C1)
    # Subproblem 2: A1 * B12 -> C12 (cols 2,3 of C1)
    # Subproblem 3: A1 * B13 -> C13 (cols 4,5 of C1)
    # Subproblem 4: A2 * B21 -> C21 (cols 0,1 of C2)
    # Subproblem 5: A2 * B22 -> C22 (cols 2,3 of C2)
    # Subproblem 6: A2 * B23 -> C23 (cols 4,5 of C2)

    # For each subproblem, we add 7 rank-1 terms.
    # A1 indices in U: rows 0,1 (i=0,1), cols 0,1 (j=0,1) -> U[i*4 + j] for i in [0,1], j in [0,1]
    #   So U[0] = A[0,0], U[1] = A[0,1], U[2] = A[1,0], U[3] = A[1,1]
    # A2 indices in U: rows 0,1, cols 2,3 -> U[i*4 + j] for i in [0,1], j in [2,3]
    #   So U[4] = A[0,2], U[5] = A[0,3], U[6] = A[1,2], U[7] = A[1,3]

    # B11 indices in V: rows 0,1 (j=0,1), cols 0,1 (k=0,1) -> V[j*6 + k] for j in [0,1], k in [0,1]
    #   V[0]=B[0,0], V[1]=B[0,1], V[6]=B[1,0], V[7]=B[1,1]
    # B12: j in [0,1], k in [2,3] -> V[j*6+k]
    # B13: j in [0,1], k in [4,5]
    # B21: j in [2,3], k in [0,1]
    # B22: j in [2,3], k in [2,3]
    # B23: j in [2,3], k in [4,5]

    # C11 indices in W: rows 0,1 (k=0,1), cols 0,1 (i=0,1) -> W[k*2 + i] for k in [0,1], i in [0,1]
    #   W[0]=C[0,0], W[1]=C[0,1], W[2]=C[1,0], W[3]=C[1,1]
    # C12: k in [2,3]
    # C13: k in [4,5]
    # C21: same k as C11, but it's added to the same C entries
    # C22: same k as C12
    # C23: same k as C13

    subproblems = [
        # (a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_col_offset)
        (0, 0, 0, 0, 0),  # A1*B11 -> C cols 0,1
        (0, 0, 0, 2, 2),  # A1*B12 -> C cols 2,3
        (0, 0, 0, 4, 4),  # A1*B13 -> C cols 4,5
        (0, 2, 2, 0, 0),  # A2*B21 -> C cols 0,1
        (0, 2, 2, 2, 2),  # A2*B22 -> C cols 2,3
        (0, 2, 2, 4, 4),  # A2*B23 -> C cols 4,5
    ]

    for sp_idx, (a_i0, a_j0, b_j0, b_k0, c_k0) in enumerate(subproblems):
        for t in range(7):
            c = sp_idx * 7 + t
            u_vals = U_strassen[t]  # [u0, u1, u2, u3] for [a,b,c,d]
            v_vals = V_strassen[t]  # [v0, v1, v2, v3] for [e,f,g,h]
            w_vals = W_strassen[t]  # [w0, w1, w2, w3] for [C00,C01,C10,C11]

            # A part: local indices (i_local, j_local) -> global (a_i0 + i, a_j0 + j)
            # u_vals[0] -> (i=0, j=0), u_vals[1] -> (i=0, j=1), u_vals[2] -> (i=1, j=0), u_vals[3] -> (i=1, j=1)
            for li, lj in [(0,0), (0,1), (1,0), (1,1)]:
                idx = li * 4 + (a_j0 + lj)
                U_s[idx][c] = u_vals[li * 2 + lj]

            # B part: local indices (j_local, k_local) -> global (b_j0 + j, b_k0 + k)
            for lj, lk in [(0,0), (0,1), (1,0), (1,1)]:
                idx = (b_j0 + lj) * 6 + (b_k0 + lk)
                V_s[idx][c] = v_vals[lj * 2 + lk]

            # C part: local indices (k_local, i_local) -> global (c_k0 + k, i)
            # w_vals[0] -> C00 (i=0, k_local=0), w_vals[1] -> C01 (i=0, k_local=1)
            # w_vals[2] -> C10 (i=1, k_local=0), w_vals[3] -> C11 (i=1, k_local=1)
            for lk, li in [(0,0), (1,0), (0,1), (1,1)]:
                # w_vals index: for (i_local, k_local): C_{i_local, k_local}
                # w_vals[0] = C_{0,0}, w_vals[1] = C_{0,1}, w_vals[2] = C_{1,0}, w_vals[3] = C_{1,1}
                # So for (li, lk): idx_w = li * 2 + lk
                idx = (c_k0 + lk) * 2 + li
                W_s[idx][c] = w_vals[li * 2 + lk]

    if time.time() < DEADLINE:
        if verify(U_s, V_s, W_s):
            best = (rank_s, U_s, V_s, W_s)

    rank_best, U_best, V_best, W_best = best
    return {"U": U_best, "V": V_best, "W": W_best}
