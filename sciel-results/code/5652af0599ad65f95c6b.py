import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    deadline = 140.0
    start = time.time()

    def build_T(n, m, p):
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    def check(U, V, W, T):
        r = U.shape[1]
        T_rec = np.zeros(T.shape)
        for c in range(r):
            T_rec += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        return float(np.max(np.abs(T_rec - T)))

    # ---------- Strassen 2x2x2 block ----------
    # Returns U,V,W for multiplying a 2x2 by a 2x2
    # U index: i*2+j ; V index: j*2+k ; W index: k*2+i
    def strassen_2x2x2():
        r = 7
        U = np.zeros((4, r))
        V = np.zeros((4, r))
        W = np.zeros((4, r))
        # Strassen's 7 products for 2x2 matrix multiplication
        # M1 = (A11-A22)(B21+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B11-B22)
        # M4 = A22(B22-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        #
        # U[i,j] encodes coefficient of A[i,j] in each product
        # V[j,k] encodes coefficient of B[j,k] in each product
        # W[k,i] encodes coefficient in C[i,k] from each product
        #
        # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=2, A[1,1]=3
        # B indices: B[0,0]=0, B[0,1]=1, B[1,0]=2, B[1,1]=3
        # C indices: C[0,0]=0, C[0,1]=1, C[1,0]=2, C[1,1]=3
        # W[k*2+i]: k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3

        # M1: (A00 - A11)(B10 + B11)
        U[0, 0] = 1; U[3, 0] = -1
        V[2, 0] = 1; V[3, 0] = 1
        # M2: (A10 + A11) B00
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        # M3: A00 (B00 - B11)
        U[0, 2] = 1
        V[0, 2] = 1; V[3, 2] = -1
        # M4: A11 (B11 - B00)
        U[3, 3] = 1
        V[3, 3] = 1; V[0, 3] = -1
        # M5: (A00 + A01) B11
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        # M6: (A10 - A00)(B00 + B01)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        # M7: (A01 - A11)(B10 + B11)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1

        # C11 = M1 + M4 - M5 + M7  -> W[0, c] for c in 0..6
        W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
        # C12 = M3 + M5
        W[1, 2] = 1; W[1, 4] = 1
        # C21 = M2 + M4
        W[2, 1] = 1; W[2, 3] = 1
        # C22 = M1 - M2 + M3 + M6
        W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1

        return U, V, W

    # ---------- Build 2x3 x 3x9 using block structure ----------
    # A is 2x3: split into A_L (2x2) and A_R (2x1)
    # B is 3x9: split rows into B_T (2x9) and B_B (1x9)
    # C = A_L * B_T + A_R * B_B  (2x2 + 2x1)
    # B_T split into 2x(9) = 2x(3*3), split columns into 3 blocks of width 3
    # B_B is 1x9, split into 3 blocks of width 3

    # For A_L (2x2) * B_T (2x9):
    #   Split B_T into 3 blocks of 2x3: B_T0, B_T1, B_T2
    #   A_L * B_Tq is 2x3. Split A_L into 2x2 (itself) and B_Tq into 2x2 + 2x1
    #   Actually let's split differently for efficiency.
    #
    # Alternative: A is 2x3, B is 3x9.
    # Split the inner dimension (3) into 2+1:
    #   A = [A_2 | A_1] where A_2 is 2x2, A_1 is 2x1
    #   B = [B_2; B_1] where B_2 is 2x9, B_1 is 1x9
    #   C = A_2 * B_2 + A_1 * B_1
    #
    # A_2 * B_2: 2x2 * 2x9. Split B_2 into 3 blocks of 2x3.
    #   For each 2x3 block B_2q: split into 2x2 + 2x1.
    #   A_2 * B_2q_left (2x2*2x2): Strassen rank 7
    #   A_2 * B_2q_right (2x2*2x1): rank 4
    #   So per q-block: 7+4 = 11. Three blocks: 33.
    #
    # A_1 * B_1: 2x1 * 1x9. This is outer-product-like.
    #   (a0, a1)^T * (b0,...,b8) = a0*(b0,...,b8) row + a1*(b0,...,b8) row
    #   Rank: 2 (one per row of A_1, since B_1 is a single row)
    #   Actually: C[i,k] = A_1[i,0] * B_1[0,k]. This is an outer product of a 2x1 and 1x9.
    #   Rank = 2 (since both dimensions > 1 but one is 1, rank = min(2*1, 1*9) = 1? No.)
    #   C[i,k] = u_i * v_k where u = A_1 column, v = B_1 row. Rank = 1.
    #   So rank 1 for this block.

    # Total Strassen-based rank: 33 + 1 = 34

    def build_strassen_decomposition():
        U = np.zeros((n * m, 34))  # 6 x 34
        V = np.zeros((m * p, 34))  # 27 x 34
        W = np.zeros((p * n, 34))  # 18 x 34

        col = 0
        # For each of the 3 q-blocks of B_2 (each 2x3):
        for q in range(3):
            # B_2q is rows 0,1 of B, columns 3q..3q+2
            # Split B_2q into left (2x2, cols 3q, 3q+1) and right (2x1, col 3q+2)

            # --- Strassen for A_2 (2x2) * B_2q_left (2x2) ---
            U_s, V_s, W_s = strassen_2x2x2()
            # U_s: 4x7, index i*2+j for A_2 (i=0,1; j=0,1)
            # V_s: 4x7, index j*2+k for B_2q_left (j=0,1; k=0,1)
            # W_s: 4x7, index k*2+i for C (k=0,1; i=0,1)

            # A_2 indices in A (2x3): A[i, j] for i in 0,1; j in 0,1
            # U total index: i*m + j = i*3 + j
            # V total index: j*p + k. For B_2q_left: j in 0,1; k in 3q, 3q+1
            #   V index: j*9 + (3q + k')
            # W total index: k*p + i... wait, W index is k*n + i where k is column of B = row of C
            #   For C (2x9): C[i, 3q + k'] where i in 0,1; k' in 0,1
            #   W index: (3q + k')*2 + i

            for c in range(7):
                # U
                for idx in range(4):
                    ui, uj = divmod(idx, 2)
                    U[ui * 3 + uj, col + c] = U_s[idx, c]
                # V
                for idx in range(4):
                    vj, vk = divmod(idx, 2)
                    V[vj * 9 + (3 * q + vk), col + c] = V_s[idx, c]
                # W
                for idx in range(4):
                    wk, wi = divmod(idx, 2)
                    W[(3 * q + wk) * 2 + wi, col + c] = W_s[idx, c]
            col += 7

            # --- Schoolbook for A_2 (2x2) * B_2q_right (2x1) ---
            # B_2q_right: column 3q+2 of B, rows 0,1
            # C contribution: C[i, 3q+2] = sum_j A_2[i,j] * B[3q+2... wait
            # A_2[i,j] for j in 0,1; B[j, 3q+2] for j in 0,1
            # C[i, 3q+2] = A[i,0]*B[0, 3q+2] + A[i,1]*B[1, 3q+2]
            # Rank 4 (2x2)
            for i in range(2):
                for j in range(2):
                    U[i * 3 + j, col] = 1
                    V[j * 9 + (3 * q + 2), col] = 1
                    W[(3 * q + 2) * 2 + i, col] = 1
                    col += 1

        # --- A_1 (2x1) * B_1 (1x9): outer product, rank 1 ---
        # A_1[i,0] for i in 0,1 -> A[i, 2]
        # B_1[0,k] for k in 0..8 -> B[2, k]
        # C[i, k] += A[i,2] * B[2,k]
        for i in range(2):
            U[i * 3 + 2, col] = 1
        for k in range(9):
            V[2 * 9 + k, col] = 1
        for i in range(2):
            for k in range(9):
                W[k * 2 + i, col] = 1
        col += 1

        assert col == 34, f"Expected 34, got {col}"
        return U, V, W

    # Build the true tensor
    T = build_T(n, m, p)

    # Verify Strassen-based decomposition
    U_s, V_s, W_s = build_strassen_decomposition()
    err = check(U_s, V_s, W_s, T)
    if err < 1e-10:
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    # Fallback: schoolbook
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
