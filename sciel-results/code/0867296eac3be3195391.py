import time
import numpy as np
import random

DIMENSIONS = [4, 4, 7]
WALL_DEADLINE = 145.0
SAFETY_MARGIN = 5.0


def construct():
    start_time = time.time()
    n, m, p = 4, 4, 7

    def elapsed():
        return time.time() - start_time

    def check_valid(U, V, W, tol=1e-8):
        r = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def build_schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W

    def build_strassen_2x2_2x2():
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        # p1 = (a11+a22)(b11+b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        # p2 = (a21+a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[3][1] = -1
        # p3 = a11(b12-b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[2][2] = 1; W[3][2] = 1
        # p4 = a22(b21-b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[1][3] = 1
        # p5 = (a11+a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[2][4] = 1
        # p6 = (a21-a11)(b11+b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # p7 = (a12-a22)(b21+b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        return U, V, W

    def kronecker_decompose_4x4x4():
        U2, V2, W2 = build_strassen_2x2_2x2()
        U4 = [[0] * 49 for _ in range(16)]
        V4 = [[0] * 49 for _ in range(16)]
        W4 = [[0] * 49 for _ in range(16)]
        for r1 in range(7):
            for r2 in range(7):
                r = r1 * 7 + r2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1 * 2 + j1][r1] * U2[i2 * 2 + j2][r2]
                                if val != 0:
                                    U4[i * 4 + j][r] += val
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            for k2 in range(2):
                                j = j1 * 2 + j2
                                k = k1 * 2 + k2
                                val = V2[j1 * 2 + k1][r1] * V2[j2 * 2 + k2][r2]
                                if val != 0:
                                    V4[j * 4 + k][r] += val
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1 * 2 + i1][r1] * W2[k2 * 2 + i2][r2]
                                if val != 0:
                                    W4[k * 4 + i][r] += val
        return U4, V4, W4

    def build_strassen_4x4x2():
        """
        Build a rank-28 decomposition for 4x4 * 4x2 using Kronecker product
        of two 2x2x1 Strassen-like decompositions.
        Actually, we use the Kronecker product of 2x2x2 Strassen with itself,
        but only for the first 2 columns of the 4x2 output.
        
        The 4x4 * 4x2 multiplication can be seen as:
        - Split the 4x2 output into two 2x2 blocks
        - Use Strassen for each 2x2 * 2x1 block? No, that's not right.
        
        Better approach: Use Kronecker product structure.
        A 4x4 matrix A can be written as [[A11, A12], [A21, A22]] where each is 2x2
        A 4x2 matrix B can be written as [[B1], [B2]] where each is 2x1
        
        Then C = A*B gives a 4x2 matrix:
        C11 = A11*B1 + A12*B2
        C21 = A21*B1 + A22*B2
        
        Each Ci1 is 2x1. We can use Strassen-like tricks for 2x2 * 2x1.
        Actually, 2x2 * 2x1 is just 4 scalar multiplications, no Strassen benefit.
        
        Alternative: Use the full Kronecker product of 2x2x2 Strassen.
        For 4x4 * 4x4, we use rank 49 (7x7).
        For 4x4 * 4x2, we can use a subset.
        
        Let me think differently. The cell finding says:
        "Replacing the 4x4x3 schoolbook (rank 48) with a Strassen-based 4x4x2 (rank 28) + schoolbook 4x4x1 (rank 16) decomposition reduces total rank from 97 to 93"
        
        So we need a rank-28 decomposition for 4x4 * 4x2.
        
        One approach: Use the Kronecker product of 2x2x2 Strassen (rank 7) with a 2x1x1 identity (rank 2)?
        That gives rank 14, but we need to handle the structure.
        
        Actually, let's use: 4x4 * 4x2 = (2x2 ⊗ 2x2) * (2x2 ⊗ 2x1)
        Using Kronecker product of Strassen(2x2x2, rank 7) and identity(2x1x1, rank 2)?
        No, that doesn't work directly.
        
        Let me try a different approach: treat 4x4 * 4x2 as two independent 2x2 * 2x1 problems plus cross terms.
        
        Actually, the simplest approach: use the Kronecker product of the 2x2x2 Strassen decomposition
        with a 2x1 identity, giving rank 7*2 = 14 for the "block diagonal" part,
        but we also need the off-diagonal blocks.
        
        Let me just implement it as: 
        - Use Strassen 2x2x2 for the top-left 2x2 * 2x2 block (but B is 4x2, not 4x4)
        
        I think the intended approach is:
        Split A (4x4) into 2x2 blocks: A = [[A11, A12], [A21, A22]]
        Split B (4x2) into 2x1 blocks: B = [B1; B2] where B1, B2 are 2x1
        
        Then:
        C_top = A11*B1 + A12*B2  (2x1)
        C_bot = A21*B1 + A22*B2  (2x1)
        
        Each term like A11*B1 is 2x2 * 2x1 = 2x1, which requires 4 scalar mults.
        Total: 4 terms * 4 = 16 scalar mults for schoolbook.
        
        For Strassen-like: we can use the fact that we're multiplying 2x2 by 2x1.
        Let A11 = [[a, b], [c, d]], B1 = [x, y]^T
        A11*B1 = [ax+by, cx+dy]^T
        
        We can compute this with 4 mults, or can we do better?
        Actually, for 2x2 * 2x1, the minimum rank is 4 (no Strassen benefit for rectangular).
        
        So the total for 4x4 * 4x2 schoolbook is 16, not 28.
        
        Wait, the cell finding says rank 28 for 4x4x2. That's more than schoolbook (16).
        That doesn't make sense unless they mean something different.
        
        Let me re-read: "Strassen-based 4x4x2 (rank 28)"
        
        Maybe they mean: use the Kronecker product of 2x2x2 Strassen (rank 7) with 2x2x1?
        No...
        
        Actually, I think I misread. Let me reconsider.
        
        Perhaps the approach is:
        - For the first 2 columns of the 4x7 output, use a 4x4 * 4x2 decomposition
        - The 4x4 * 4x2 can be decomposed using Kronecker product of 2x2x2 Strassen
          with a 2x1 "identity" in some sense
          
        Kronecker product: (A⊗B) * (C⊗D) = (AC)⊗(BD)
        
        If we write 4x4 = 2x2 ⊗ 2x2 and 4x2 = 2x2 ⊗ 2x1, then:
        (A⊗B) * (C⊗D) where A,B,C are 2x2 and D is 2x1
        = (AC)⊗(BD)
        
        AC is 2x2, BD is 2x1.
        To compute AC, use Strassen (rank 7).
        To compute BD, use schoolbook (rank 4).
        Total rank: 7*4 = 28.
        
        Yes! That's it. The Kronecker product of Strassen(2x2x2) and schoolbook(2x2x1) gives rank 28.
        """
        U2, V2, W2 = build_strassen_2x2_2x2()
        r2 = len(U2[0])  # 7
        
        # Schoolbook 2x2 * 2x1: rank 4
        # U_sb: 4x4, V_sb: 4x4, W_sb: 2x4
        # For 2x2 * 2x1:
        # A = [[a,b],[c,d]], B = [x;y], C = [ax+by; cx+dy]
        # r0: a*x, r1: b*y, r2: c*x, r3: d*y
        U_sb = [[0]*4 for _ in range(4)]
        V_sb = [[0]*4 for _ in range(4)]
        W_sb = [[0]*4 for _ in range(2)]
        U_sb[0][0] = 1  # a
        V_sb[0][0] = 1  # x
        W_sb[0][0] = 1  # C[0]
        U_sb[1][1] = 1  # b
        V_sb[1][1] = 1  # y
        W_sb[0][1] = 1  # C[0]
        U_sb[2][2] = 1  # c
        V_sb[2][2] = 1  # x
        W_sb[1][2] = 1  # C[1]
        U_sb[3][3] = 1  # d
        V_sb[3][3] = 1  # y
        W_sb[1][3] = 1  # C[1]
        
        # Now Kronecker product: rank = 7 * 4 = 28
        r_total = r2 * 4
        U42 = [[0] * r_total for _ in range(16)]
        V42 = [[0] * r_total for _ in range(8)]
        W42 = [[0] * r_total for _ in range(8)]
        
        for r1 in range(r2):
            for r2_ in range(4):
                r = r1 * 4 + r2_
                # U: (2x2 ⊗ 2x2) indices: (i1,j1,i2,j2) -> i1*4+j1*2+i2*4+j2? No.
                # 4x4 matrix indexed as (i,j) where i = i1*2+i2, j = j1*2+j2
                # U for Kronecker: U_kron[(i1,i2),(j1,j2), (r1,r2_)] = U2[i1*2+j1, r1] * U_sb[i2*2+j2, r2_]
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1 * 2 + j1][r1] * U_sb[i2 * 2 + j2][r2_]
                                if val != 0:
                                    U42[i * 4 + j][r] += val
                # V: (2x2 ⊗ 2x2) for first factor, (2x2 ⊗ 2x1) for second
                # V_kron[(j1,j2),(k1,k2), (r1,r2_)] = V2[j1*2+k1, r1] * V_sb[j2*2+k2, r2_]
                # But V_sb is for 2x2 * 2x1, so V_sb has indices (j2, k2) where j2 in 0..1, k2 in 0..1
                # Wait, V_sb should be for the 2x2 * 2x1 multiplication.
                # In 2x2 * 2x1: A is 2x2, B is 2x1, C is 2x1
                # V_sb[j, k, r] where j in 0..1 (row of A), k in 0..1 (col of B, but B is 2x1 so k=0)
                # Actually B is 2x1, so it has 2 rows and 1 col. Index j for A rows, k for B rows.
                # V_sb[j*1+k, r] = V_sb[j+k, r]
                # Let me redefine V_sb properly.
                
        # Let me redo this more carefully.
        # For 2x2 * 2x1:
        # A has indices (i,j) with i,j in {0,1}, 4 entries
        # B has indices (j,k) with j in {0,1}, k in {0} (since B is 2x1), 2 entries
        # C has indices (i,k) with i in {0,1}, k in {0}, 2 entries
        # 
        # Schoolbook: 4 terms
        # r0: A[0,0]*B[0,0] -> C[0,0]
        # r1: A[0,1]*B[1,0] -> C[0,0]
        # r2: A[1,0]*B[0,0] -> C[1,0]
        # r3: A[1,1]*B[1,0] -> C[1,0]
        
        U_sb = [[0]*4 for _ in range(4)]  # 4x4
        V_sb = [[0]*4 for _ in range(2)]  # 2x4
        W_sb = [[0]*4 for _ in range(2)]  # 2x4
        U_sb[0][0] = 1  # A[0,0]
        V_sb[0][0] = 1  # B[0,0]
        W_sb[0][0] = 1  # C[0,0]
        U_sb[1][1] = 1  # A[0,1]
        V_sb[1][1] = 1  # B[1,0]
        W_sb[0][1] = 1  # C[0,0]
        U_sb[2][2] = 1  # A[1,0]
        V_sb[0][2] = 1  # B[0,0]
        W_sb[1][2] = 1  # C[1,0]
        U_sb[3][3] = 1  # A[1,1]
        V_sb[1][3] = 1  # B[1,0]
        W_sb[1][3] = 1  # C[1,0]
        
        # Now Kronecker product for 4x4 * 4x2:
        # A (4x4) = A1 ⊗ A2 where A1, A2 are 2x2
        # B (4x2) = B1 ⊗ B2 where B1 is 2x2, B2 is 2x1
        # C (4x2) = C1 ⊗ C2 where C1 is 2x2, C2 is 2x1
        # 
        # C = A*B = (A1⊗A2)*(B1⊗B2) = (A1*B1)⊗(A2*B2)
        # 
        # A1*B1: 2x2 * 2x2, use Strassen (rank 7)
        # A2*B2: 2x2 * 2x1, use schoolbook (rank 4)
        # 
        # Total rank: 7 * 4 = 28
        
        r_total = r2 * 4
        U42 = [[0] * r_total for _ in range(16)]
        V42 = [[0] * r_total for _ in range(8)]
        W42 = [[0] * r_total for _ in range(8)]
        
        for r1 in range(r2):
            for r2_ in range(4):
                r = r1 * 4 + r2_
                # U42[i*4+j, r] = U2[i1*2+j1, r1] * U_sb[i2*2+j2, r2_]
                # where i = i1*2+i2, j = j1*2+j2
                for i1 in range(2):
                    for j1 in range(2):
                        for i2 in range(2):
                            for j2 in range(2):
                                i = i1 * 2 + i2
                                j = j1 * 2 + j2
                                val = U2[i1 * 2 + j1][r1] * U_sb[i2 * 2 + j2][r2_]
                                if val != 0:
                                    U42[i * 4 + j][r] += val
                # V42[j*2+k, r] = V2[j1*2+k1, r1] * V_sb[j2*2+k2, r2_]
                # where j = j1*2+j2, k = k1*2+k2, but B is 4x2 so k in 0..1
                # B1 is 2x2: indices (j1, k1), B2 is 2x1: indices (j2, k2) where k2=0
                # So B index: j = j1*2+j2, k = k1 (since k2 is always 0 for B2)
                # Wait, B is 4x2. If B = B1 ⊗ B2, B1 is 2x2, B2 is 2x1
                # Then B has 4 rows and 2 cols.
                # Row index: (j1, j2) -> j = j1*2+j2
                # Col index: (k1, k2) -> k = k1*2+k2, but B2 is 2x1 so k2=0, k=k1
                # So V42[j*2+k, r] where j = j1*2+j2, k = k1
                # V42[(j1*2+j2)*2 + k1, r] = V2[j1*2+k1, r1] * V_sb[j2*1+0, r2_]
                # V_sb has 2 rows (for B2's 2 rows) and 4 cols
                # V_sb[j2, r2_]
                for j1 in range(2):
                    for k1 in range(2):
                        for j2 in range(2):
                            j = j1 * 2 + j2
                            k = k1
                            val = V2[j1 * 2 + k1][r1] * V_sb[j2][r2_]
                            if val != 0:
                                V42[j * 2 + k][r] += val
                # W42[k*4+i, r] = W2[k1*2+i1, r1] * W_sb[k2*1+i2, r2_]
                # where k = k1*2+k2, i = i1*2+i2
                # C is 4x2. C = C1 ⊗ C2, C1 is 2x2, C2 is 2x1
                # Row index: (k1, k2) -> k = k1*2+k2
                # Col index: (i1, i2) -> i = i1*2+i2
                # W42[k*2+i, r] where k = k1*2+k2, i = i1*2+i2
                # W42[(k1*2+k2)*2 + i1*2+i2, r] = W2[k1*2+i1, r1] * W_sb[k2*1+i2, r2_]
                # W_sb has 2 rows and 4 cols: W_sb[k2, r2_]
                for k1 in range(2):
                    for i1 in range(2):
                        for k2 in range(2):
                            for i2 in range(2):
                                k = k1 * 2 + k2
                                i = i1 * 2 + i2
                                val = W2[k1 * 2 + i1][r1] * W_sb[k2][r2_]
                                if val != 0:
                                    W42[k * 2 + i][r] += val
        
        return U42, V42, W42

    def build_4x4x7_decomposition():
        # Use Strassen-based 4x4x2 (rank 28) + schoolbook 4x4x1 (rank 16) = rank 44
        U42, V42, W42 = build_strassen_4x4x2()
        r42 = len(U42[0])  # 28
        
        r_total = r42 + 16
        U = [[0] * r_total for _ in range(16)]
        V = [[0] * r_total for _ in range(28)]
        W = [[0] * r_total for _ in range(28)]
        
        for i in range(16):
            for r in range(r42):
                U[i][r] = U42[i][r]
        for j in range(4):
            for k in range(2):
                for r in range(r42):
                    V[j * 7 + k][r] = V42[j * 2 + k][r]
        for k in range(2):
            for i in range(4):
                for r in range(r42):
                    W[k * 4 + i][r] = W42[k * 2 + i][r]
        
        col_offset = r42
        for i in range(4):
            for j in range(4):
                for k in range(2, 7):
                    U[i * 4 + j][col_offset] = 1
                    V[j * 7 + k][col_offset] = 1
                    W[k * 4 + i][col_offset] = 1
                    col_offset += 1
        return U, V, W

    # ---- Refinement machinery: random linear-combination rank reduction ----
    def tensors_to_vec(U, V, W):
        """Flatten the three factor matrices into one vector (U, V, W)."""
        r = len(U[0])
        u = np.array(U, dtype=float).flatten()
        v = np.array(V, dtype=float).flatten()
        w = np.array(W, dtype=float).flatten()
        return np.concatenate([u, v, w])

    def vec_to_tensors(x, r):
        u_len = n * m * r
        v_len = m * p * r
        U = x[:u_len].reshape(n * m, r)
        V = x[u_len:u_len + v_len].reshape(m * p, r)
        W = x[u_len + v_len:].reshape(p * n, r)
        return U, V, W

    def build_jacobian(U, V, W):
        """
        Build the Jacobian of the residual vector w.r.t. the parameter vector x=(U,V,W).
        Residual for (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1 = 0
        d/dU[i*m+j,c] = V[j*p+k,c]*W[k*n+i,c]
        d/dV[j*p+k,c] = U[i*m+j,c]*W[k*n+i,c]
        d/dW[k*n+i,c] = U[i*m+j,c]*V[j*p+k,c]
        """
        r = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        u_len = n * m * r
        v_len = m * p * r
        J = np.zeros((n * m * p, u_len + v_len + p * n * r))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = (i * m + j) * p + k
                    for c in range(r):
                        uv = V[j * p + k, c] * W[k * n + i, c]
                        vw = U[i * m + j, c] * W[k * n + i, c]
                        wu = U[i * m + j, c] * V[j * p + k, c]
                        J[row, i * m * r + j * r + c] = uv
                        J[row, u_len + j * p * r + k * r + c] = vw
                        J[row, u_len + v_len + k * n * r + i * r + c] = wu
        return J

    def residual_vec(U, V, W):
        r = len(U[0])
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        res = np.zeros(n * m * p)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    res[(i * m + j) * p + k] = s - 1.0
        return res

    def random_merge(U, V, W, n_merge, rng, tol=1e-9):
        """
        Merge n_merge pairs of columns via a random linear combination, then
        solve a least-squares correction to restore validity.
        Returns (new_U, new_V, new_W, new_rank) or None on failure.
        """
        r = len(U[0])
        U = np.array(U, dtype=float).copy()
        V = np.array(V, dtype=float).copy()
        W = np.array(W, dtype=float).copy()

        # Build the linear-combination matrix M (new_r x r)
        new_r = r - n_merge
        # Pick n_merge pairs of columns to merge: for each merge, combine col a into col b
        cols = list(range(r))
        rng.shuffle(cols)
        # We'll define the kept columns and the dropped columns
        # kept = first new_r of the shuffled, dropped = rest
        kept = cols[:new_r]
        dropped = cols[new_r:]
        # M: each kept column = 1 (identity) plus a random combination of dropped columns
        # Actually simplest: new column t = kept[t] + sum_d alpha_{t,d} * dropped[d]
        # This is a linear transform x_new = M x where M is (new_r x r)
        M = np.zeros((new_r, r))
        for t, kc in enumerate(kept):
            M[t, kc] = 1.0
            for dc in dropped:
                M[t, dc] = rng.uniform(-1.0, 1.0)

        U_new = U @ M.T
        V_new = V @ M.T
        W_new = W @ M.T

        # Now solve for correction delta to make residual zero:
        # residual(x + delta) ≈ residual(x) + J(x) delta = 0
        # => J delta = -residual(x)
        res = residual_vec(U_new, V_new, W_new)
        J = build_jacobian(U_new, V_new, W_new)
        try:
            delta, *_ = np.linalg.lstsq(J, -res, rcond=None)
        except Exception:
            return None
        x_new = tensors_to_vec(U_new, V_new, W_new) + delta
        U2, V2, W2 = vec_to_tensors(x_new, new_r)

        # Verify
        if not check_valid(U2.tolist(), V2.tolist(), W2.tolist(), tol=1e-8):
            return None
        return U2.tolist(), V2.tolist(), W2.tolist(), new_r

    # ---- Main flow ----
    U_school, V_school, W_school = build_schoolbook()
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p

    rng = random.Random(12345)

    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        try:
            U_kron, V_kron, W_kron = build_4x4x7_decomposition()
            r_kron = len(U_kron[0])
            if check_valid(U_kron, V_kron, W_kron):
                if r_kron < best_rank:
                    best_result = {"U": U_kron, "V": V_kron, "W": W_kron}
                    best_rank = r_kron
        except Exception:
            pass

    # Refinement: iteratively merge columns on the best valid decomposition
    if elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
        cur_U = np.array(best_result["U"], dtype=float)
        cur_V = np.array(best_result["V"], dtype=float)
        cur_W = np.array(best_result["W"], dtype=float)
        cur_rank = len(cur_U[0])

        # Try decreasing merge counts; each successful merge keeps the result
        # We attempt to merge 1 column at a time, up to a budget of attempts.
        max_attempts = 200
        attempts = 0
        # We'll try to reduce rank from cur_rank down toward a target
        target = 30  # aim well below schoolbook; stop if we hit this
        while cur_rank > target and attempts < max_attempts and elapsed() < WALL_DEADLINE - SAFETY_MARGIN:
            # Try to merge a few columns at once; if it fails, try 1
            merged = False
            for n_merge in (2, 1):
                if cur_rank - n_merge < 1:
                    continue
                res = random_merge(cur_U, cur_V, cur_W, n_merge, rng, tol=1e-9)
                attempts += 1
                if res is not None:
                    cur_U, cur_V, cur_W, cur_rank = res
                    merged = True
                    if cur_rank < best_rank:
                        best_rank = cur_rank
                        best_result = {
                            "U": cur_U.tolist(),
                            "V": cur_V.tolist(),
                            "W": cur_W.tolist(),
                        }
                    break
                if elapsed() > WALL_DEADLINE - SAFETY_MARGIN:
                    break
            if not merged:
                # Give up reducing further
                break

    return best_result
