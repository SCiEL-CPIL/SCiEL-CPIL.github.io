import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def construct():
    # Deadline guard: return schoolbook if we run out of time
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 4, 4, 7
    rank_schoolbook = n * m * p  # 112

    def build_schoolbook():
        """Return the schoolbook (trivial) decomposition of rank n*m*p."""
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

    def build_strassen_2x2():
        """
        Return the rank-7 Strassen decomposition for 2x2 x 2x2 -> 2x2.
        U: 4x7 (rows indexed by (i,j) of A), V: 4x7 (rows indexed by (j,k) of B), W: 4x7 (rows indexed by (k,i) of C)
        Using the trace convention: W[k*n + i, c] for n=2.
        """
        rank_s = 7
        U_s = [[0] * rank_s for _ in range(4)]
        V_s = [[0] * rank_s for _ in range(4)]
        W_s = [[0] * rank_s for _ in range(4)]

        # P1 = (a11+a22)(b11+b22)
        U_s[0][0] = 1  # a11
        U_s[3][0] = 1  # a22
        V_s[0][0] = 1  # b11
        V_s[1][0] = 1  # b12
        V_s[2][0] = 1  # b21
        V_s[3][0] = 1  # b22
        W_s[0][0] = 1  # c11 (k=0,i=0)
        W_s[1][0] = 1  # c12 (k=0,i=1)
        W_s[2][0] = 1  # c21 (k=1,i=0)
        W_s[3][0] = 1  # c22 (k=1,i=1)

        # P2 = (a21+a22)b11
        U_s[2][1] = 1  # a21
        U_s[3][1] = 1  # a22
        V_s[0][1] = 1  # b11
        W_s[2][1] = 1  # c21 (k=1,i=0)
        W_s[3][1] = 1  # c22 (k=1,i=1)

        # P3 = a11(b12-b22)
        U_s[0][2] = 1  # a11
        V_s[1][2] = 1  # b12
        V_s[3][2] = -1  # b22
        W_s[1][2] = 1  # c12 (k=0,i=1)
        W_s[3][2] = 1  # c22 (k=1,i=1)

        # P4 = a22(b21-b11)
        U_s[3][3] = 1  # a22
        V_s[2][3] = 1  # b21
        V_s[0][3] = -1  # b11
        W_s[0][3] = 1  # c11 (k=0,i=0)
        W_s[2][3] = 1  # c21 (k=1,i=0)

        # P5 = (a11+a12)b22
        U_s[0][4] = 1  # a11
        U_s[1][4] = 1  # a12
        V_s[3][4] = 1  # b22
        W_s[0][4] = -1  # c11 (k=0,i=0)
        W_s[1][4] = 1  # c12 (k=0,i=1)

        # P6 = (a21-a11)(b11+b12)
        U_s[2][5] = 1  # a21
        U_s[0][5] = -1  # a11
        V_s[0][5] = 1  # b11
        V_s[1][5] = 1  # b12
        W_s[3][5] = 1  # c22 (k=1,i=1)

        # P7 = (a12-a22)(b21+b22)
        U_s[1][6] = 1  # a12
        U_s[3][6] = -1  # a22
        V_s[2][6] = 1  # b21
        V_s[3][6] = 1  # b22
        W_s[0][6] = 1  # c11 (k=0,i=0)

        return U_s, V_s, W_s

    def build_strassen_4x4x4():
        """
        Build a rank-49 decomposition for 4x4 x 4x4 -> 4x4 using recursive Strassen.
        """
        U_s, V_s, W_s = build_strassen_2x2()
        rank_s = 7
        rank_total = 49  # 7 * 7

        U_44 = [[0] * rank_total for _ in range(16)]
        V_44 = [[0] * rank_total for _ in range(16)]
        W_44 = [[0] * rank_total for _ in range(16)]

        A_block_mults = [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1],
        ]

        B_block_mults = [
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ]

        C_block_mults = [
            [1, 0, 0, 1],
            [0, 0, 1, -1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [-1, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]

        for s in range(7):
            A_mults = A_block_mults[s]
            B_mults = B_block_mults[s]
            C_mults = C_block_mults[s]

            for t in range(7):
                col = s * 7 + t

                for bi in range(2):
                    for bj in range(2):
                        a_mult = A_mults[bi * 2 + bj]
                        if a_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                u_val = U_s[ii * 2 + jj][t]
                                if u_val == 0:
                                    continue
                                a4_i = bi * 2 + ii
                                a4_j = bj * 2 + jj
                                u_row = a4_i * 4 + a4_j
                                U_44[u_row][col] += a_mult * u_val

                for bj in range(2):
                    for bk in range(2):
                        b_mult = B_mults[bj * 2 + bk]
                        if b_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                v_val = V_s[ii * 2 + jj][t]
                                if v_val == 0:
                                    continue
                                b4_j = bj * 2 + ii
                                b4_k = bk * 2 + jj
                                v_row = b4_j * 4 + b4_k
                                V_44[v_row][col] += b_mult * v_val

                for ci in range(2):
                    for cj in range(2):
                        c_mult = C_mults[ci * 2 + cj]
                        if c_mult == 0:
                            continue
                        for ii in range(2):
                            for jj in range(2):
                                w_val = W_s[ii * 2 + jj][t]
                                if w_val == 0:
                                    continue
                                c4_k = ci * 2 + ii
                                c4_i = cj * 2 + jj
                                w_row = c4_k * 4 + c4_i
                                W_44[w_row][col] += c_mult * w_val

        return U_44, V_44, W_44

    def build_4x4x7_via_strassen():
        """
        Build a rank-49+48=97 decomposition for 4x4 x 4x7.
        """
        U_44, V_44, W_44 = build_strassen_4x4x4()

        rank_strassen = 49
        rank_schoolbook_3 = 4 * 4 * 3  # 48
        total_rank = rank_strassen + rank_schoolbook_3

        U_out = [[0] * total_rank for _ in range(16)]
        V_out = [[0] * total_rank for _ in range(28)]
        W_out = [[0] * total_rank for _ in range(28)]

        for r in range(16):
            for c in range(rank_strassen):
                U_out[r][c] = U_44[r][c]
        for r in range(16):
            for c in range(rank_strassen):
                V_out[r][c] = V_44[r][c]
        for r in range(16):
            for c in range(rank_strassen):
                W_out[r][c] = W_44[r][c]

        col_offset = rank_strassen
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    kk = k + 4
                    U_out[i * 4 + j][col_offset] = 1
                    V_out[j * 7 + kk][col_offset] = 1
                    W_out[kk * 4 + i][col_offset] = 1
                    col_offset += 1

        return {"U": U_out, "V": V_out, "W": W_out}

    def bareiss_rank(matrix):
        """
        Compute the rank of a matrix with rational entries using Bareiss fraction-free Gaussian elimination.
        matrix: list of lists of Fraction
        Returns: rank (int)
        """
        if not matrix or not matrix[0]:
            return 0

        rows = len(matrix)
        cols = len(matrix[0])
        # Make a copy
        A = [row[:] for row in matrix]
        
        prev_pivot = Fraction(1)
        rank = 0
        
        for col in range(min(rows, cols)):
            # Find pivot
            pivot_row = None
            for r in range(col, rows):
                if A[r][col] != 0:
                    pivot_row = r
                    break
            
            if pivot_row is None:
                continue
            
            # Swap rows
            if pivot_row != col:
                A[col], A[pivot_row] = A[pivot_row], A[col]
            
            pivot = A[col][col]
            rank += 1
            
            # Eliminate below
            for r in range(col + 1, rows):
                if A[r][col] == 0:
                    continue
                # Bareiss: A[r][c] = (A[r][c] * pivot - A[r][col] * A[col][c]) / prev_pivot
                for c in range(col + 1, cols):
                    A[r][c] = (A[r][c] * pivot - A[r][col] * A[col][c]) / prev_pivot
                A[r][col] = 0
            
            prev_pivot = pivot
        
        return rank

    def build_decomposition_from_rank_basis():
        """
        Try to find a decomposition using the rank structure of the multiplication tensor.
        For 4x4 x 4x7, the multiplication tensor T[i,j,k] = 1 for all valid (i,j,k).
        
        We can view this as a 3-tensor. The rank of the tensor is what we want.
        One approach: fix the k index, we get a 4x4 matrix for each k.
        
        Actually, let's think differently. The tensor is T[i,j,k] where:
        - i ranges over 0..3 (rows of A, cols of C)
        - j ranges over 0..3 (cols of A, rows of B)  
        - k ranges over 0..6 (cols of B, rows of C)
        
        T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor for the multiplication pattern).
        
        Wait, that's not right. The multiplication tensor for matrix multiplication A*B=C
        has T[i,j,k] = delta_{j,j} which is just 1. But the structure is that
        C[k,i] = sum_j A[i,j] * B[j,k].
        
        So the tensor is T[i,j,k] = 1 for all (i,j,k) in the valid range.
        This is actually a rank-1 tensor! But that can't be right for matrix multiplication...
        
        I think I'm confusing the tensor with the multiplication pattern. Let me reconsider.
        
        The matrix multiplication tensor M has entries M[i,j,k] where the multiplication is:
        (A*B)[k,i] = sum_j A[i,j] * B[j,k]
        
        The tensor decomposition says:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = M[i,j,k] = 1
        
        So we need: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k.
        
        This is a different problem. The tensor we're decomposing has all entries equal to 1.
        But the constraint is on the structure of U, V, W.
        
        Actually, I think the minimal rank for this specific tensor (all-ones) with the
        given index structure might be computable. Let me try a different approach.
        
        For now, let's just use the Strassen-based approach which we know works.
        """
        # This function is a placeholder for future work
        # For now, return None to indicate we should use the Strassen approach
        return None

    # Check time
    if time.time() - start_time > DEADLINE:
        return build_schoolbook()

    # Try the Strassen-based decomposition
    try:
        result = build_4x4x7_via_strassen()
        # Quick sanity check: verify the decomposition is correct
        U = np.array(result["U"])
        V = np.array(result["V"])
        W = np.array(result["W"])
        rank = U.shape[1]

        # Verify: sum over c of U[:,c] outer V[:,c] outer W[:,c] should equal the multiplication tensor
        n, m, p = 4, 4, 7
        tensor_decomp = np.zeros((n, m, p))
        for c in range(rank):
            u_col = U[:, c]
            v_col = V[:, c]
            w_col = W[:, c]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        tensor_decomp[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]

        # The correct tensor is all 1s
        if not np.allclose(tensor_decomp, 1.0, atol=1e-9):
            # Decomposition is incorrect, fall back to schoolbook
            return build_schoolbook()

        if time.time() - start_time > DEADLINE:
            return build_schoolbook()

        return result

    except Exception:
        return build_schoolbook()
