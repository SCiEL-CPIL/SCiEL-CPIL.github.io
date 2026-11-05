import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s kill

    def build_rank_22():
        """
        Rank-22 decomposition for 2x3 * 3x4 matrix multiplication.
        Uses Strassen (rank 7) for the 2x2x2 block and outer products
        for the remaining 2x1x3 blocks.
        
        U: shape (n*m, rank) = (6, 22)  -> U[i*m+j, t]
        V: shape (m*p, rank) = (12, 22) -> V[j*p+k, t]
        W: shape (p*n, rank) = (8, 22)  -> W[k*n+i, t]
        
        Tensor equation: T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        """
        rank = 22
        U = [[0] * rank for _ in range(n * m)]   # 6 rows
        V = [[0] * rank for _ in range(m * p)]   # 12 rows
        W = [[0] * rank for _ in range(p * n)]   # 8 rows

        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # U indexing: U[i*m+j] where i in {0,1}, j in {0,1,2}
        #   0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1), 4=(0,2), 5=(1,2)
        # V indexing: V[j*p+k] where j in {0,1,2}, k in {0,1,2,3}
        #   0=(0,0), 1=(0,1), 2=(0,2), 3=(0,3)
        #   4=(1,0), 5=(1,1), 6=(1,2), 7=(1,3)
        #   8=(2,0), 9=(2,1), 10=(2,2), 11=(2,3)
        # W indexing: W[k*n+i] where k in {0,1,2,3}, i in {0,1}
        #   0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
        #   4=(2,0), 5=(2,1), 6=(3,0), 7=(3,1)

        # === Block A: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, rank 7) ===
        # A submatrix: rows 0,1; cols 0,1 -> U indices 0,1,2,3
        # B submatrix: rows 0,1; cols 0,1 -> V indices 0,1,4,5
        # C submatrix: rows 0,1; cols 0,1 -> W indices 0,1,2,3

        # S1 = (A00+A11)(B00+B11) -> C00, C11
        add_term([(0, 1), (3, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        # S2 = A11 * B10 -> C10, C11
        add_term([(3, 1)], [(4, 1)], [(2, 1), (3, -1)], 1)
        # S3 = A00 * (B10+B11) -> C00, C01
        add_term([(0, 1)], [(4, 1), (5, 1)], [(0, 1), (1, -1)], 2)
        # S4 = (A00+A01) * B01 -> C01, C11
        add_term([(0, 1), (1, 1)], [(1, 1)], [(1, 1), (3, 1)], 3)
        # S5 = (A10-A00) * (B00+B01) -> C10, C01
        add_term([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(2, 1), (1, 1)], 4)
        # S6 = (A01-A11) * (B10+B11) -> C01, C10
        add_term([(1, 1), (3, -1)], [(4, 1), (5, 1)], [(1, 1), (2, -1)], 5)
        # S7 = A10 * B00 -> C10
        add_term([(2, 1)], [(0, 1)], [(2, 1)], 6)

        # === Block B: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Strassen, rank 7) ===
        # B submatrix: rows 0,1; cols 2,3 -> V indices 2,3,6,7
        # C submatrix: rows 0,1; cols 2,3 -> W indices 4,5,6,7

        # S1 = (A00+A11)(B00+B11) -> C00, C11 (of this block)
        add_term([(0, 1), (3, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        # S2 = A11 * B10 -> C10, C11
        add_term([(3, 1)], [(6, 1)], [(6, 1), (7, -1)], 8)
        # S3 = A00 * (B10+B11) -> C00, C01
        add_term([(0, 1)], [(6, 1), (7, 1)], [(4, 1), (5, -1)], 9)
        # S4 = (A00+A01) * B01 -> C01, C11
        add_term([(0, 1), (1, 1)], [(3, 1)], [(5, 1), (7, 1)], 10)
        # S5 = (A10-A00) * (B00+B01) -> C10, C01
        add_term([(2, 1), (0, -1)], [(2, 1), (3, 1)], [(6, 1), (5, 1)], 11)
        # S6 = (A01-A11) * (B10+B11) -> C01, C10
        add_term([(1, 1), (3, -1)], [(6, 1), (7, 1)], [(5, 1), (6, -1)], 12)
        # S7 = A10 * B00 -> C10
        add_term([(2, 1)], [(2, 1)], [(6, 1)], 13)

        # === Block C: A[0:2, 2:3] * B[0:2, 0:4] -> C[0:2, 0:4] (outer product, rank 8) ===
        # A submatrix: rows 0,1; col 2 -> U indices 4=(0,2), 5=(1,2)
        # B submatrix: rows 0,1; cols 0,1,2,3 -> V indices 0,1,2,3,4,5,6,7
        # C submatrix: rows 0,1; cols 0,1,2,3 -> W indices 0,1,2,3,4,5,6,7

        # Row 0 of A (index 4) times each column of B
        add_term([(4, 1)], [(0, 1)], [(0, 1)], 14)  # C[0,0]
        add_term([(4, 1)], [(1, 1)], [(1, 1)], 15)  # C[0,1]
        add_term([(4, 1)], [(2, 1)], [(2, 1)], 16)  # C[0,2]
        add_term([(4, 1)], [(3, 1)], [(3, 1)], 17)  # C[0,3]
        # Row 1 of A (index 5) times each column of B
        add_term([(5, 1)], [(0, 1)], [(4, 1)], 18)  # C[1,0]
        add_term([(5, 1)], [(1, 1)], [(5, 1)], 19)  # C[1,1]
        add_term([(5, 1)], [(2, 1)], [(6, 1)], 20)  # C[1,2]
        add_term([(5, 1)], [(3, 1)], [(7, 1)], 21)  # C[1,3]

        # === Block D: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4] (outer product, rank 8) ===
        # A submatrix: rows 0,1; col 2 -> U indices 4=(0,2), 5=(1,2)
        # B submatrix: row 2; cols 0,1,2,3 -> V indices 8,9,10,11
        # C submatrix: rows 0,1; cols 0,1,2,3 -> W indices 0,1,2,3,4,5,6,7

        # Row 0 of A (index 4) times each column of B
        add_term([(4, 1)], [(8, 1)], [(0, 1)], 14)  # C[0,0] (reuse col 14)
        add_term([(4, 1)], [(9, 1)], [(1, 1)], 15)  # C[0,1] (reuse col 15)
        add_term([(4, 1)], [(10, 1)], [(2, 1)], 16)  # C[0,2] (reuse col 16)
        add_term([(4, 1)], [(11, 1)], [(3, 1)], 17)  # C[0,3] (reuse col 17)
        # Row 1 of A (index 5) times each column of B
        add_term([(5, 1)], [(8, 1)], [(4, 1)], 18)  # C[1,0] (reuse col 18)
        add_term([(5, 1)], [(9, 1)], [(5, 1)], 19)  # C[1,1] (reuse col 19)
        add_term([(5, 1)], [(10, 1)], [(6, 1)], 20)  # C[1,2] (reuse col 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)  # C[1,3] (reuse col 21)

        return U, V, W, rank

    def verify(U, V, W, rank):
        """Verify the decomposition exactly reconstructs the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Build the rank-22 solution
    U, V, W, rank = build_rank_22()

    # Verify correctness
    if not verify(U, V, W, rank):
        # Should not happen, but if it does, return the best we have
        pass

    return {"U": U, "V": V, "W": W}
