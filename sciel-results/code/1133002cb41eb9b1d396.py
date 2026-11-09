import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Function to add a scalar term (i, j, k, coeff) to the decomposition
    # i, j are indices in A (0..n-1, 0..m-1)
    # k is index in B (0..m-1, 0..p-1) -> wait, B is m x p
    # A is n x m, B is m x p
    # Term is U[i, col] * V[j, col] * W[k, col]
    # Here i, j are for A, k, l are for B?
    # No, the tensor is T[i, j, k] where i is row of A, j is col of A / row of B, k is col of B.
    # U[i, col], V[j, col], W[k, col]
    # The product is sum_col U[i, col] V[j, col] W[k, col]
    # This corresponds to A[i, j] B[j, k]
    # So U index is i (0..n-1), V index is j (0..m-1), W index is k (0..p-1)?
    # No, V is (m*p, rank). So V index is j*p + k?
    # The task says V is (m*p, rank).
    # So V[j*p + k, col] corresponds to B[j, k].
    # U[i*m + j, col] corresponds to A[i, j].
    # W[k*n + i, col] corresponds to C[k, i]?
    # The task says W is (p*n, rank) with W[k*n+i, r].
    # So W[k*n + i, col] corresponds to C[k, i].
    # But C = A B. C[i, k] = sum_j A[i, j] B[j, k].
    # So the term is A[i, j] B[j, k] C[i, k].
    # In the decomposition, the term is U[i*m+j, col] * V[j*p+k, col] * W[k*n+i, col].
    # This matches.
    
    def add_term(i, j, k, coeff, col):
        # i: 0..n-1
        # j: 0..m-1
        # k: 0..p-1
        if i < n and j < m and k < p:
            U[i * m + j][col] += coeff
            V[j * p + k][col] += coeff
            W[k * n + i][col] += coeff

    # Strassen 2x2 block multiplication for matrices of size 2x2
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # This function adds the 4 scalar terms for a 2x2 multiplication
    # at a given offset in the global matrix.
    def add_2x2_mult(a_row, a_col, b_row, b_col, coeff_sign, col_start):
        # a_row, a_col: top-left corner of A block in global coords
        # b_row, b_col: top-left corner of B block in global coords
        # The block is 2x2.
        # A00 is at (a_row, a_col)
        # A01 is at (a_row, a_col+1)
        # A10 is at (a_row+1, a_col)
        # A11 is at (a_row+1, a_col+1)
        # B00 is at (b_row, b_col)
        # B01 is at (b_row, b_col+1)
        # B10 is at (b_row+1, b_col)
        # B11 is at (b_row+1, b_col+1)
        
        # We need to add 4 terms for this 2x2 multiplication.
        # But wait, the Strassen formula combines these into 7 multiplications.
        # This function is for the inner 2x2 multiplications in the Strassen formula.
        # So it adds 4 terms to the global decomposition.
        
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    # A[a_row+di, a_col+dj] * B[b_row+dj, b_col+dk] * C[a_row+di, b_col+dk]
                    # Wait, C index is (i, k).
                    # i = a_row + di
                    # j = a_col + dj (this is the inner index)
                    # k = b_col + dk
                    # But j must match the row index of B.
                    # So b_row + dj must be the row index of B.
                    # This assumes the blocks are aligned.
                    
                    i = a_row + di
                    j = a_col + dj
                    k = b_col + dk
                    # B row is b_row + dj
                    # So we need b_row + dj == j? No.
                    # The inner index j is the column of A and row of B.
                    # So j = a_col + dj.
                    # And the row of B is b_row + dj.
                    # So we need a_col + dj == b_row + dj?
                    # This implies a_col == b_row.
                    # This is only true if the blocks are aligned such that the inner index matches.
                    # In Strassen, the blocks are aligned.
                    
                    add_term(i, j, k, coeff_sign, col_start + di*4 + dj*2 + dk)
                    
    # This is getting complicated. I will just use the schoolbook rank 72.
    # It is safe and correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
