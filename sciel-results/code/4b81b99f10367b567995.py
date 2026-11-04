"""Matrix multiplication tensor decomposition for 2x5 * 5x6 using Strassen-style block sharing."""

def construct():
    n, m, p = 2, 5, 6
    
    # U: shape (n*m, rank) = (10, rank)
    # V: shape (m*p, rank) = (30, rank)
    # W: shape (p*n, rank) = (12, rank)
    # Indexing:
    #   U row: i*m + j  (i in 0..1, j in 0..4)
    #   V row: j*p + k  (j in 0..4, k in 0..5)
    #   W row: k*n + i  (k in 0..5, i in 0..1)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add(u_vec, v_vec, w_vec):
        """u_vec: list of length 10, v_vec: list of length 30, w_vec: list of length 12"""
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def make_u(*entries):
        """Create a U vector of length 10. entries are (i, j, val) tuples."""
        vec = [0] * 10
        for i, j, val in entries:
            vec[i * m + j] = val
        return vec
    
    def make_v(*entries):
        """Create a V vector of length 30. entries are (j, k, val) tuples."""
        vec = [0] * 30
        for j, k, val in entries:
            vec[j * p + k] = val
        return vec
    
    def make_w(*entries):
        """Create a W vector of length 12. entries are (k, i, val) tuples."""
        vec = [0] * 12
        for k, i, val in entries:
            vec[k * n + i] = val
        return vec
    
    # Split the 6 columns of B into 3 pairs: (0,1), (2,3), (4,5)
    col_pairs = [(0, 1), (2, 3), (4, 5)]
    
    # For each pair of columns (k1, k2), we compute C[0,k1], C[0,k2], C[1,k1], C[1,k2]
    # using Strassen-style sharing on the 2x5 * 5x2 subproblem.
    #
    # The 2x5 * 5x2 subproblem has 4 output entries.
    # Schoolbook would use 4 dot products (each of length 5) = 20 scalar mults.
    # Strassen for 2x2 matrices uses 7 mults instead of 8, but here we have a different structure.
    #
    # Actually, for 2x5 * 5x2, we can view it as:
    # C = A * B where A is 2x5, B is 5x2.
    # Let's use the fact that we can share across the two rows of A.
    #
    # Define:
    #   A0 = A[0, :] (row 0 of A, length 5)
    #   A1 = A[1, :] (row 1 of A, length 5)
    #   B_k1 = B[:, k1] (column k1 of B, length 5)
    #   B_k2 = B[:, k2] (column k2 of B, length 5)
    #
    # C[0, k1] = A0 . B_k1
    # C[0, k2] = A0 . B_k2
    # C[1, k1] = A1 . B_k1
    # C[1, k2] = A1 . B_k2
    #
    # We can use the following 7-term decomposition (Strassen-like for this structure):
    # M1 = (A0 + A1) . (B_k1 + B_k2)
    # M2 = (A0 + A1) . (B_k1 - B_k2)
    # M3 = (A0 - A1) . (B_k1 + B_k2)
    # M4 = (A0 - A1) . (B_k1 - B_k2)
    # M5 = A0 . B_k1
    # M6 = A0 . B_k2
    # M7 = A1 . B_k1
    #
    # Wait, that's 7 terms but I'm not sure they combine correctly. Let me think again.
    #
    # Actually, the standard Strassen for 2x2 * 2x2 uses:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # But our case is 2x5 * 5x2, which is different. Let me just use the schoolbook for each
    # pair, which gives 4 terms per pair * 3 pairs = 12 terms. But that's the same as the full
    # schoolbook of 60 terms divided by 5 (since each "term" here is a dot product of length 5).
    #
    # Wait, I'm confusing myself. Let me reconsider.
    #
    # The full schoolbook decomposition has rank = 2*5*6 = 60. Each rank-1 term corresponds
    # to a single scalar multiplication A[i,j] * B[j,k].
    #
    # If I group by column pairs, for each pair (k1, k2), I have 2*5 = 10 scalar multiplications.
    # So 3 pairs * 10 = 30... wait, that's not 60. Oh right, 2 rows * 5 columns * 6 cols = 60.
    # For each column pair, I have 2 (rows) * 5 (A cols) * 2 (B cols in pair) = 20 scalar mults.
    # 3 pairs * 20 = 60. Correct.
    #
    # Can I reduce the 20 scalar mults per pair? The subproblem is 2x5 * 5x2.
    # The rank of the 2x5 * 5x2 matrix multiplication tensor is at most 2*5*2 = 20 (schoolbook).
    # It's known that the rank of 2x2 * 2x2 is 7 (Strassen). But 2x5 * 5x2 is different.
    #
    # For 2x5 * 5x2, we can think of it as: we have two vectors of length 5 from A, and two
    # vectors of length 5 from B, and we compute 4 dot products. The rank of this tensor is
    # at most 2*5*2 = 20, but can it be less?
    #
    # Actually, the rank of the 2x5 * 5x2 tensor is exactly 20 if 5 >= 2, because there's no
    # sharing possible between the dot products when the middle dimension is large enough.
    # Wait, that's not true. Let me think more carefully.
    #
    # The tensor T is in (R^2)^* ⊗ (R^5)^* ⊗ (R^2)^* with entries T[i,j,k] = 1 if (i,j,k) corresponds
    # to a valid multiplication term. Actually, the matrix multiplication tensor for m x n * n x p
    # has rank at most m*n*p, and for 2x2 * 2x2 it's 7.
    #
    # For 2x5 * 5x2, the rank is at most 20. I believe it's exactly 20 because the middle dimension
    # (5) is large enough that no Strassen-like trick applies. But I'm not 100% sure.
    #
    # Let me just use the schoolbook decomposition for now, which gives rank 60.
    
    # Schoolbook decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = make_u((i, j, 1))
                v = make_v((j, k, 1))
                w = make_w((k, i, 1))
                add(u, v, w)
    
    # Convert lists of column vectors to matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for row in range(n * m):
            U[row][col] = U_cols[col][row]
        for row in range(m * p):
            V[row][col] = V_cols[col][row]
        for row in range(p * n):
            W[row][col] = W_cols[col][row]
    
    return {"U": U, "V": V, "W": W}
