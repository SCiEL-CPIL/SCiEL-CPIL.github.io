import numpy as np
import time

DIMENSIONS = [2, 2, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS  # 2, 2, 7
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (14, rank)
    # W: (p*n, rank) = (14, rank)
    # Indexing: a = i*m+j (row of A, col of A), b = j*p+k (row of B, col of B), c = k*n+i (row of C, col of C)
    # W[k*n + i, r] follows the trace convention

    # Build target tensor T of shape (4, 14, 14)
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def check_decomposition(U, V, W, tol=1e-9):
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_rec, T, atol=tol)

    # Strassen-based rank-25 decomposition for 2x2 * 2x7
    # Strassen for 2x2 * 2x2 uses 7 products:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B12-B11)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # For 2x2 * 2x7, the B matrix is 2x7. We partition B into blocks of 2 columns:
    # B = [B^1 | B^2 | B^3 | B^4 | b^5] where B^j is 2x2 for j=1..4, b^5 is 2x1
    # Actually, let's think of it as: B has 7 columns. We can group them.
    # 
    # A better approach: think of the 2x2 * 2x7 as computing C = A * B where A is 2x2, B is 2x7.
    # C is 2x7. C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # We can use Strassen's 7-product scheme on the 2x2 block structure.
    # Partition B into 3 groups: first 2 cols, next 2 cols, next 2 cols, last 1 col.
    # Actually, let's use a systematic approach.
    #
    # For 2x2 * 2x7:
    # A = [[a, b], [c, d]]  (indices: a=0*2+0=0, b=0*2+1=1, c=1*2+0=2, d=1*2+1=3)
    # B is 2x7: rows j=0,1; cols k=0..6
    #   B[0,k] = index 0*7+k = k
    #   B[1,k] = index 1*7+k = 7+k
    # C is 2x7: rows i=0,1; cols k=0..6
    #   C[i,k] = index k*2+i
    #   C[0,k] = k*2+0 = 2k
    #   C[1,k] = k*2+1 = 2k+1
    #
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    #
    # Strassen 7 products for 2x2 * 2x2 (using columns k=0,1 as an example):
    # Let B^col = [[B00, B01], [B10, B11]] for a pair of columns.
    #
    # For each pair of output columns (k, k+1), we need:
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[0,k+1] = a*B[0,k+1] + b*B[1,k+1]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    # C[1,k+1] = c*B[0,k+1] + d*B[1,k+1]
    #
    # Using Strassen:
    # M1 = (a+d)(B[0,k]+B[1,k+1])  -- wait, let me be more careful.
    #
    # Standard Strassen for C = A*B where A,B are 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    #
    # For 2x2 * 2x7, I'll partition the 7 columns of B into:
    # - 3 pairs: (0,1), (2,3), (4,5)  -> 3 Strassen blocks, each giving 4 C entries
    # - 1 single: column 6 -> needs 2 products (schoolbook for 2x2 * 2x1)
    # Total: 3*7 + 2 = 23? No, that's not right either.
    #
    # Actually, let me reconsider. The rank of the 2x2 * 2x7 tensor.
    # The 2x2 * 2x2 tensor has rank 7 (Strassen).
    # The 2x2 * 2x7 tensor can be seen as 3 copies of 2x2*2x2 plus one 2x2*2x1.
    # 2x2*2x1 has rank 2 (schoolbook, since it's just two dot products).
    # But we can potentially share some structure.
    #
    # Actually, a cleaner approach: the tensor for 2x2 * 2x7 has 4*14*14 = 784 entries,
    # with 2*2*7 = 28 non-zero entries (each T[a,b,c]=1 for the 28 (i,j,k) combos).
    #
    # Let me just directly construct a rank-25 decomposition using Strassen-like structure.
    #
    # For columns 0-5 (3 pairs), use Strassen (7 products each) = 21 products
    # For column 6, use 2 products (schoolbook for 2x2 * 2x1)
    # Total: 23 products.
    #
    # Wait, but the Strassen products for different column pairs are independent,
    # so we can't share them. Total rank would be 3*7 + 2 = 23.
    #
    # Hmm, but actually we might do better. Let me think about this differently.
    #
    # Actually, let me just go with a known good decomposition. The key insight from
    # the cell findings is that a rank-25 Strassen decomposition works. Let me
    # construct it carefully.
    #
    # For 2x2 * 2x7:
    # A indices: a0=0 (i=0,j=0), a1=1 (i=0,j=1), a2=2 (i=1,j=0), a3=3 (i=1,j=1)
    # B indices: b_{j,k} = j*7+k, so b00=0,b01=1,...,b06=6, b10=7,...,b16=13
    # C indices: c_{i,k} = k*2+i, so c_{0,0}=0,c_{1,0}=1,c_{0,1}=2,c_{1,1}=3,...,c_{0,6}=12,c_{1,6}=13
    #
    # T[a,b,c] = 1 iff there exist i,j,k such that a=i*2+j, b=j*7+k, c=k*2+i
    # i.e., T[i*2+j, j*7+k, k*2+i] = 1
    #
    # Let me enumerate all 28 non-zero entries:
    # i=0,j=0: a=0, b=k, c=2k for k=0..6 -> (0,0,0),(0,1,2),(0,2,4),(0,3,6),(0,4,8),(0,5,10),(0,6,12)
    # i=0,j=1: a=1, b=7+k, c=2k for k=0..6 -> (1,7,0),(1,8,2),(1,9,4),(1,10,6),(1,11,8),(1,12,10),(1,13,12)
    # i=1,j=0: a=2, b=k, c=2k+1 for k=0..6 -> (2,0,1),(2,1,3),(2,2,5),(2,3,7),(2,4,9),(2,5,11),(2,6,13)
    # i=1,j=1: a=3, b=7+k, c=2k+1 for k=0..6 -> (3,7,1),(3,8,3),(3,9,5),(3,10,7),(3,11,9),(3,12,11),(3,13,13)

    # Now let's build a rank-25 decomposition.
    # Strategy: For the 2x2 * 2x7 tensor, use Strassen's 7-product scheme
    # applied to overlapping column pairs to reduce rank.
    #
    # Actually, the simplest correct approach: use the fact that we can decompose
    # 2x2 * 2x7 as a sum of 7 rank-1 tensors per "Strassen block" but sharing
    # across blocks where possible.
    #
    # Let me try a different structural approach. The 2x2 * 2x7 tensor can be
    # viewed as: for each k (0..6), we have a 2x2 * 2x1 sub-tensor (rank 2 each),
    # giving 14 total. But we can do better by grouping.
    #
    # Group columns as: {0,1}, {2,3}, {4,5}, {6}
    # For each pair {k, k+1}, the sub-tensor is 2x2 * 2x2 (rank 7 by Strassen)
    # For single column {6}, the sub-tensor is 2x2 * 2x1 (rank 2)
    # Total: 3*7 + 2 = 23
    #
    # But wait - can we share Strassen products between adjacent pairs?
    # If we use a sliding window approach... probably not easily.
    #
    # Let me just go with rank 23 using 3 Strassen blocks + 1 schoolbook block.
    #
    # Actually, I realize I should just verify what rank is achievable.
    # The known lower bound for 2x2 * 2x7 is at least 20 (from various results).
    # Strassen gives 7 for 2x2*2x2, and the "linear" part adds more.
    #
    # Let me just construct the rank-23 decomposition explicitly.

    def strassen_2x2_2x7():
        """Construct rank-23 decomposition: 3 Strassen blocks (cols 0-1, 2-3, 4-5) + schoolbook (col 6)"""
        rank = 23
        U = np.zeros((4, rank))
        V = np.zeros((14, rank))
        W = np.zeros((14, rank))

        col_offset = 0

        # Strassen block for columns (k0, k0+1)
        def add_strassen_block(k0, k1, offset):
            # k0, k1 are the two column indices in B (and C)
            # B indices: b_{0,k0}=k0, b_{0,k1}=k1, b_{1,k0}=7+k0, b_{1,k1}=7+k1
            # C indices: c_{0,k0}=2*k0, c_{1,k0}=2*k0+1, c_{0,k1}=2*k1, c_{1,k1}=2*k1+1
            # A indices: a0=0 (A[0,0]), a1=1 (A[0,1]), a2=2 (A[1,0]), a3=3 (A[1,1])
            
            # Strassen products:
            # M1 = (a+d)(e+h) where e=B[0,k0], f=B[0,k1], g=B[1,k0], h=B[1,k1]
            # M2 = (c+d)*e
            # M3 = a*(f-h)
            # M4 = d*(g-e)
            # M5 = (a+b)*h
            # M6 = (c-a)*(e+f)
            # M7 = (b-d)*(g+h)
            
            # M1: (a+d)(e+h)
            r = offset
            U[0, r] = 1  # a
            U[3, r] = 1  # d
            V[k0, r] = 1      # e = B[0,k0]
            V[7+k1, r] = 1    # h = B[1,k1]
            W[2*k0, r] = 1    # C[0,k0] = C11
            W[2*k0+1, r] = 1  # C[1,k0] = C21 (wait, need to be careful)
            # C11 = M1+M4-M5+M7 -> contributes to C[0,k0]
            # C21 = M2+M4 -> contributes to C[1,k0]
            # C12 = M3+M5 -> contributes to C[0,k1]
            # C22 = M1-M2+M3+M6 -> contributes to C[1,k1]
            # So M1 contributes to C[0,k0] and C[1,k1]
            W[2*k0, r] = 1    # C[0,k0]
            W[2*k1+1, r] = 1  # C[1,k1]
            r += 1

            # M2: (c+d)*e
            U[2, r] = 1  # c
            U[3, r] = 1  # d
            V[k0, r] = 1  # e = B[0,k0]
            W[2*k0+1, r] = 1  # C[1,k0] (C21)
            W[2*k1+1, r] = -1 # C[1,k1] (C22)
            r += 1

            # M3: a*(f-h)
            U[0, r] = 1  # a
            V[k1, r] = 1      # f = B[0,k1]
            V[7+k0, r] = -1   # -g = -B[1,k0]  (wait, M3 = a*(f-h) where f=B[0,k1], h=B[1,k1])
            # Correction: M3 = a*(f-h) where f=B[0,k1], h=B[1,k1]
            V[k1, r] = 1      # f = B[0,k1]
            V[7+k1, r] = -1   # -h = -B[1,k1]
            W[2*k1, r] = 1    # C[0,k1] (C12)
            W[2*k1+1, r] = 1  # C[1,k1] (C22)
            r += 1

            # M4: d*(g-e) where g=B[1,k0], e=B[0,k0]
            U[3, r] = 1  # d
            V[7+k0, r] = 1  # g = B[1,k0]
            V[k0, r] = -1   # -e = -B[0,k0]
            W[2*k0, r] = 1    # C[0,k0] (C11)
            W[2*k0+1, r] = 1  # C[1,k0] (C21)
            r += 1

            # M5: (a+b)*h where h=B[1,k1]
            U[0, r] = 1  # a
            U[1, r] = 1  # b
            V[7+k1, r] = 1  # h = B[1,k1]
            W[2*k0, r] = -1  # C[0,k0] (C11, with -M5)
            W[2*k1, r] = 1   # C[0,k1] (C12)
            r += 1

            # M6: (c-a)*(e+f) where e=B[0,k0], f=B[0,k1]
            U[2, r] = 1  # c
            U[0, r] = -1 # -a
            V[k0, r] = 1  # e
            V[k1, r] = 1  # f
            W[2*k1+1, r] = 1  # C[1,k1] (C22)
            r += 1

            # M7: (b-d)*(g+h) where g=B[1,k0], h=B[1,k1]
            U[1, r] = 1  # b
            U[3, r] = -1 # -d
            V[7+k0, r] = 1  # g
            V[7+k1, r] = 1  # h
            W[2*k0, r] = 1  # C[0,k0] (C11)
            r += 1

            return offset + 7

        # Block 1: columns 0, 1
        col_offset = add_strassen_block(0, 1, col_offset)
        # Block 2: columns 2, 3
        col_offset = add_strassen_block(2, 3, col_offset)
        # Block 3: columns 4, 5
        col_offset = add_strassen_block(4, 5, col_offset)

        # Schoolbook for column 6: 2 products
        # C[0,6] = a*B[0,6] + b*B[1,6]
        # C[1,6] = c*B[0,6] + d*B[1,6]
        # Product 1: a * B[0,6]
        r = col_offset
        U[0, r] = 1
        V[6, r] = 1   # B[0,6] = index 0*7+6 = 6
        W[12, r] = 1  # C[0,6] = 6*2+0 = 12
        r += 1

        # Product 2: b * B[1,6]
        U[1, r] = 1
        V[13, r] = 1  # B[1,6] = index 1*7+6 = 13
        W[12, r] = 1  # C[0,6] = 12
        r += 1

        # Product 3: c * B[0,6]
        U[2, r] = 1
        V[6, r] = 1
        W[13, r] = 1  # C[1,6] = 6*2+1 = 13
        r += 1

        # Product 4: d * B[1,6]
        U[3, r] = 1
        V[13, r] = 1
        W[13, r] = 1
        r += 1

        # That gives 4 products for the last column, total = 21 + 4 = 25
        # Let me fix: schoolbook for 2x2 * 2x1 needs 4 products (not 2)
        # C[0,6] = a*B[0,6] + b*B[1,6]  -> 2 terms
        # C[1,6] = c*B[0,6] + d*B[1,6]  -> 2 terms
        # Total 4 rank-1 terms for this column.
        # Grand total: 3*7 + 4 = 25

        return U, V, W

    U, V, W = strassen_2x2_2x7()
    best_rank = 25

    # Verify
    if not check_decomposition(U, V, W):
        # Fallback: try to fix or use schoolbook
        # Debug: print which entries are wrong
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_rec - T
        # Try to repair small errors
        # If errors are tiny, round
        if np.max(np.abs(diff)) < 1e-6:
            # Round to nearest integer
            U = np.round(U)
            V = np.round(V)
            W = np.round(W)
            if check_decomposition(U, V, W):
                pass
            else:
                # Use schoolbook
                rank = 28
                U = np.zeros((4, rank))
                V = np.zeros((14, rank))
                W = np.zeros((14, rank))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U[i*m+j, col] = 1
                            V[j*p+k, col] = 1
                            W[k*n+i, col] = 1
                            col += 1
                best_rank = 28
        else:
            # Use schoolbook
            rank = 28
            U = np.zeros((4, rank))
            V = np.zeros((14, rank))
            W = np.zeros((14, rank))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m+j, col] = 1
                        V[j*p+k, col] = 1
                        W[k*n+i, col] = 1
                        col += 1
            best_rank = 28

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
