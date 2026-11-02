import time

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit

    n, m, p = 3, 4, 4

    # Build the rank-44 base solution using Strassen + trivial
    rank_base = 44
    U_base = [[0]*rank_base for _ in range(n*m)]
    V_base = [[0]*rank_base for _ in range(m*p)]
    W_base = [[0]*rank_base for _ in range(p*n)]
    col = 0

    def add_strassen(U_mat, V_mat, W_mat, a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        c = start_col
        # m1 = (a00+a11)(b00+b11)
        U_mat[a_r*m + a_c][c] += 1
        U_mat[(a_r+1)*m + (a_c+1)][c] += 1
        V_mat[b_r*p + b_c][c] += 1
        V_mat[(b_r+1)*p + (b_c+1)][c] += 1
        W_mat[c_c*n + c_r][c] += 1
        W_mat[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1
        # m2 = (a10+a11)b00
        U_mat[(a_r+1)*m + a_c][c] += 1
        U_mat[(a_r+1)*m + (a_c+1)][c] += 1
        V_mat[b_r*p + b_c][c] += 1
        W_mat[c_c*n + (a_r+1)][c] += 1
        W_mat[(c_c+1)*n + (a_r+1)][c] -= 1
        c += 1
        # m3 = a00(b01-b11)
        U_mat[a_r*m + a_c][c] += 1
        V_mat[b_r*p + (b_c+1)][c] += 1
        V_mat[(b_r+1)*p + (b_c+1)][c] -= 1
        W_mat[(c_c+1)*n + a_r][c] += 1
        W_mat[(c_c+1)*n + (a_r+1)][c] += 1
        c += 1
        # m4 = a11(b10-b00)
        U_mat[(a_r+1)*m + (a_c+1)][c] += 1
        V_mat[(b_r+1)*p + b_c][c] += 1
        V_mat[b_r*p + b_c][c] -= 1
        W_mat[c_c*n + a_r][c] += 1
        W_mat[c_c*n + (a_r+1)][c] += 1
        c += 1
        # m5 = (a00+a01)b11
        U_mat[a_r*m + a_c][c] += 1
        U_mat[a_r*m + (a_c+1)][c] += 1
        V_mat[(b_r+1)*p + (b_c+1)][c] += 1
        W_mat[c_c*n + a_r][c] -= 1
        W_mat[(c_c+1)*n + a_r][c] += 1
        c += 1
        # m6 = (a10-a00)(b00+b01)
        U_mat[(a_r+1)*m + a_c][c] += 1
        U_mat[a_r*m + a_c][c] -= 1
        V_mat[b_r*p + b_c][c] += 1
        V_mat[b_r*p + (b_c+1)][c] += 1
        W_mat[(c_c+1)*n + (a_r+1)][c] += 1
        c += 1
        # m7 = (a01-a11)(b10+b11)
        U_mat[a_r*m + (a_c+1)][c] += 1
        U_mat[(a_r+1)*m + (a_c+1)][c] -= 1
        V_mat[(b_r+1)*p + b_c][c] += 1
        V_mat[(b_r+1)*p + (b_c+1)][c] += 1
        W_mat[c_c*n + a_r][c] += 1
        c += 1
        return c

    def add_trivial_1x4x2(U_mat, V_mat, W_mat, a_r, b_c, start_col):
        c = start_col
        for j in range(4):
            U_mat[a_r*m + j][c] += 1
            V_mat[j*p + b_c][c] += 1
            W_mat[b_c*n + a_r][c] += 1
            c += 1
            U_mat[a_r*m + j][c] += 1
            V_mat[j*p + (b_c+1)][c] += 1
            W_mat[(b_c+1)*n + a_r][c] += 1
            c += 1
        return c

    # Left Side (cols 0,1)
    col = add_strassen(U_base, V_base, W_base, 0, 0, 0, 0, 0, 0, col)
    col = add_strassen(U_base, V_base, W_base, 0, 2, 2, 0, 0, 0, col)
    col = add_trivial_1x4x2(U_base, V_base, W_base, 2, 0, col)
    # Right Side (cols 2,3)
    col = add_strassen(U_base, V_base, W_base, 0, 0, 0, 2, 0, 2, col)
    col = add_strassen(U_base, V_base, W_base, 0, 2, 2, 2, 0, 2, col)
    col = add_trivial_1x4x2(U_base, V_base, W_base, 2, 2, col)

    # Verify base solution is correct
    def verify(U_mat, V_mat, W_mat):
        # Check that sum_k U[i,k]*V[j,k]*W[k*n+i', j'] matches matmul
        # For matrix multiplication: C[i,j] = sum_k A[i,k]*B[k,j]
        # Tensor: T[i,j,k] where i in A, j in B, k in C
        # W[k*n+i] is the C index for output (i,j)
        # Actually W is indexed as W[k*n+i, r] where k is C row, i is C col
        # Let me re-check the convention: W[k*n+i, r]
        # So W[row, col] where row = c_row * n + c_col
        # U[a_idx, r] where a_idx = a_row * m + a_col
        # V[b_idx, r] where b_idx = b_row * p + b_col
        # C[c_row, c_col] = sum_k A[a_row, k] * B[k, c_col]
        # In tensor form: for each r, U[a_idx, r] * V[b_idx, r] * W[c_idx, r]
        # should sum to the matmul tensor
        # Let's verify with random matrices
        import random
        for trial in range(3):
            A = [[random.randint(-2, 2) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-2, 2) for _ in range(p)] for _ in range(m)]
            # Compute true C
            C_true = [[sum(A[i][k]*B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]
            # Compute C from decomposition
            C_decomp = [[0]*p for _ in range(n)]
            for r in range(len(U_mat[0])):
                for i in range(n):
                    for k in range(m):
                        u_val = U_mat[i*m + k][r]
                        if u_val == 0:
                            continue
                        for j in range(p):
                            v_val = V_mat[k*p + j][r]
                            if v_val == 0:
                                continue
                            w_val = W_mat[j*n + i][r]
                            if w_val == 0:
                                continue
                            C_decomp[i][j] += u_val * v_val * w_val
            if C_true != C_decomp:
                return False
        return True

    if not verify(U_base, V_base, W_base):
        # Fallback to trivial 48-rank solution
        rank_triv = n * m * p
        U_triv = [[0]*rank_triv for _ in range(n*m)]
        V_triv = [[0]*rank_triv for _ in range(m*p)]
        W_triv = [[0]*rank_triv for _ in range(p*n)]
        idx = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U_triv[i*m + k][idx] = 1
                    V_triv[k*p + j][idx] = 1
                    W_triv[j*n + i][idx] = 1
                    idx += 1
        return {"U": U_triv, "V": V_triv, "W": W_triv}

    best_result = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = rank_base

    # Now try to find a rank-43 solution by attempting to merge two rank-1 terms
    # Strategy: try all pairs of columns and see if we can express one as a linear
    # combination of the other plus adjustments, or find a dependency.
    # 
    # Alternative approach: Use the structure of Strassen. In each Strassen block,
    # the 7 products are linearly independent in general. But across blocks,
    # there might be dependencies.
    #
    # Let's try a different approach: attempt to find if any 43 columns can span
    # the same space by checking if any column is a linear combination of others
    # in a way that preserves the tensor decomposition.
    #
    # Actually, the cleanest approach for rank reduction is:
    # 1. Build the 44-term decomposition
    # 2. Try to find if two terms can be "merged" into one by finding
    #    coefficients such that the sum of two rank-1 tensors equals one rank-1 tensor
    #    (which is generally impossible unless they're already collinear)
    #
    # Better approach: Look for a different block decomposition.
    # What if we use a 3x3x4 or 3x4x3 split instead of 3x4x2?
    #
    # Let's try: Split B into cols 0,1,2 and cols 3.
    # Left: 3x4x3, Right: 3x4x1
    # 3x4x1: rank 4 (trivial: 3*4/3... no, it's just 4 terms: A[*,k]*B[k,3])
    # Actually 3x4x1 has rank 4 (each of the 4 k-indices gives one term)
    # 3x4x3: harder. Split into 2x4x3 + 1x4x3.
    # 2x4x3: Split A into 2x2+2x2, B into 2x3+2x3.
    #   2x2x3 + 2x2x3. Each 2x2x3 can be done with Strassen-like:
    #   Actually 2x2x3: use 2 Strassen 2x2x2 blocks? No, that's 14 for 2x2x4.
    #   2x2x3: rank is at least 6. Can we do it in 7?
    #   Yes: use 3x3 Strassen? No, that's for 3x3x3.
    #   For 2x2x3: we can use 2x2x2 (7) + 2x2x1 (4) = 11? No that's too much.
    #   Actually 2x2x3 can be done in 7? Let's think...
    #   The trivial rank is 2*2*3=12. Strassen gives 7 for 2x2x2.
    #   For 2x2x3: C[:,0:2] uses 7 terms, C[:,2] uses 4 terms. Total 11.
    #   Can we do better? Maybe 9 or 10?
    #
    # This is getting complex. Let me try a simpler optimization:
    # Check if the rank-44 solution can be reduced by finding a linear
    # dependency among the 44 "product vectors" in a generalized sense.
    #
    # For now, let's just return the rank-44 solution which we've verified.
    # The matching-based approach is too complex to implement correctly in time.
    
    if time.time() > deadline - 10:
        return best_result

    # Try one more thing: see if we can use a 3x3x4 decomposition
    # A is 3x4, B is 4x4, C is 3x4
    # Split B into cols 0,1,2 (3 cols) and col 3 (1 col)
    # Part 1: 3x4x3 (A * B[:,0:3] -> C[:,0:3])
    # Part 2: 3x4x1 (A * B[:,3] -> C[:,3])
    # Part 2 rank: 4 (one term per k)
    # Part 1: Split A rows into 0,1 and 2
    #   Top: 2x4x3. Split A into 2x2+2x2, B into 2x3+2x3
    #     2x2x3 + 2x2x3. Each 2x2x3: use 2x2x2 Strassen (7) for first 2 cols + 2x2x1 (4) for last col = 11
    #     Total top: 22
    #   Bottom: 1x4x3. Rank 12 (4 k-indices * 3 cols)
    #   Part 1 total: 22 + 12 = 34
    # Part 2 total: 4
    # Grand total: 38? Wait that seems too good. Let me recheck.
    #
    # Actually 1x4x3: C[2,j] = sum_k A[2,k]*B[k,j] for j=0,1,2
    # That's 4*3 = 12 terms. Yes.
    # 2x2x3: C[0:2, 0:3] = A[0:2,0:2]*B[0:2,0:3] + A[0:2,2:4]*B[2:4,0:3]
    # Each 2x2x3: for 2x2x2 part (cols 0,1): 7 terms. For 2x2x1 part (col 2): 4 terms. Total 11.
    # So 2x4x3 = 11 + 11 = 22.
    # 1x4x3 = 12.
    # 3x4x3 = 22 + 12 = 34.
    # 3x4x1 = 4.
    # Total = 38.
    #
    # Wait, but this can't be right because the trivial rank is 48 and we already have 44.
    # Let me recheck 2x2x3.
    # 2x2x3: A is 2x2, B is 2x3, C is 2x3.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    #
    # Using Strassen for cols 0,1 (2x2x2): 7 terms.
    # For col 2 (2x2x1): 4 terms (A[0,0]*B[0,2], A[0,1]*B[1,2], A[1,0]*B[0,2], A[1,1]*B[1,2]).
    # Total: 11. Yes.
    #
    # So the 3x4x4 split into 3x4x3 + 3x4x1 gives rank 34 + 4 = 38.
    # That would be a huge improvement! Let me implement and verify this.
    
    # Build 3x4x3 + 3x4x1 solution
    rank_new = 38
    U_new = [[0]*rank_new for _ in range(n*m)]
    V_new = [[0]*rank_new for _ in range(m*p)]
    W_new = [[0]*rank_new for _ in range(p*n)]
    col = 0

    # Part 1: 3x4x3 (cols 0,1,2 of C)
    # Top: 2x4x3 (rows 0,1 of A)
    #   Block 1: A[0:2,0:2] * B[0:2,0:3] -> C[0:2,0:3]
    #     2x2x3: Strassen for cols 0,1 + trivial for col 2
    col = add_strassen(U_new, V_new, W_new, 0, 0, 0, 0, 0, 0, col)  # 2x2x2 for C[0:2,0:2]
    # Add 2x2x1 for C[0:2,2]
    for a_r in range(2):
        for a_c in range(2):
            U_new[(a_r)*m + a_c][col] += 1
            V_new[(a_r)*p + a_c][col] += 1  # Wait, B row should be 0,1 not 0,1
            # Actually B[0:2, 2] means B rows 0,1, col 2
            # U accesses A[0:2, 0:2], V accesses B[0:2, 2]
            # So V row is 0 or 1, col is 2
            V_new[a_r*p + 2][col] = 1  # This is wrong, let me redo
            col += 1
    # I'm making errors. Let me be more careful.
    # For 2x2x1: A is 2x2 (rows ar, ar+1; cols ac, ac+1), B is 2x1 (rows br, br+1; col bc), C is 2x1 (rows cr, cr+1; col cc)
    # 4 terms: A[ar,ac]*B[br,bc] -> C[cr,cc], A[ar,ac+1]*B[br+1,bc] -> C[cr,cc], etc.
    
    # Let me just verify the 44-rank solution and return it. The 38-rank idea needs careful implementation.
    # Given time constraints, return the verified rank-44 solution.
    
    return best_result
