"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        U = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, -1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, -1, 1, 0, 0],
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, -1, 0, 0],
        ]
        return U, V, W

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        """Faster verification using einsum."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    # Start with schoolbook baseline
    best_result = schoolbook()
    best_rank = n * m * p

    # Strategy: Use Strassen on 2x2x2 blocks for i in {0,1}, and for i=2 use
    # a rank-4 decomposition of the 1x4x5 tensor (using the 4 independent k-rows
    # shared with the V factor from the i=0,1 blocks).
    #
    # Key insight: For i=2, we need sum_c U[2,j,c]*V[j,k,c]*W[k,2,c] = 1 for all j,k.
    # If we set W[k,2,c] = delta_{k,c} for c in {0,1,2,3} and U[2,j,c] = delta_{j,c},
    # then we need V[j,c,c] = 1 for j=c. But V is shared!
    #
    # Alternative: Use rank 50. For i in {0,1}: 32 terms (as before).
    # For i=2: use 18 terms instead of 20 by sharing 2 terms with the k=4 schoolbook.
    # Actually, let's try a different approach: use the same V and W factors for i=2.
    #
    # Let's try: For i=2, use rank-4 terms where U[2,j,c]=delta_{j,c}, W[k,2,c]=delta_{k,c},
    # and we need V[j,c,c]=1. But V[j,c,c] is already set by the Strassen blocks.
    # This won't work directly.
    #
    # Instead, let's try to reduce from 52 to 50 by finding a better decomposition.
    # Use the following: for the 2x4x5 subproblem (i in {0,1}), instead of 32,
    # try to use 30 terms by a more clever Strassen application.
    #
    # Actually, let's just verify the 52-rank solution works and try small variations.

    U_cols = []
    V_cols = []
    W_cols = []

    # Helper to add a Strassen block for sub-problem at (i_offset, j_offset, k_offset)
    def add_strassen_block(i_off, j_off, k_off, target_cols=None):
        if target_cols is None:
            target_cols = (U_cols, V_cols, W_cols)
        uc, vc, wc = target_cols
        U_s, V_s, W_s = strassen_2x2x2()
        for c in range(7):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            for ii in range(2):
                for jj in range(2):
                    U_col[(i_off+ii)*m + (j_off+jj)] = U_s[ii*2+jj][c]
            for jj in range(2):
                for kk in range(2):
                    V_col[(j_off+jj)*p + (k_off+kk)] = V_s[jj*2+kk][c]
            for kk in range(2):
                for ii in range(2):
                    W_col[(k_off+kk)*n + (i_off+ii)] = W_s[kk*2+ii][c]
            uc.append(U_col)
            vc.append(V_col)
            wc.append(W_col)

    # Helper to add schoolbook terms for a sub-block
    def add_schoolbook_block(i_list, j_list, k_list, target_cols=None):
        if target_cols is None:
            target_cols = (U_cols, V_cols, W_cols)
        uc, vc, wc = target_cols
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U_col = [0.0] * (n * m)
                    V_col = [0.0] * (m * p)
                    W_col = [0.0] * (p * n)
                    U_col[i*m + j] = 1.0
                    V_col[j*p + k] = 1.0
                    W_col[k*n + i] = 1.0
                    uc.append(U_col)
                    vc.append(V_col)
                    wc.append(W_col)

    # For i in {0,1}, j in {0,1}: Strassen on k={0,1} and k={2,3}, schoolbook k=4
    add_strassen_block(0, 0, 0)  # covers i={0,1}, j={0,1}, k={0,1}
    add_strassen_block(0, 0, 2)  # covers i={0,1}, j={0,1}, k={2,3}
    add_schoolbook_block([0,1], [0,1], [4])  # k=4 for i={0,1}, j={0,1}

    # For i in {0,1}, j in {2,3}: Strassen on k={0,1} and k={2,3}, schoolbook k=4
    add_strassen_block(0, 2, 0)  # covers i={0,1}, j={2,3}, k={0,1}
    add_strassen_block(0, 2, 2)  # covers i={0,1}, j={2,3}, k={2,3}
    add_schoolbook_block([0,1], [2,3], [4])  # k=4 for i={0,1}, j={2,3}

    # For i in {0,1}, k=4: replace 8 schoolbook terms with 1 rank-1 term
    # T[i,j,4]=1 for i in {0,1}, j in {0,1,2,3} is a 2x4 all-ones matrix (rank 1)
    U_col = [0.0] * (n * m)
    V_col = [0.0] * (m * p)
    W_col = [0.0] * (p * n)
    for i in [0, 1]:
        for j in range(m):
            U_col[i*m + j] = 1.0
    for j in range(m):
        V_col[j*p + 4] = 1.0
    for i in [0, 1]:
        W_col[4*n + i] = 1.0
    # Remove the 8 schoolbook k=4 terms we just added and use this instead
    # Actually, let's not add them in the first place - we'll handle k=4 separately
    # This is a temporary approach; we'll rebuild properly below

    # For i=2: use a single rank-1 term since T[2,j,k]=1 is rank-1
    # U[2,j,c]=1 for all j, V[j,k,c]=1 for all j,k, W[k,2,c]=1 for all k
    U_col = [0.0] * (n * m)
    V_col = [0.0] * (m * p)
    W_col = [0.0] * (p * n)
    for j in range(m):
        U_col[2*m + j] = 1.0
    for j in range(m):
        for k in range(p):
            V_col[j*p + k] = 1.0
    for k in range(p):
        W_col[k*n + 2] = 1.0
    U_cols.append(U_col)
    V_cols.append(V_col)
    W_cols.append(W_col)

    # Convert to row-major format
    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]

    if verify_fast(U_mat, V_mat, W_mat, n, m, p):
        if rank < best_rank:
            best_result = {"U": U_mat, "V": V_mat, "W": W_mat}
            best_rank = rank

    # Now try to also reduce the k=4 terms for i in {0,1}
    # The k=4 slice: T[i,j,4] = 1 for i in {0,1}, j in {0,1,2,3}
    # This is a 2x4 all-ones matrix, which has rank 1.
    # We can use one term: U[i,j,c]=1 for i in {0,1}, all j; V[j,4,c]=1 for all j; W[4,i,c]=1 for i in {0,1}
    # But we need to make sure this doesn't interfere with other slices.
    # This term contributes to T[i,j,k] for all k via V[j,k,c]. If we set V[j,k,c]=0 for k!=4,
    # and W[k,i,c]=0 for k!=4, then it only affects k=4.
    
    # Let's build a second version with this optimization
    U_cols2 = list(U_cols)
    V_cols2 = list(V_cols)
    W_cols2 = list(W_cols)
    
    # Remove the 8 schoolbook k=4 terms for i in {0,1} and replace with 1 term
    # The schoolbook k=4 terms are at indices: after 28 strassen terms, the next 8 are k=4 for i in {0,1}
    # Let's rebuild from scratch to be safe
    
    U_cols2 = []
    V_cols2 = []
    W_cols2 = []
    
    # Strassen blocks (same as before)
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    
    # For i in {0,1}, k=4: use 1 term instead of 8
    U_col = [0.0] * (n * m)
    V_col = [0.0] * (m * p)
    W_col = [0.0] * (p * n)
    for i in [0, 1]:
        for j in range(m):
            U_col[i*m + j] = 1.0
    for j in range(m):
        V_col[j*p + 4] = 1.0
    for i in [0, 1]:
        W_col[4*n + i] = 1.0
    U_cols2.append(U_col)
    V_cols2.append(V_col)
    W_cols2.append(W_col)
    
    # For i=2: use 1 term (as above)
    U_col = [0.0] * (n * m)
    V_col = [0.0] * (m * p)
    W_col = [0.0] * (p * n)
    for j in range(m):
        U_col[2*m + j] = 1.0
    for j in range(m):
        for k in range(p):
            V_col[j*p + k] = 1.0
    for k in range(p):
        W_col[k*n + 2] = 1.0
    U_cols2.append(U_col)
    V_cols2.append(V_col)
    W_cols2.append(W_col)
    
    rank2 = len(U_cols2)
    U_mat2 = [[U_cols2[c][r] for c in range(rank2)] for r in range(n*m)]
    V_mat2 = [[V_cols2[c][r] for c in range(rank2)] for r in range(m*p)]
    W_mat2 = [[W_cols2[c][r] for c in range(rank2)] for r in range(p*n)]
    
    if verify_fast(U_mat2, V_mat2, W_mat2, n, m, p):
        if rank2 < best_rank:
            best_result = {"U": U_mat2, "V": V_mat2, "W": W_mat2}
            best_rank = rank2

    return best_result
# EVOLVE-BLOCK-END
