"""Rewritten program for matrix multiplication 3x4x5 tensor decomposition.
Uses Strassen block recursion with circuit sparsification for rank reduction.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
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
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    def tensor_contribution(U_col, V_col, W_col, n, m, p):
        """Compute the tensor contribution of a single rank term."""
        U_arr = np.array(U_col, dtype=float).reshape(n, m)
        V_arr = np.array(V_col, dtype=float).reshape(m, p)
        W_arr = np.array(W_col, dtype=float).reshape(p, n)
        # The contribution is U_col[i,j] * V_col[j,k] * W_col[k,i] for each (i,j,k)
        # This is a rank-1 tensor in the space of (n,m,p)-tensors
        contrib = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib[i, j, k] = U_arr[i, j] * V_arr[j, k] * W_arr[k, i]
        return contrib

    def try_prune_term(U_cols, V_cols, W_cols, term_idx, n, m, p, tol=1e-9):
        """
        Try to prune term_idx by expressing its contribution as a linear combination
        of other terms. Returns (success, new_U_cols, new_V_cols, new_W_cols).
        """
        rank = len(U_cols)
        if rank <= 1:
            return False, U_cols, V_cols, W_cols
        
        # Compute the contribution of the term to remove
        target = tensor_contribution(U_cols[term_idx], V_cols[term_idx], W_cols[term_idx], n, m, p)
        
        # Build a system: find coefficients alpha_c such that
        # sum_c alpha_c * contrib_c = target, where c != term_idx
        # We'll use least squares to find if this is possible
        
        other_indices = [c for c in range(rank) if c != term_idx]
        if not other_indices:
            return False, U_cols, V_cols, W_cols
        
        # Build matrix A where A[:, c] = flatten(contrib_c)
        A = np.zeros((n * m * p, len(other_indices)), dtype=float)
        for idx, c in enumerate(other_indices):
            contrib = tensor_contribution(U_cols[c], V_cols[c], W_cols[c], n, m, p)
            A[:, idx] = contrib.flatten()
        
        b = target.flatten()
        
        # Solve least squares
        try:
            coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=tol)
            # Check if the solution is good
            reconstruction = A @ coeffs
            if np.allclose(reconstruction, b, atol=tol, rtol=0):
                # Found a valid representation. Now we need to update the matrices.
                # The new matrices will have the same number of terms, but term_idx is removed.
                # Actually, we're expressing term_idx as a combination of others, so we can
                # just remove term_idx and adjust the others.
                
                # Wait - this approach doesn't directly work because we need to maintain
                # the structure where each term has its own U, V, W vectors.
                # Let's use a different approach: check if the term is linearly dependent
                # in a simpler way.
                pass
        except:
            pass
        
        return False, U_cols, V_cols, W_cols

    def sparsify_decomposition(U_cols, V_cols, W_cols, n, m, p, deadline, tol=1e-9):
        """
        Iteratively try to prune terms from the decomposition.
        Uses a greedy approach: try removing each term and check if the
        remaining terms still span the tensor.
        """
        U_cols = [list(x) for x in U_cols]
        V_cols = [list(x) for x in V_cols]
        W_cols = [list(x) for x in W_cols]
        
        rank = len(U_cols)
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            for term_idx in range(len(U_cols)):
                if time.time() > deadline - 5.0:
                    break
                # Try to see if this term is redundant
                # A term is redundant if its contribution can be expressed as
                # a linear combination of the other terms' contributions
                rank = len(U_cols)
                if rank <= 1:
                    break
                    
                other_indices = [c for c in range(rank) if c != term_idx]
                if not other_indices:
                    continue
                
                # Build the contribution matrix
                target = tensor_contribution(U_cols[term_idx], V_cols[term_idx], W_cols[term_idx], n, m, p)
                A = np.zeros((n * m * p, len(other_indices)), dtype=float)
                for idx, c in enumerate(other_indices):
                    contrib = tensor_contribution(U_cols[c], V_cols[c], W_cols[c], n, m, p)
                    A[:, idx] = contrib.flatten()
                
                b = target.flatten()
                
                # Check if b is in the column space of A
                # Use rank check
                A_aug = np.column_stack([A, b.reshape(-1, 1)])
                rank_A = np.linalg.matrix_rank(A, tol=tol)
                rank_A_aug = np.linalg.matrix_rank(A_aug, tol=tol)
                
                if rank_A == rank_A_aug:
                    # The term is redundant - remove it
                    U_cols.pop(term_idx)
                    V_cols.pop(term_idx)
                    W_cols.pop(term_idx)
                    changed = True
                    break
        
        return U_cols, V_cols, W_cols

    # Strategy: Build the 3x4x5 tensor using Strassen on 2x2x2 blocks.
    U_cols = []
    V_cols = []
    W_cols = []

    U_s, V_s, W_s = strassen_2x2x2()

    # For i in {0,1}, j in {0,1}, k in {0,1}: Strassen
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + j] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[j*p + k] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[k*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # For i in {0,1}, j in {2,3}, k in {0,1}: Strassen
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + (2+j)] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[(2+j)*p + k] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[k*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # For i in {0,1}, j in {0,1}, k in {2,3}: Strassen
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + j] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[j*p + (2+k)] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[(2+k)*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # For i in {0,1}, j in {2,3}, k in {2,3}: Strassen
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        for i in range(2):
            for j in range(2):
                U_col[i*m + (2+j)] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[(2+j)*p + (2+k)] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[(2+k)*n + i] = W_s[k*2+i][c]
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    # For i in {0,1}, all j, k=4: schoolbook
    for i in range(2):
        for j in range(m):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[i*m + j] = 1.0
            V_col[j*p + 4] = 1.0
            W_col[4*n + i] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # For i=2, all j, k: schoolbook
    for j in range(m):
        for k in range(p):
            U_col = [0.0] * (n * m)
            V_col = [0.0] * (m * p)
            W_col = [0.0] * (p * n)
            U_col[2*m + j] = 1.0
            V_col[j*p + k] = 1.0
            W_col[k*n + 2] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # Apply circuit sparsification
    U_cols, V_cols, W_cols = sparsify_decomposition(U_cols, V_cols, W_cols, n, m, p, deadline)

    rank = len(U_cols)
    U_mat = [[U_cols[c][r] for c in range(rank)] for r in range(n*m)]
    V_mat = [[V_cols[c][r] for c in range(rank)] for r in range(m*p)]
    W_mat = [[W_cols[c][r] for c in range(rank)] for r in range(p*n)]

    best_result = {"U": U_mat, "V": V_mat, "W": W_mat}
    best_rank = rank

    if verify_fast(U_mat, V_mat, W_mat, n, m, p):
        pass
    else:
        # Fallback to schoolbook if verification fails
        schoolbook_rank = n * m * p
        U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
        V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
        W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j][col] = 1
                    V_sb[j*p+k][col] = 1
                    W_sb[k*n+i][col] = 1
                    col += 1
        best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
        best_rank = schoolbook_rank

    return best_result
# EVOLVE-BLOCK-END
