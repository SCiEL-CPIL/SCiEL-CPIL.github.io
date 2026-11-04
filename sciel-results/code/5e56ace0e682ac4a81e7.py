import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n  # 8, 24, 12

    # Build the matmul tensor as a 3D array T[a,b,c]
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try to find a low-rank decomposition
    # Reshape T to a matrix and use SVD to find approximate rank
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # This is equivalent to: for each c, the matrix T[:,:,c] has rank <= rank
    
    # Let's try different ranks and check if we can find an exact decomposition
    # First, get the numerical rank of the unfolded matrices
    
    # Unfold along c: matrix of shape (nm*mp, pn)
    T_c = T.reshape(nm * mp, pn)
    # SVD to find numerical rank
    U_svd, s_vals, Vt_svd = np.linalg.svd(T_c, full_matrices=False)
    # The numerical rank is the number of significant singular values
    tol = 1e-10
    num_rank = np.sum(s_vals > tol * s_vals[0])
    
    # For exact decomposition, we need to find integer/half-integer coefficients
    # Let's try to find a decomposition of rank r for various r
    
    best_rank = None
    best_U = None
    best_V = None
    best_W = None
    
    # Try ranks from a lower bound up
    # Lower bound: max(nm, mp, pn) = max(8, 24, 12) = 24
    # But the tensor rank can be higher than the matrix rank of unfoldings
    
    # Let's try to find an exact decomposition using a greedy approach
    # or by solving a system of linear equations
    
    # Alternative: use the known Strassen construction and look for dependencies
    
    # First, let's construct the rank-42 Strassen decomposition
    U42 = np.zeros((nm, 42), dtype=float)
    V42 = np.zeros((mp, 42), dtype=float)
    W42 = np.zeros((pn, 42), dtype=float)
    
    def add_strassen_block(i_start, j_start, k_start, offset, U_mat, V_mat, W_mat):
        a00 = (i_start + 0) * m + (j_start + 0)
        a01 = (i_start + 0) * m + (j_start + 1)
        a10 = (i_start + 1) * m + (j_start + 0)
        a11 = (i_start + 1) * m + (j_start + 1)

        b00 = (j_start + 0) * p + (k_start + 0)
        b01 = (j_start + 0) * p + (k_start + 1)
        b10 = (j_start + 1) * p + (k_start + 0)
        b11 = (j_start + 1) * p + (k_start + 1)

        w00 = (k_start + 0) * n + (i_start + 0)
        w01 = (k_start + 1) * n + (i_start + 0)
        w10 = (k_start + 0) * n + (i_start + 1)
        w11 = (k_start + 1) * n + (i_start + 1)

        # M1: (A00+A11)(B00+B11)
        U_mat[a00, offset] += 1.0
        U_mat[a11, offset] += 1.0
        V_mat[b00, offset] += 1.0
        V_mat[b11, offset] += 1.0
        W_mat[w00, offset] += 1.0
        W_mat[w11, offset] += 1.0

        # M2: (A10+A11)B00
        U_mat[a10, offset + 1] += 1.0
        U_mat[a11, offset + 1] += 1.0
        V_mat[b00, offset + 1] += 1.0
        W_mat[w10, offset + 1] += 1.0
        W_mat[w11, offset + 1] -= 1.0

        # M3: A00(B01-B11)
        U_mat[a00, offset + 2] += 1.0
        V_mat[b01, offset + 2] += 1.0
        V_mat[b11, offset + 2] -= 1.0
        W_mat[w01, offset + 2] += 1.0
        W_mat[w11, offset + 2] += 1.0

        # M4: A11(B10-B00)
        U_mat[a11, offset + 3] += 1.0
        V_mat[b10, offset + 3] += 1.0
        V_mat[b00, offset + 3] -= 1.0
        W_mat[w00, offset + 3] += 1.0
        W_mat[w10, offset + 3] += 1.0

        # M5: (A00+A01)B11
        U_mat[a00, offset + 4] += 1.0
        U_mat[a01, offset + 4] += 1.0
        V_mat[b11, offset + 4] += 1.0
        W_mat[w00, offset + 4] -= 1.0
        W_mat[w01, offset + 4] += 1.0

        # M6: (A10-A00)(B00+B01)
        U_mat[a10, offset + 5] += 1.0
        U_mat[a00, offset + 5] -= 1.0
        V_mat[b00, offset + 5] += 1.0
        V_mat[b01, offset + 5] += 1.0
        W_mat[w11, offset + 5] += 1.0

        # M7: (A01-A11)(B10+B11)
        U_mat[a01, offset + 6] += 1.0
        U_mat[a11, offset + 6] -= 1.0
        V_mat[b10, offset + 6] += 1.0
        V_mat[b11, offset + 6] += 1.0
        W_mat[w00, offset + 6] += 1.0

    offset = 0
    for i_start in range(0, n, 2):
        for j_start in range(0, m, 2):
            for k_start in range(0, p, 2):
                add_strassen_block(i_start, j_start, k_start, offset, U42, V42, W42)
                offset += 7

    # Now try to find linear dependencies among the 42 terms
    # A term r is a tensor T_r = U42[:,r] ⊗ V42[:,r] ⊗ W42[:,r]
    # We want to find if T_r can be expressed as a linear combination of other terms
    
    # Flatten each term to a vector
    terms = []
    for r in range(42):
        term_vec = np.zeros(nm * mp * pn)
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    term_vec[a * mp * pn + b * pn + c] = U42[a, r] * V42[b, r] * W42[c, r]
        terms.append(term_vec)
    
    # Check if the 42 term vectors are linearly independent
    # Build a matrix with terms as columns
    terms_mat = np.column_stack(terms)
    
    # Find the rank of this matrix using SVD
    U_rank, s_rank, Vt_rank = np.linalg.svd(terms_mat, full_matrices=False)
    tol_rank = 1e-10 * s_rank[0] if len(s_rank) > 0 else 1e-10
    numerical_rank = np.sum(s_rank > tol_rank)
    
    # Use SVD to find a minimal set of linearly independent terms
    # The right singular vectors (Vt_rank) give us the basis in the column space
    # We need to find which original terms form a basis
    if numerical_rank < 42:
        # Use QR decomposition with column pivoting to find a minimal basis
        # This is more robust than greedy for this problem
        from scipy.linalg import svd
        # Use the left singular vectors to identify independent columns
        # A column is independent if it has significant weight in the first 'numerical_rank' left singular vectors
        
        # Alternative: use the fact that if we have the SVD, we can find
        # a subset of columns that spans the same space
        # Use a greedy approach but with pre-computed projections for speed
        
        # Pre-compute the projection matrix
        basis_left = U_rank[:, :numerical_rank]  # shape (2304, numerical_rank)
        
        # For each term, compute its coordinates in the basis
        # term_r ≈ basis_left @ (basis_left^T @ term_r)
        # If the reconstruction is good, the term is in the span
        
        # More efficient: use the right singular vectors
        # Vt_rank has shape (numerical_rank, 42)
        # Each column of Vt_rank^T tells us how much each term contributes to each singular direction
        
        # Find a minimal set using the structure of Vt_rank
        # A term is "important" if it has large norm in the Vt_rank space
        
        # Use a simple greedy but vectorized:
        selected = []
        remaining = list(range(42))
        
        while len(selected) < numerical_rank and remaining:
            # Find the term in remaining that is most "independent" from selected
            if not selected:
                # Just pick the first one
                sel_idx = remaining[0]
            else:
                # For each remaining term, compute how well it's explained by selected
                sel_terms = np.column_stack([terms_mat[:, s] for s in selected])
                # Project each remaining term onto the span of selected
                # Use least squares: for each remaining term r, compute residual norm
                best_sel = -1
                best_residual = float('inf')
                for r in remaining:
                    # Compute residual of term r after projecting onto selected
                    # Use the already computed SVD structure
                    x = np.linalg.lstsq(sel_terms, terms_mat[:, r], rcond=None)[0]
                    recon = sel_terms @ x
                    residual = np.linalg.norm(terms_mat[:, r] - recon)
                    if residual < best_residual:
                        best_residual = residual
                        best_sel = r
                sel_idx = best_sel
            
            selected.append(sel_idx)
            remaining.remove(sel_idx)
        
        new_rank = len(selected)
        
        # Build the new U, V, W matrices
        U_new = np.zeros((nm, new_rank), dtype=float)
        V_new = np.zeros((mp, new_rank), dtype=float)
        W_new = np.zeros((pn, new_rank), dtype=float)
        
        for new_r, old_r in enumerate(selected):
            U_new[:, new_r] = U42[:, old_r]
            V_new[:, new_r] = V42[:, old_r]
            W_new[:, new_r] = W42[:, old_r]
        
        # Verify with vectorized check
        def verify_decomp(U_mat, V_mat, W_mat, rank_val, tol=1e-8):
            # Vectorized verification
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    for k in range(p):
                        b = j * p + k
                        c = k * n + i
                        s = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                        if abs(s - 1.0) > tol:
                            return False
            return True
        
        if verify_decomp(U_new, V_new, W_new, new_rank):
            best_rank = new_rank
            best_U = U_new
            best_V = V_new
            best_W = W_new
        else:
            # Fall back to rank 42
            best_rank = 42
            best_U = U42
            best_V = V42
            best_W = W42
    else:
        # No dependencies found, use rank 42
        best_rank = 42
        best_U = U42
        best_V = V42
        best_W = W42

    # If we found a better rank, use it; otherwise use 42
    if best_rank is None:
        best_rank = 42
        best_U = U42
        best_V = V42
        best_W = W42

    # Verify the final decomposition
    def verify_final(U_mat, V_mat, W_mat, rank_val, tol=1e-8):
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for k in range(p):
                    b = j * p + k
                    c = k * n + i
                    s = np.dot(U_mat[a, :], V_mat[b, :] * W_mat[c, :])
                    if abs(s - 1.0) > tol:
                        return False
        return True

    if not verify_final(best_U, best_V, best_W, best_rank):
        # Fall back to schoolbook rank 48
        best_rank = n * m * p
        best_U = np.zeros((nm, best_rank), dtype=float)
        best_V = np.zeros((mp, best_rank), dtype=float)
        best_W = np.zeros((pn, best_rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * (m * p) + j * p + k
                    best_U[i * m + j, r] = 1
                    best_V[j * p + k, r] = 1
                    best_W[k * n + i, r] = 1

    # Check time
    if time.time() > deadline - 5:
        # Return immediately with what we have
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
