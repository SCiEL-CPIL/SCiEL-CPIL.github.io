"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 blocks and schoolbook for smaller blocks.
Then applies fast peeling to remove linearly dependent rank-1 terms.
Target rank: 78 (possibly reducible).
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    """
    start_time = time.time()
    deadline = 145.0

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_222_local(i0, j0, k0):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1: (A11+A22)*(B11+B22)
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])

        # M2: (A21+A22)*B11
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])

        # M3: A11*(B12-B22)
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])

        # M4: A22*(B21-B11)
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])

        # M5: (A11+A12)*B22
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])

        # M6: (A21-A11)*(B11+B12)
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])

        # M7: (A12-A22)*(B21+B22)
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def m127_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x7 block.
        Rank 14.
        """
        for j in range(2):
            for k in range(7):
                jj = j0 + j
                kk = k0 + k
                add_rank1_term([i0*m+jj], [1.0],
                               [jj*p+kk], [1.0],
                               [kk*n+i0], [1.0])

    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block.
        Rank 25.
        """
        strassen_222_local(i0, j0, k0)
        strassen_222_local(i0, j0, k0 + 2)
        strassen_222_local(i0, j0, k0 + 4)
        k6 = k0 + 6
        for i in range(2):
            for j in range(2):
                ii = i0 + i
                jj = j0 + j
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k6], [1.0],
                               [k6*n+ii], [1.0])

    def m147_decomp(i0, j0, k0):
        """
        Decomposition for 1x4 * 4x7 block.
        Uses Strassen-like approach: split m=4 into 2+2.
        For each pair of j's, use rank-14 schoolbook (no improvement possible
        for 1xm * mxp since rank = m*p).
        Actually 1x4*4x7 has rank 28 (schoolbook). No improvement.
        """
        m127_decomp(i0, j0, k0)
        m127_decomp(i0, j0+2, k0)

    def m247_decomp(i0, j0, k0):
        """
        Decomposition for 2x4 * 4x7 block.
        Split m=4 into 2+2. Two 2x2x7 blocks.
        """
        m227_decomp(i0, j0, k0)
        m227_decomp(i0, j0+2, k0)

    def m347_decomp():
        """
        Full 3x4 * 4x7 decomposition.
        Split n=3 into 1+2.
        """
        # n=1 part (i=0): 1x4 * 4x7
        m147_decomp(0, 0, 0)
        
        # n=2 part (i=1,2): 2x4 * 4x7
        m247_decomp(1, 0, 0)

    m347_decomp()

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    # Fast peeling using SVD-based rank detection
    def try_peel_fast(U_mat, V_mat, W_mat):
        """
        Remove linearly dependent rank-1 terms using SVD on the tensor matrix.
        Build M[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        Then find columns that are in the span of others.
        """
        U_curr = U_mat.copy()
        V_curr = V_mat.copy()
        W_curr = W_mat.copy()
        n_terms = U_curr.shape[1]
        
        if n_terms <= 1:
            return U_curr, V_curr, W_curr
        
        total_entries = n * m * p
        
        # Build tensor matrix M: (total_entries, n_terms)
        # M[idx, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        M = np.zeros((total_entries, n_terms))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * (m * p) + j * p + k
                    M[idx, :] = U_curr[i*m+j, :] * V_curr[j*p+k, :] * W_curr[k*n+i, :]
        
        # Use SVD to find the rank and identify dependent columns
        # A column is redundant if it's in the span of the others.
        # Efficient approach: iteratively check using QR with column pivoting.
        
        # First pass: use SVD to get the numerical rank
        try:
            s, U_svd, Vt = np.linalg.svd(M, full_matrices=False)
            tol = max(M.shape) * np.finfo(float).eps * s[0] if len(s) > 0 else 1e-10
            rank_M = np.sum(s > tol)
        except:
            return U_curr, V_curr, W_curr
        
        if rank_M >= n_terms:
            # All columns are linearly independent
            return U_curr, V_curr, W_curr
        
        # Now find which columns to remove.
        # Use QR with column pivoting to identify the "important" columns.
        # The columns NOT selected by pivoting are the dependent ones.
        try:
            # QR with column pivoting
            Q, R, P = np.linalg.qr(M, mode='complete')
            # R is upper triangular; columns where R[k,k] is small are dependent
            # Actually, let's use a different approach:
            # Greedy: for each column, check if it's in span of previously kept columns
            
            kept = []
            removed = set()
            basis = np.zeros((total_entries, 0))
            
            for r in range(n_terms):
                if time.time() > deadline - 10:
                    break
                    
                if len(kept) == 0:
                    kept.append(r)
                    basis = M[:, [r]].copy()
                    continue
                
                # Check if column r is in span of kept columns
                A = M[:, kept]
                b = M[:, r]
                
                # Use least squares
                try:
                    coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                    residual = np.linalg.norm(A @ coeffs - b)
                    if residual < 1e-8 * max(1.0, np.linalg.norm(b)):
                        removed.add(r)
                    else:
                        kept.append(r)
                        basis = np.column_stack([basis, b])
                except:
                    kept.append(r)
                    basis = np.column_stack([basis, b])
            
            if not removed:
                return U_curr, V_curr, W_curr
            
            keep_list = [r for r in range(n_terms) if r not in removed]
            return U_curr[:, keep_list], V_curr[:, keep_list], W_curr[:, keep_list]
            
        except:
            return U_curr, V_curr, W_curr

    # Apply peeling
    U, V, W = try_peel_fast(U, V, W)

    rank = U.shape[1]
    
    # Final verification
    valid = True
    for i in range(n):
        if not valid:
            break
        for j in range(m):
            if not valid:
                break
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-9:
                    valid = False
                    break
    
    if not valid:
        # Fall back to original
        U_orig = np.array(U_vecs).T
        V_orig = np.array(V_vecs).T
        W_orig = np.array(W_vecs).T
        return {"U": U_orig.tolist(), "V": V_orig.tolist(), "W": W_orig.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
