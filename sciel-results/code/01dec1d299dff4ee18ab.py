import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    deadline = time.time() + 145.0

    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def check_decomposition(U, V, W):
        """Check if sum_r U[:,r] * V[:,r] * W[:,r] == T (up to tolerance)"""
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-9)

    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

    def make_rank42():
        U = np.zeros((n*m, 42))
        V = np.zeros((m*p, 42))
        W = np.zeros((p*n, 42))
        
        blocks = [
            ([0, 1], [0, 1], [0, 1], [0, 1]),
            ([0, 1], [0, 1], [0, 1], [2, 3]),
            ([0, 1], [0, 1], [0, 1], [4, 5]),
            ([0, 1], [2, 3], [2, 3], [0, 1]),
            ([0, 1], [2, 3], [2, 3], [2, 3]),
            ([0, 1], [2, 3], [2, 3], [4, 5]),
        ]
        
        strassen = [
            ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
            ([(2,1),(3,1)], [(0,1)],       [(2,1),(3,-1)]),
            ([(0,1)],        [(1,1),(3,-1)],[(1,1),(3,1)]),
            ([(3,1)],        [(2,1),(0,-1)],[(0,1),(2,1)]),
            ([(0,1),(1,1)],  [(3,1)],       [(0,-1),(1,1)]),
            ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
            ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
        ]
        
        col = 0
        for a_rows, a_cols, b_rows, b_cols in blocks:
            a_map = {0: a_rows[0]*m+a_cols[0], 1: a_rows[0]*m+a_cols[1],
                     2: a_rows[1]*m+a_cols[0], 3: a_rows[1]*m+a_cols[1]}
            b_map = {0: b_rows[0]*p+b_cols[0], 1: b_rows[0]*p+b_cols[1],
                     2: b_rows[1]*p+b_cols[0], 3: b_rows[1]*p+b_cols[1]}
            w_map = {0: b_cols[0]*n+a_rows[0], 1: b_cols[1]*n+a_rows[0],
                     2: b_cols[0]*n+a_rows[1], 3: b_cols[1]*n+a_rows[1]}
            
            for u_c, v_c, w_c in strassen:
                for li, cf in u_c:
                    U[a_map[li], col] += cf
                for li, cf in v_c:
                    V[b_map[li], col] += cf
                for li, cf in w_c:
                    W[w_map[li], col] += cf
                col += 1
        
        return U, V, W
    
    U, V, W = make_rank42()
    
    # Verify the rank-42 decomposition is correct
    if not check_decomposition(U, V, W):
        # Fallback: schoolbook decomposition (rank = n*m*p = 48)
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, idx] = 1.0
                    V[j*p+k, idx] = 1.0
                    W[k*n+i, idx] = 1.0
                    idx += 1
        return to_lists(U, V, W)

    # Attempt to reduce rank by checking linear dependencies of the rank-1 tensors
    def try_reduce_rank(U, V, W):
        num_cols = U.shape[1]
        # Flatten each rank-1 tensor T_r = U[:,r] \otimes V[:,r] \otimes W[:,r]
        # Shape of flattened tensor: (n*m) * (m*p) * (p*n)
        flat_size = (n*m) * (m*p) * (p*n)
        
        # Matrix M of shape (flat_size, num_cols)
        M = np.zeros((flat_size, num_cols))
        
        for r in range(num_cols):
            # T_r[i, j, k] = U[i, r] * V[j, r] * W[k, r]
            # Flatten: index = (i * (m*p) + j) * (p*n) + k
            # We can use einsum to compute the outer product and reshape
            tensor_r = np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            M[:, r] = tensor_r.flatten()
            
        # Compute rank of M
        # Use SVD to find rank and basis
        try:
            U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            return U, V, W
        
        # Tolerance for rank determination
        tol = 1e-9 * max(M.shape) * s[0] if s.size > 0 else 1e-9
        rank = np.sum(s > tol)
        
        if rank == num_cols:
            return U, V, W
        
        # The columns of M are linearly dependent.
        # We want to find a set of `rank` columns that form a basis.
        # Then express the other columns as linear combinations of the basis.
        # Let B be the basis matrix (flat_size x rank)
        B = M[:, :rank] # This assumes the first 'rank' columns are independent, which is not always true.
        # Better: use the SVD to find a basis
        # The first 'rank' columns of U_svd form a basis for the column space.
        # But we need to express the *original* columns in terms of a subset of *original* columns.
        
        # Let's select a pivot set of columns from the original M.
        # We can use the QR decomposition with column pivoting.
        Q, R, P = np.linalg.qr(M, mode='complete') # NumPy doesn't have pivoted QR directly in all versions, 
        # but we can use scipy.linalg.qr with pivoting if available, or just check subsets.
        
        # Simpler: If rank < num_cols, we can try to remove columns one by one.
        # This is slow. 
        # Alternative: Just return the rank-42 solution if reduction is too complex.
        # Given the time, let's just return the 42 solution unless we find an easy reduction.
        
        # However, if rank is significantly lower (e.g., 40), it's worth it.
        # For now, let's just check if rank < 42 and try to find a minimal linearly independent subset.
        
        # Find a linearly independent subset of columns
        independent_cols = []
        basis = np.zeros((flat_size, 0))
        
        for r in range(num_cols):
            col = M[:, r].reshape(-1, 1)
            # Check if col is in the span of current basis
            # Project col onto basis
            if basis.shape[1] == 0:
                independent_cols.append(r)
                basis = col
            else:
                # Solve basis * x = col
                try:
                    x, residuals, rank_b, s_vals = np.linalg.lstsq(basis, col, rcond=None)
                    residual = np.linalg.norm(col - basis @ x)
                    if residual > tol:
                        independent_cols.append(r)
                        basis = np.hstack([basis, col])
                except np.linalg.LinAlgError:
                    independent_cols.append(r)
                    basis = np.hstack([basis, col])
            
            if len(independent_cols) == rank:
                break
        
        if len(independent_cols) == rank and rank < num_cols:
            # We have a set of independent columns.
            # We need to express the full tensor T as a sum of these rank-1 terms.
            # T = sum_{r in independent} c_r T_r + sum_{r not in independent} T_r
            # This is not how it works. T is fixed.
            # T = sum_{r=1}^{42} T_r.
            # If T_42 = sum_{r=1}^{41} d_r T_r, then T = sum_{r=1}^{41} (1+d_r) T_r.
            # We need to find d_r such that T_removed = sum d_r T_basis.
            
            # Let's just return the 42 solution for safety, as implementing the merge is complex and risky.
            pass

        return U, V, W

    U_reduced, V_reduced, W_reduced = try_reduce_rank(U, V, W)
    
    if check_decomposition(U_reduced, V_reduced, W_reduced):
        U, V, W = U_reduced, V_reduced, W_reduced

    return to_lists(U, V, W)
