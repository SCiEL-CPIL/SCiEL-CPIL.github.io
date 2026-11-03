import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = 2, 4, 6
    deadline = time.time() + 140.0
    
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
    
    def to_lists(U, V, W):
        return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    def check_decomposition(U, V, W):
        reconstructed = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.allclose(reconstructed, T, atol=1e-9)
    
    def reduce_rank(U, V, W):
        rank = U.shape[1]
        # Flatten tensors
        tensors = []
        for r in range(rank):
            t = U[:, r] * V[:, r] * W[:, r] # This is not the tensor, this is the contribution to the sum?
            # No, the tensor contribution is U[:,r] outer V[:,r] outer W[:,r]
            # We need to check linear independence of the tensors T_r
            # Flatten T_r
            T_r = np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).flatten()
            tensors.append(T_r)
        
        tensors = np.array(tensors)
        # Perform SVD or Rank Revealation to find independent columns
        # We want to find a subset of columns that spans the same space
        # Since the number of columns is small (42), we can use Gaussian elimination
        # on the rows of the matrix where each row is a tensor?
        # No, we want to find if any tensor is a linear combination of others.
        # Let M be the matrix with tensors as rows.
        # We want to find the rank of M.
        # If rank < 42, we can remove redundant rows.
        
        # Use SVD to find rank
        U_svd, s, Vt = np.linalg.svd(tensors, full_matrices=False)
        # Tolerance for rank
        tol = 1e-9
        rank_found = np.sum(s > tol)
        
        if rank_found < rank:
            # Keep the first rank_found right singular vectors?
            # No, we need to keep the original columns that are independent.
            # We can find the pivot columns of the row space.
            # Or simply, iterate and remove dependent columns.
            # A column r is dependent if T_r is in span of others.
            
            kept = []
            for r in range(rank):
                if time.time() > deadline - 10:
                    break
                # Check if T_r is in span of T_kept
                if not kept:
                    kept.append(r)
                    continue
                
                T_kept = tensors[kept]
                # Solve T_kept @ x = T_r
                # Use least squares
                x, residuals, rank_ls, s_ls = np.linalg.lstsq(T_kept, tensors[r], rcond=None)
                if residuals and residuals[0] < tol:
                    # Dependent
                    pass
                else:
                    kept.append(r)
            
            if len(kept) < rank:
                U = U[:, kept]
                V = V[:, kept]
                W = W[:, kept]
        
        return U, V, W

    U, V, W = reduce_rank(U, V, W)
    
    if not check_decomposition(U, V, W):
        U, V, W = make_rank42()
    
    return to_lists(U, V, W)
