import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the block-Strassen decomposition (rank 189)
    block_size = 2
    num_blocks = 3
    
    # Strassen 2x2 combinations
    A_s0 = [(0,0,1), (1,1,1)]
    B_s0 = [(0,0,1), (1,1,1)]
    C_s0 = [(0,0,1), (1,1,1)]
    
    A_s1 = [(1,0,1), (1,1,1)]
    B_s1 = [(0,0,1)]
    C_s1 = [(1,0,1), (1,1,-1)]
    
    A_s2 = [(0,0,1)]
    B_s2 = [(0,1,1), (1,1,-1)]
    C_s2 = [(0,1,1), (1,1,1)]
    
    A_s3 = [(1,1,1)]
    B_s3 = [(1,0,1), (0,0,-1)]
    C_s3 = [(0,0,1), (1,0,1)]
    
    A_s4 = [(0,0,1), (0,1,1)]
    B_s4 = [(1,1,1)]
    C_s4 = [(0,0,-1), (0,1,1)]
    
    A_s5 = [(1,0,1), (0,0,-1)]
    B_s5 = [(0,0,1), (0,1,1)]
    C_s5 = [(1,1,1)]
    
    A_s6 = [(0,1,1), (1,1,-1)]
    B_s6 = [(1,0,1), (1,1,1)]
    C_s6 = [(0,0,1)]
    
    A_combs = [A_s0, A_s1, A_s2, A_s3, A_s4, A_s5, A_s6]
    B_combs = [B_s0, B_s1, B_s2, B_s3, B_s4, B_s5, B_s6]
    C_combs = [C_s0, C_s1, C_s2, C_s3, C_s4, C_s5, C_s6]
    
    rank = num_blocks * num_blocks * num_blocks * 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for i_b in range(num_blocks):
        for j_b in range(num_blocks):
            for k_b in range(num_blocks):
                for s in range(7):
                    col = ((i_b * num_blocks + j_b) * num_blocks + k_b) * 7 + s
                    
                    for rb, cb, coeff in A_combs[s]:
                        i_glob = i_b * block_size + rb
                        j_glob = k_b * block_size + cb
                        U[i_glob * m + j_glob, col] += coeff
                        
                    for rb, cb, coeff in B_combs[s]:
                        j_glob = k_b * block_size + rb
                        k_glob = j_b * block_size + cb
                        V[j_glob * p + k_glob, col] += coeff
                        
                    for rb, cb, coeff in C_combs[s]:
                        i_glob = i_b * block_size + rb
                        j_glob = j_b * block_size + cb
                        W[j_glob * n + i_glob, col] += coeff
    
    # Verify correctness
    def verify(U_mat, V_mat, W_mat):
        # Check that the tensor decomposition is correct
        # For random A, B, check if sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] == (A@B)[i,k]
        for _ in range(5):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(rank):
                            C_computed[i, k] += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                return False
        return True
    
    if not verify(U, V, W):
        # Fallback: return schoolbook (shouldn't happen)
        rank_sb = n * m * p
        U_sb = np.zeros((n*m, rank_sb))
        V_sb = np.zeros((m*p, rank_sb))
        W_sb = np.zeros((p*n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m+j, col] = 1.0
                    V_sb[j*p+k, col] = 1.0
                    W_sb[k*n+i, col] = 1.0
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Attempt rank reduction by finding linear dependencies
    # Build the tensor T[i,j,k,r] = U[i,j,r] * V[j,k,r] * W[k,i,r]
    # and check if columns are linearly dependent
    # For efficiency, work with a subset of (i,j,k) triples
    
    # Sample a set of (i,j,k) triples to check dependencies
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                triples.append((i, j, k))
    
    # Build matrix M where M[t, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for triple t
    M = np.zeros((len(triples), rank))
    for t, (i, j, k) in enumerate(triples):
        for r in range(rank):
            M[t, r] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Find the rank of M
    # If rank(M) < rank, there are linear dependencies
    try:
        U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
        # Count significant singular values
        tol = 1e-10
        num_significant = np.sum(s > tol)
        
        if num_significant < rank:
            # There are dependencies; try to find a reduced basis
            # Use the left singular vectors corresponding to significant singular values
            # to identify a subset of columns that spans the same space
            
            # Get the right singular vectors for significant components
            V_significant = Vt[:num_significant, :].T  # shape (rank, num_significant)
            
            # Find a subset of columns that are linearly independent
            # Use QR decomposition with column pivoting
            Q, R, piv = np.linalg.qr(M.T, mode='complete', pivoting=True)
            # piv gives the permutation of columns
            # Find the first num_significant pivoted columns
            independent_cols = np.sort(piv[:num_significant])
            
            if len(independent_cols) < rank:
                # Use only the independent columns
                U_reduced = U[:, independent_cols]
                V_reduced = V[:, independent_cols]
                W_reduced = W[:, independent_cols]
                
                # Verify the reduced decomposition
                if verify(U_reduced, V_reduced, W_reduced):
                    return {"U": U_reduced.tolist(), "V": V_reduced.tolist(), "W": W_reduced.tolist()}
    except Exception:
        pass
    
    # Return the original rank-189 decomposition
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
