"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10
    
    # Build the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i, j, k] = 1 if the entry C[i//m, k%p]... wait, let me get the indexing right.
    # A is n×m, indexed by (i, j) where i in [0,n), j in [0,m)
    # B is m×p, indexed by (j, k) where j in [0,m), k in [0,p)
    # C is n×p, indexed by (i, k) where i in [0,n), k in [0,p)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # In the tensor decomposition: C[i,k] = sum_c U[i*m+j... wait
    # U has shape (n*m, rank), indexed by (i*m + j, c) for A's entries
    # V has shape (m*p, rank), indexed by (j*p + k, c) for B's entries  
    # W has shape (p*n, rank), indexed by (k*n + i, c) for C's entries
    #
    # The tensor T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T[a,b,c] = 1 if i=i' and k=k' and j=j' (the middle index must match)
    # Wait, that's not right. Let me reconsider.
    #
    # The bilinear form is: C[i,k] = sum_j A[i,j] B[j,k]
    # In tensor form: C[i,k] = sum_{j} T[i,j,k] A[i,j] B[j,k] where T[i,j,k] = 1
    # But we need to map to the flattened indices.
    #
    # U row index: a = i*m + j (i in [0,n), j in [0,m))
    # V row index: b = j*p + k (j in [0,m), k in [0,p))
    # W row index: c = k*n + i (k in [0,p), i in [0,n)) -- note: k*n+i, not i*n+k
    #
    # The tensor T[a, b, c] = 1 iff:
    #   a = i*m + j, b = j'*p + k, c = k'*n + i'
    #   and i = i', k = k', j = j'
    # i.e., the A index (i,j), B index (j,k), C index (i,k) are consistent.
    
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def _check_exact(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T, with integer/half-integer entries."""
        # Check entries are integer or half-integer
        for name, mat in [('U', U), ('V', V), ('W', W)]:
            for val in mat:
                rounded = round(val * 2)
                if abs(val * 2 - rounded) > tol:
                    return False
        # Check decomposition
        rank = U.shape[1]
        for c in range(rank):
            pass
        # Reconstruct: T_recon[a,b,c] = sum_c U[a,c]*V[b,c]*W[c,c]
        # More efficiently: for each c, outer product
        T_recon = np.zeros_like(T)
        for c in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
        if np.max(np.abs(T_recon - T)) > tol:
            return False
        return True
    
    def _round_to_half_int(mat, tol=1e-6):
        """Round matrix entries to nearest half-integer."""
        return np.round(mat * 2) / 2.0
    
    def _als_decompose(T, rank, n_iter=200, tol=1e-8, seed=42):
        """ALS decomposition of tensor T into rank factors."""
        rng = np.random.RandomState(seed)
        nm, mp, pn = T.shape
        
        # Initialize
        U = rng.randn(nm, rank) * 0.1
        V = rng.randn(mp, rank) * 0.1
        W = rng.randn(pn, rank) * 0.1
        
        for it in range(n_iter):
            # Solve for U: for each a, U[a,:] minimizes ||T[a,:,:] - sum_c U[a,c] V[:,c] W[:,c]||
            # This is a linear system: for fixed a, 
            # T[a,:,:] = sum_c U[a,c] * (V[:,c] outer W[:,c])
            # Let M[a] be the matrix V outer W reshaped: M[a] has shape (mp*pn, rank)
            # Actually: T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
            # For fixed a: T[a,:,:] = sum_d U[a,d] * (V[:,d] * W[:,d])^T reshaped
            # So T[a, :, :] = sum_d U[a,d] * outer(V[:,d], W[:,d])
            # Vectorize: T[a].flatten() = sum_d U[a,d] * (V[:,d] * W[:,d]).flatten()
            # So we solve: T_a_flat = U[a,:] @ (V*W)_flat^T where (V*W)_flat is rank x (mp*pn)
            
            # Build matrix M_U of shape (mp*pn, rank) where M_U[b*pn+c, d] = V[b,d]*W[c,d]
            M_U = np.einsum('bd,cd->bcd', V, W).reshape(mp * pn, rank)
            # Solve for each a: T[a].flatten() = M_U^T @ U[a,:]
            # U[a,:] = (M_U^T @ M_U)^{-1} M_U^T @ T[a].flatten()
            try:
                gram = M_U.T @ M_U
                rhs = M_U.T @ T.reshape(nm, -1).T  # shape (rank, nm)
                U_new = np.linalg.solve(gram, rhs).T  # shape (nm, rank)
            except np.linalg.LinAlgError:
                U_new = np.linalg.lstsq(M_U.T, T.reshape(nm, -1).T, rcond=None)[0].T
            
            # Solve for V: for each b, T[:,b,:] = sum_d V[b,d] * (U[:,d] * W[:,d])^T
            M_V = np.einsum('ad,cd->acd', U, W).reshape(nm * pn, rank)
            try:
                gram = M_V.T @ M_V
                rhs = M_V.T @ T.reshape(-1, mp).T.reshape(mp, -1).T  # need to be careful
                # T[:, b, :] flattened is shape (nm*pn,)
                rhs = M_V.T @ T.reshape(-1, mp).T  # (rank, mp)
                V_new = np.linalg.solve(gram, rhs).T  # (mp, rank)
            except np.linalg.LinAlgError:
                V_new = np.linalg.lstsq(M_V.T, T.reshape(-1, mp).T, rcond=None)[0].T
            
            # Solve for W: for each c, T[:,:,c] = sum_d W[c,d] * (U[:,d] * V[:,d])^T
            M_W = np.einsum('ad,bd->abd', U, V).reshape(nm * mp, rank)
            try:
                gram = M_W.T @ M_W
                rhs = M_W.T @ T.reshape(-1, pn).T  # (rank, pn)
                W_new = np.linalg.solve(gram, rhs).T  # (pn, rank)
            except np.linalg.LinAlgError:
                W_new = np.linalg.lstsq(M_W.T, T.reshape(-1, pn).T, rcond=None)[0].T
            
            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            if it % 10 == 0:
                T_recon = np.einsum('ad,bd,cd->abc', U, V, W)
                err = np.max(np.abs(T_recon - T))
                if err < tol:
                    break
        
        return U, V, W
    
    def _try_rank(rank, max_seeds=10):
        """Try to find a rank-r decomposition."""
        for seed in range(max_seeds):
            U, V, W = _als_decompose(T, rank, n_iter=300, tol=1e-10, seed=seed)
            # Round to half-integers
            U_r = _round_to_half_int(U)
            V_r = _round_to_half_int(V)
            W_r = _round_to_half_int(W)
            if _check_exact(U_r, V_r, W_r, tol=1e-6):
                return U_r, V_r, W_r
        return None
    
    # First, return schoolbook as fallback
    def _schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Try to find lower rank
    best_result = _schoolbook()
    for rank in range(30, 1, -1):  # try from 29 down to 2
        if time.time() > deadline - 30:
            break
        result = _try_rank(rank)
        if result is not None:
            best_result = result
            break  # Found the minimum rank
    
    U, V, W = best_result
    # Convert to lists
    U_list = [[float(U[i, j]) for j in range(U.shape[1])] for i in range(U.shape[0])]
    V_list = [[float(V[i, j]) for j in range(V.shape[1])] for i in range(V.shape[0])]
    W_list = [[float(W[i, j]) for j in range(W.shape[1])] for i in range(W.shape[0])]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END

