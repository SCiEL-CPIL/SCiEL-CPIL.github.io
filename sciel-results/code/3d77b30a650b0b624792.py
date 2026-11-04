"""Tensor decomposition for 3x4 @ 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, atol=1e-8):
    """Check if the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=atol)

def _als_decompose(n, m, p, rank, max_iter=200, tol=1e-10, seed=42):
    """Alternating Least Squares for 3-tensor factorization.
    
    Find U (n*m x rank), V (m*p x rank), W (p*n x rank) such that:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    rng = np.random.RandomState(seed)
    
    # Initialize
    U = rng.randn(n * m, rank) * 0.1
    V = rng.randn(m * p, rank) * 0.1
    W = rng.randn(p * n, rank) * 0.1
    
    # Precompute index mappings
    # For each (i,j,k), we need U[i*m+j,:], V[j*p+k,:], W[k*n+i,:]
    # ALS: fix V and W, solve for U; fix U and W, solve for V; fix U and V, solve for W
    
    for iteration in range(max_iter):
        # --- Solve for U ---
        # For each (i,j): sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c] summed over k) = 1
        # Let G_U[i*m+j, c] = sum_k V[j*p+k, c] * W[k*n+i, c]
        # Then we need: U[i*m+j, :] @ G_U[i*m+j, :]^T ... 
        # Actually: for fixed (i,j), the equation is:
        #   sum_c U[i*m+j,c] * H[i,j,c] = 1 where H[i,j,c] = sum_k V[j*p+k,c]*W[k*n+i,c]
        # This is a linear system: U[i*m+j, :] @ H[i,j, :]^T = 1 (scalar)
        # But we have one equation per (i,j) and rank unknowns. Underdetermined if rank > 1.
        # We need to use all (i,j,k) equations together.
        
        # Better approach: vectorize. The full tensor equation is:
        # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # 
        # Fix V and W. For each c, define:
        #   A_c[i,j,k] = V[j*p+k,c] * W[k*n+i,c]
        # Then: sum_c U[i*m+j,c] * A_c[i,j,k] = 1
        #
        # This is linear in U. Let's set up the linear system.
        # Unknowns: U flattened = (n*m*rank,)
        # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # 
        # Let's build the matrix. Rows: n*m*p equations. Cols: n*m*rank unknowns.
        # This is 72 x 36 = 2592 for rank=6... too big for large rank.
        
        # Alternative: use the structure. For fixed (i,j), the equations over k are:
        #   sum_c U[i*m+j,c] * (sum_k' V[j*p+k',c] * W[k'*n+i,c] * delta_{k',k}) ... 
        # No, k is free. For fixed (i,j):
        #   For each k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Let B[i,j,k,c] = V[j*p+k,c] * W[k*n+i,c]
        # Then: sum_c U[i*m+j,c] * B[i,j,k,c] = 1 for each k
        # This is: U[i*m+j, :] @ B[i,j,:,:]^T = ones(p)
        # Where B[i,j,:,:] is (p, rank)
        # So: for each (i,j), we have a (p x rank) matrix B_ij, and we want U[i*m+j,:] (1 x rank) such that U[i*m+j,:] @ B_ij^T = ones(p)
        # If p >= rank, this is overdetermined. Use least squares.
        # If p < rank, underdetermined. Use minimum norm solution.
        
        U_new = np.zeros_like(U)
        for i in range(n):
            for j in range(m):
                # Build B_ij: shape (p, rank)
                # B_ij[k, c] = V[j*p+k, c] * W[k*n+i, c]
                B_ij = np.zeros((p, rank))
                for k in range(p):
                    B_ij[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve: U[i*m+j, :] @ B_ij.T = ones(p)
                # i.e., B_ij @ U[i*m+j, :].T = ones(p)
                # Use lstsq
                try:
                    sol, _, _, _ = np.linalg.lstsq(B_ij, np.ones(p), rcond=None)
                    U_new[i*m+j, :] = sol
                except np.linalg.LinAlgError:
                    U_new[i*m+j, :] = U[i*m+j, :]
        
        # --- Solve for V ---
        V_new = np.zeros_like(V)
        for j in range(m):
            for k in range(p):
                # For fixed (j,k): sum_c V[j*p+k,c] * (sum_i U[i*m+j,c] * W[k*n+i,c]) = 1
                # Let C_jk[c] = sum_i U[i*m+j,c] * W[k*n+i,c]
                # Then: V[j*p+k, :] @ C_jk = 1 (scalar)
                # But we need all (i,j,k) equations. For fixed (j,k), varying i:
                #   For each i: sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1
                # Let D[j,k,i,c] = U[i*m+j,c] * W[k*n+i,c]
                # Then: sum_c V[j*p+k,c] * D[j,k,i,c] = 1 for each i
                # V[j*p+k, :] @ D[j,k,:,:].T = ones(n)
                D_jk = np.zeros((n, rank))
                for i in range(n):
                    D_jk[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(D_jk, np.ones(n), rcond=None)
                    V_new[j*p+k, :] = sol
                except np.linalg.LinAlgError:
                    V_new[j*p+k, :] = V[j*p+k, :]
        
        # --- Solve for W ---
        W_new = np.zeros_like(W)
        for k in range(p):
            for i in range(n):
                # For fixed (k,i): sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * V[j*p+k,c]) = 1
                # For each j: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1
                # Let E[ki,j,c] = U[i*m+j,c] * V[j*p+k,c]
                # W[k*n+i, :] @ E[k,i,:,:].T = ones(m)
                E_ki = np.zeros((m, rank))
                for j in range(m):
                    E_ki[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    sol, _, _, _ = np.linalg.lstsq(E_ki, np.ones(m), rcond=None)
                    W_new[k*n+i, :] = sol
                except np.linalg.LinAlgError:
                    W_new[k*n+i, :] = W[k*n+i, :]
        
        # Check convergence
        U_diff = np.linalg.norm(U_new - U)
        V_diff = np.linalg.norm(V_new - V)
        W_diff = np.linalg.norm(W_new - W)
        U, V, W = U_new, V_new, W_new
        
        if max(U_diff, V_diff, W_diff) < tol:
            break
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    deadline = time.time() + 110  # Leave 10s margin
    
    # Start with schoolbook as fallback
    rank_school = n * m * p  # 72
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try ALS with decreasing ranks
    # Lower bound is max(n*m, m*p, n*p) = max(12, 24, 18) = 24
    # Try ranks from 48 down to 24
    ranks_to_try = [48, 42, 36, 30, 27, 24]
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        found = False
        for seed in [42, 123, 7, 999, 555]:
            if time.time() > deadline:
                break
            try:
                U, V, W = _als_decompose(n, m, p, rank, max_iter=100, tol=1e-10, seed=seed)
                if _verify(U, V, W, n, m, p, atol=1e-6):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    found = True
                    break
            except Exception:
                continue
        
        if found:
            # Try to go lower
            continue
        else:
            # This rank didn't work, try lower
            continue
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
