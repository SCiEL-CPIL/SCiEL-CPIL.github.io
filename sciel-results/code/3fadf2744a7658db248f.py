import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    # Tensor T of shape (n, m, p)
    # T[i, j, k] = 1 if it's part of the multiplication tensor
    # The multiplication tensor is sum_{i,j,k} e_i (in C) * e_j (in A) * e_k (in B)
    # Actually, the tensor is defined by T[i,j,k] = 1 for all i,j,k?
    # No, the tensor for matrix multiplication C = AB is:
    # T = sum_{i=1}^n sum_{j=1}^m sum_{k=1}^p e_i (in C) \otimes e_j (in A) \otimes e_k (in B)
    # Wait, the indices are:
    # A is n x m. B is m x p. C is n x p.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # So the tensor has entries T[i, j, k] = 1 for all i, j, k.
    # The shape is (n, m, p) = (3, 3, 8).
    
    T = np.ones((n, m, p))
    
    # Target rank
    # Schoolbook is 72.
    # We try to find lower ranks.
    
    def als_decompose(T, rank, max_iter=20, tol=1e-10, seed=0):
        np.random.seed(seed)
        n, m, p = T.shape
        # Initialize with SVD
        T1 = T.transpose(0, 1, 2).reshape(n, m*p)
        U, s, Vt = np.linalg.svd(T1, full_matrices=False)
        U = U[:, :rank]
        
        T2 = T.transpose(1, 0, 2).reshape(m, n*p)
        V, s, Wt = np.linalg.svd(T2, full_matrices=False)
        V = V[:, :rank]
        
        T3 = T.transpose(2, 0, 1).reshape(p, n*m)
        W, s, Ut = np.linalg.svd(T3, full_matrices=False)
        W = W[:, :rank]
        
        for _ in range(max_iter):
            # Update U
            # T1 = sum_r U_r * (V_r * W_r)^T
            # V_r * W_r is a vector of length m*p
            VW = (V * W).T  # rank x (m*p)
            T1_approx = U @ VW
            # We want to minimize || T1 - U VW ||^2
            # For fixed VW, optimal U is such that U = T1 VW^T (VW VW^T)^-1 ?
            # Actually, ALS updates one factor at a time.
            # Fix V, W. Update U.
            # T1 = U @ (V * W).T
            # Let M = (V * W).T  (rank x m*p)
            # We want U such that U M^T approx T1? No.
            # T1[i, l] = sum_r U[i, r] M[r, l]
            # So T1 = U M^T
            # U = T1 M (M^T M)^-1
            M = (V * W).T
            try:
                U = T1 @ M @ np.linalg.inv(M.T @ M)
            except np.linalg.LinAlgError:
                pass
                
            # Update V
            # T2 = V @ (U * W).T
            M = (U * W).T
            try:
                V = T2 @ M @ np.linalg.inv(M.T @ M)
            except np.linalg.LinAlgError:
                pass
                
            # Update W
            # T3 = W @ (U * V).T
            M = (U * V).T
            try:
                W = T3 @ M @ np.linalg.inv(M.T @ M)
            except np.linalg.LinAlgError:
                pass
                
            # Check error
            T_approx = np.einsum('ir,jr,kr->ijk', U, V, W)
            error = np.linalg.norm(T - T_approx)
            if error < tol:
                break
                
        return U, V, W, error

    def check_exact(U, V, W):
        # Snap to nearest 0.5
        U_s = np.round(U * 2) / 2
        V_s = np.round(V * 2) / 2
        W_s = np.round(W * 2) / 2
        
        # Check if exact
        T_approx = np.einsum('ir,jr,kr->ijk', U_s, V_s, W_s)
        if np.allclose(T_approx, T, atol=1e-9):
            return U_s, V_s, W_s
        return None

    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_result = None
    best_rank = 72
    
    # Try ranks from 68 down to 50
    for rank in range(68, 49, -1):
        if time.time() > deadline:
            break
            
        # Try a few random seeds
        for seed in range(5):
            if time.time() > deadline:
                break
            
            U, V, W, error = als_decompose(T, rank, max_iter=30, seed=seed)
            
            if error < 1e-6:
                res = check_exact(U, V, W)
                if res is not None:
                    U_s, V_s, W_s = res
                    # Convert to lists
                    U_list = U_s.tolist()
                    V_list = V_s.tolist()
                    W_list = W_s.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
                    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
