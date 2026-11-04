import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Target ranks to try
    target_ranks = [50, 55, 60, 65, 70, 75]
    
    # Initialize the tensor T
    # T is a 3D array of size (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
                
    # Function to check if a decomposition is exact
    def is_exact(U, V, W, tol=1e-9):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # Reconstruction: sum_r U[:, r] V[:, r] W[:, r]
        # This is a tensor contraction.
        # We can compute the difference and check norm.
        # Efficient way: 
        # C = np.einsum('ir,jr,kr->ijk', U, V, W)
        # But T is sparse.
        # Let's just compute the full tensor and compare.
        # Dimensions are small (16, 20, 20), so it's fine.
        C = np.einsum('ir,jr,kr->ijk', U, V, W)
        diff = C - T
        return np.max(np.abs(diff)) < tol

    # Function to run NMTF and then snap to exact solution
    def try_rank(R):
        # Initialize U, V, W with small positive random values
        np.random.seed(42) # For reproducibility in debugging, but maybe random is better?
        # Actually, let's use random seed based on R to vary
        np.random.seed(R)
        
        U = np.random.rand(nm, R) * 0.1 + 0.01
        V = np.random.rand(mp, R) * 0.1 + 0.01
        W = np.random.rand(pn, R) * 0.1 + 0.01
        
        # NMTF updates
        # Minimize || T - sum U V W ||^2
        # Update rules for NMTF are not standard for tri-factorization.
        # Standard NMTF is for matrix factorization.
        # For tensor, we can use ALS (Alternating Least Squares) with non-negativity constraints.
        # Or just gradient descent.
        # Given time limit, let's use a simple iterative refinement.
        
        # Actually, let's use a heuristic: 
        # 1. Run a few steps of gradient descent on the non-negative manifold.
        # 2. Then, identify the support.
        # 3. Solve linear system.
        
        # Since implementing robust NMTF for tensors is complex, 
        # and the directive says "built on this approach", 
        # I will simulate the search by trying random sparse supports of size R?
        # No, that's not NMTF.
        
        # Let's stick to the plan:
        # 1. Initialize random non-negative U, V, W.
        # 2. Perform a few iterations of multiplicative updates?
        # Multiplicative updates for tensor decomposition:
        # U <- U * (T_(1) W V) / (U W^T V^T) ? 
        # This is getting complicated.
        
        # Alternative interpretation:
        # The "NMTF" might refer to finding a non-negative approximation, 
        # and then using that to guide the exact search.
        
        # Let's try a simpler approach for the exact search:
        # We know the rank is likely close to 50.
        # We can try to construct the decomposition using known algorithms?
        # No, I must write the program.
        
        # Let's try to solve the linear system directly for a given support.
        # But finding the support is the hard part.
        
        # Let's try a random search for a valid support of size R.
        # This is not NMTF.
        
        # Okay, I will implement a basic ALS with non-negativity, 
        # run it for a few iterations, then snap.
        
        # Reshape T to matrices for ALS
        # T_(1) is nm x (mp*pn)
        # T_(2) is mp x (nm*pn)
        # T_(3) is pn x (nm*mp)
        
        T1 = T.reshape(nm, mp * pn)
        T2 = T.reshape(mp, nm * pn)
        T3 = T.reshape(pn, nm * mp)
        
        # Initialize
        U = np.random.rand(nm, R) + 0.1
        V = np.random.rand(mp, R) + 0.1
        W = np.random.rand(pn, R) + 0.1
        
        for it in range(100):
            # Update U
            # U = argmin_U || T1 - U (W^T V^T) ||_F
            # Let M = W^T V^T (R x R) -> No, V is (mp, R), W is (pn, R)
            # (W^T V^T) is not defined. 
            # The term is sum_r U[:, r] V[:, r] W[:, r]
            # For fixed V, W, the term for U is:
            # C = sum_r U[:, r] (V[:, r] W[:, r]^T) ? No.
            # C[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
            # Fix j, k. C[:, j, k] = U (V[j, :] * W[k, :])^T ?
            # V[j, :] is a row vector of length R. W[k, :] is a row vector of length R.
            # Element-wise product is a vector of length R.
            # So C[:, j, k] = U @ (V[j, :] * W[k, :])^T
            # We want to match T[:, j, k].
            # This is a linear least squares problem for each (j, k).
            # But we can do it all at once.
            # Let A be a matrix of size (mp*pn) x R.
            # A[(j*pn+k), r] = V[j, r] * W[k, r]
            # Then T1 = U @ A^T
            # So U = T1 @ A @ (A^T A)^{-1}
            
            A = np.zeros((mp * pn, R))
            for j in range(mp):
                for k in range(pn):
                    A[j * pn + k, :] = V[j, :] * W[k, :]
            
            # Solve U A^T = T1
            # U = T1 A (A^T A)^{-1}
            # Use lstsq
            U_new, _, _, _ = np.linalg.lstsq(A.T, T1.T, rcond=None)
            U = np.maximum(U_new.T, 0) # Non-negativity
            
            # Update V
            # Similarly, B[(i*pn+k), r] = U[i, r] * W[k, r]
            B = np.zeros((nm * pn, R))
            for i in range(nm):
                for k in range(pn):
                    B[i * pn + k, :] = U[i, :] * W[k, :]
            # T2 = V B^T
            V_new, _, _, _ = np.linalg.lstsq(B.T, T2.T, rcond=None)
            V = np.maximum(V_new.T, 0)
            
            # Update W
            # C[(i*mp+j), r] = U[i, r] * V[j, r]
            C_mat = np.zeros((nm * mp, R))
            for i in range(nm):
                for j in range(mp):
                    C_mat[i * mp + j, :] = U[i, :] * V[j, :]
            # T3 = W C_mat^T
            W_new, _, _, _ = np.linalg.lstsq(C_mat.T, T3.T, rcond=None)
            W = np.maximum(W_new.T, 0)
            
            # Check exactness
            if is_exact(U, V, W):
                return U, V, W
                
        # If not exact, try to snap
        # This is a heuristic snap
        # If entries are close to 0, set to 0.
        # If close to 0.5, 1, etc?
        # For now, just return None if not exact.
        return None

    for R in target_ranks:
        res = try_rank(R)
        if res is not None:
            U, V, W = res
            # Convert to lists of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
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
# EVOLVE-BLOCK-END
