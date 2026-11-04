"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random
from itertools import product

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    # Precompute matricizations for ALS
    # T_U: shape (nm, mp*pn) where T_U[u, v*pn+w] = T[u,v,w]
    # T_V: shape (mp, nm*pn) where T_V[v, u*pn+w] = T[u,v,w]
    # T_W: shape (pn, nm*mp) where T_W[w, u*mp+v] = T[u,v,w]
    T_U = T.reshape(nm, mp * pn)
    T_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
    T_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
    
    def verify_decomposition(U, V, W, r, atol=1e-6):
        """Check if U, V, W exactly decompose T."""
        recon = np.zeros((nm, mp, pn), dtype=float)
        for c in range(r):
            recon += np.einsum('u,v,w->uvw', U[:,c], V[:,c], W[:,c])
        return np.allclose(recon, T, atol=atol)
    
    def round_to_half(x):
        """Round to nearest half-integer."""
        return np.round(x * 2) / 2.0
    
    def als_iteration(U, V, W, T_U, T_V, T_W, r, iters=50):
        """Perform ALS iterations to minimize reconstruction error."""
        for it in range(iters):
            # Update U
            # For each column c, solve: U[:,c] = argmin ||T_U - U @ (V^T ⊗ W^T)||_F
            # More precisely: minimize sum_u ||T_U[u,:] - sum_c' U[u,c'] * (V^T ⊗ W^T)[c',:]||^2
            # Let M = V.T @ W.T reshaped... actually let's think differently.
            # T = sum_c U_c ⊗ V_c ⊗ W_c
            # T_U = sum_c (V_c ⊗ W_c) @ U_c^T  ... no
            # Actually: T_U[u, v*pn+w] = sum_c U[u,c] * V[v,c] * W[w,c]
            # So T_U = sum_c U[:,c] * (V[:,c] ⊗ W[:,c])^T
            # Let A_c = V[:,c] ⊗ W[:,c] (shape mp*pn)
            # Then T_U = U @ A^T where A is (r, mp*pn) with A[c,:] = A_c
            A = np.zeros((r, mp * pn))
            for c in range(r):
                A[c, :] = np.kron(V[:,c], W[:,c])
            # Solve U = T_U @ A.T @ (A @ A.T)^{-1}
            try:
                AATA = A @ A.T
                U_new = T_U @ A.T @ np.linalg.inv(AATA)
            except np.linalg.LinAlgError:
                U_new = T_U @ A.T @ np.linalg.pinv(AATA)
            U = U_new
            
            # Update V
            # T_V[v, u*pn+w] = sum_c V[v,c] * U[u,c] * W[w,c]
            # T_V = V @ B^T where B[c,:] = kron(U[:,c], W[:,c])
            B = np.zeros((r, nm * pn))
            for c in range(r):
                B[c, :] = np.kron(U[:,c], W[:,c])
            try:
                BBT = B @ B.T
                V_new = T_V @ B.T @ np.linalg.inv(BBT)
            except np.linalg.LinAlgError:
                V_new = T_V @ B.T @ np.linalg.pinv(BBT)
            V = V_new
            
            # Update W
            # T_W[w, u*mp+v] = sum_c W[w,c] * U[u,c] * V[v,c]
            # T_W = W @ C^T where C[c,:] = kron(U[:,c], V[:,c])
            C = np.zeros((r, nm * mp))
            for c in range(r):
                C[c, :] = np.kron(U[:,c], V[:,c])
            try:
                CCT = C @ C.T
                W_new = T_W @ C.T @ np.linalg.inv(CCT)
            except np.linalg.LinAlgError:
                W_new = T_W @ C.T @ np.linalg.pinv(CCT)
            W = W_new
        
        return U, V, W
    
    best = None
    best_rank = nm * mp * pn  # schoolbook rank
    
    # Try ranks from a reasonable upper bound downward
    # Start with schoolbook and try to go lower
    for r in range(144, 60, -1):
        for trial in range(5):
            random.seed(42 + trial * 1000 + r)
            np.random.seed(42 + trial * 1000 + r)
            
            # Initialize with random half-integers
            U = np.random.choice([-0.5, 0.0, 0.5], size=(nm, r))
            V = np.random.choice([-0.5, 0.0, 0.5], size=(mp, r))
            W = np.random.choice([-0.5, 0.0, 0.5], size=(pn, r))
            
            # Run ALS
            U, V, W = als_iteration(U, V, W, T_U, T_V, T_W, r, iters=30)
            
            # Round to half-integers
            U_rounded = round_to_half(U)
            V_rounded = round_to_half(V)
            W_rounded = round_to_half(W)
            
            # Verify rounded version
            if verify_decomposition(U_rounded, V_rounded, W_rounded, r):
                best = (r, U_rounded, V_rounded, W_rounded)
                break
        
        if best is not None:
            break
    
    if best is not None:
        r, U, V, W = best
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fall back to schoolbook
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
