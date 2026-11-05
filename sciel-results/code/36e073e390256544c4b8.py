import numpy as np
import time
import random

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T of shape (nm, mp, pn)
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Flatten T to vector for linear algebra
    T_flat = T.flatten()
    
    def rank_decomposition(U, V, W):
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Tensor = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # This is equivalent to matmul of (U * V) and W? No.
        # Tensor[a,b,c] = sum_c U[a,c] V[b,c] W[c,c]
        # We can compute the error by reconstructing T.
        # Efficiently: 
        # Residual = T - sum_c u_c v_c w_c^T (flattened)
        # Let's just compute the full tensor if r is small, or use matrix ops.
        # T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
        # Error = np.linalg.norm(T - T_recon)
        pass

    def try_rank(r, max_iters=100, tol=1e-9):
        # Initialize
        U = np.random.randn(nm, r)
        V = np.random.randn(mp, r)
        W = np.random.randn(pn, r)
        
        # Normalize to prevent divergence
        U /= np.linalg.norm(U, axis=0, keepdims=True)
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        
        for it in range(max_iters):
            # Update U
            # T[a,b,c] = sum_c U[a,c] (V[b,c] W[c,c])
            # For fixed c, term is U[:,c] * (V[:,c] * W[:,c])
            # We want to minimize || T - sum_c U[:,c] (V[:,c] \otimes W[:,c]) ||
            # This is a linear least squares problem for U.
            # T_flat[a] = sum_c U[a,c] * M_c[a] where M_c is derived from V, W?
            # No, T is 3D.
            # Let's reshape T to (nm, mp*pn).
            # T[a, b, c] -> T2D[a, b*pn+c]
            # T2D = sum_c U[:,c] * (V[:,c] \otimes W[:,c])^T ?
            # (V[:,c] \otimes W[:,c]) is a vector of length mp*pn.
            # Let X_c = np.kron(V[:,c], W[:,c])
            # T2D[a, :] = sum_c U[a,c] X_c
            # So T2D = U * X^T
            # We can solve for U given X.
            
            T2D = T.reshape(nm, mp * pn)
            
            # Compute X (r x mp*pn)
            # X[c, b*pn+k] = V[b,c] * W[k,c]
            # We can compute this efficiently.
            # V is (mp, r), W is (pn, r).
            # X is (r, mp*pn).
            # X = (V * W[:, np.newaxis, :]).reshape(r, -1)? No.
            # Let's construct X.
            # X[c] = np.kron(V[:,c], W[:,c])
            X = np.zeros((r, mp * pn))
            for c in range(r):
                X[c] = np.kron(V[:, c], W[:, c])
            
            # Solve U * X^T = T2D
            # U = T2D * X^T (X X^T)^-1
            # Use lstsq
            try:
                U_new, residuals, rank, sv = np.linalg.lstsq(X.T, T2D, rcond=None)
                # U_new is (r, nm). We need (nm, r).
                U = U_new.T
            except np.linalg.LinAlgError:
                continue
                
            # Update V
            # T2D_V = T.reshape(mp, nm * pn)
            # T2D_V[b, a*pn+c] = sum_c V[b,c] U[a,c] W[c,c]
            # Let Y_c = np.kron(U[:,c], W[:,c])
            # T2D_V = V * Y^T
            T2D_V = T.reshape(mp, nm * pn)
            Y = np.zeros((r, nm * pn))
            for c in range(r):
                Y[c] = np.kron(U[:, c], W[:, c])
            try:
                V_new, residuals, rank, sv = np.linalg.lstsq(Y.T, T2D_V, rcond=None)
                V = V_new.T
            except np.linalg.LinAlgError:
                continue

            # Update W
            # T2D_W = T.reshape(pn, nm * mp)
            # T2D_W[k, a*mp+b] = sum_c W[k,c] U[a,c] V[b,c]
            # Let Z_c = np.kron(U[:,c], V[:,c])
            # T2D_W = W * Z^T
            T2D_W = T.reshape(pn, nm * mp)
            Z = np.zeros((r, nm * mp))
            for c in range(r):
                Z[c] = np.kron(U[:, c], V[:, c])
            try:
                W_new, residuals, rank, sv = np.linalg.lstsq(Z.T, T2D_W, rcond=None)
                W = W_new.T
            except np.linalg.LinAlgError:
                continue
            
            # Check error
            # Reconstruct T
            # T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
            # err = np.linalg.norm(T - T_recon)
            # To save time, check Frobenius norm of residual in one mode?
            # Or just check every 10 iterations.
            if it % 10 == 0 or it == max_iters - 1:
                T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.linalg.norm(T - T_recon)
                if err < tol:
                    return U, V, W, True
                    
        return None, None, None, False

    start_time = time.time()
    deadline = start_time + 140  # 140s limit, 10s buffer
    
    # Try ranks from 70 down to 40
    best_result = None
    for r in range(70, 39, -1):
        if time.time() > deadline - 10:
            break
            
        # Try a few restarts for this rank
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            U, V, W, success = try_rank(r)
            if success:
                # Convert to list of lists for output
                U_list = [[int(round(x)) if abs(x - round(x)) < 1e-6 else x for x in row] for row in U.tolist()]
                V_list = [[int(round(x)) if abs(x - round(x)) < 1e-6 else x for x in row] for row in V.tolist()]
                W_list = [[int(round(x)) if abs(x - round(x)) < 1e-6 else x for x in row] for row in W.tolist()]
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # If no better rank found, return schoolbook
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
