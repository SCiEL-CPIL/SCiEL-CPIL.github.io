import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    # Build target tensor - only 63 non-zero entries
    T = np.zeros((n*m, m*p, p*n))
    nonzero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
                nonzero_indices.append((a, b, c))
    
    # Precompute index arrays for fast verification
    nz_a = np.array([idx[0] for idx in nonzero_indices])
    nz_b = np.array([idx[1] for idx in nonzero_indices])
    nz_c = np.array([idx[2] for idx in nonzero_indices])
    
    def verify_halfint(U, V, W):
        """Verify U,V,W are valid half-integer decompositions."""
        U2 = np.round(U * 2)
        V2 = np.round(V * 2)
        W2 = np.round(W * 2)
        
        # Check they're actually half-integers
        if not np.allclose(U2, U * 2, atol=0.01):
            return False
        if not np.allclose(V2, V * 2, atol=0.01):
            return False
        if not np.allclose(W2, W * 2, atol=0.01):
            return False
        
        U2 = U2.astype(np.int64)
        V2 = V2.astype(np.int64)
        W2 = W2.astype(np.int64)
        
        # Check non-zero entries: T[a,b,c] = sum_k U[a,k]*V[b,k]*W[c,k] = 8
        for idx in range(len(nz_a)):
            a, b, c = nz_a[idx], nz_b[idx], nz_c[idx]
            val = int(np.dot(U2[a, :], V2[b, :] * W2[c, :]))
            if val != 8:
                return False
        
        # Full check
        T_calc = np.einsum('ak,bk,ck->abc', U2, V2, W2)
        if T_calc.sum() != 8 * len(nz_a):
            return False
        if not np.all(T_calc[nz_a, nz_b, nz_c] == 8):
            return False
        # Check zeros are zero
        mask = np.ones((n*m, m*p, p*n), dtype=bool)
        mask[nz_a, nz_b, nz_c] = False
        if np.any(T_calc[mask] != 0):
            return False
        
        return True
    
    def solve_als(r, time_limit, U_init=None, V_init=None, W_init=None):
        """Solve rank-r decomposition using proper ALS with Kronecker products."""
        if U_init is None:
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
        else:
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
        
        best_residual = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(60):
            if time.time() > time_limit:
                break
            
            # Update U: For each row a, solve (V⊗W)^T (V⊗W) U[a,:] = (V⊗W)^T T[a,:,:]
            # Using Kronecker: the design matrix for row a is kron(W, V) transposed appropriately
            # T[a,b,c] = sum_k U[a,k] * V[b,k] * W[c,k]
            # Vectorizing: T[a].flatten() = M @ U[a,:].flatten() where M[b*p*n+c, k] = V[b,k]*W[c,k]
            M_U = np.zeros((m*p*p*n, r))
            for b in range(m*p):
                for c in range(p*n):
                    M_U[b*p*n + c, :] = V[b, :] * W[c, :]
            
            for a in range(n*m):
                t_a = T[a, :, :].flatten()
                U[a, :], _, _, _ = np.linalg.lstsq(M_U, t_a, rcond=None)
            
            # Update V
            M_V = np.zeros((n*m*p*n, r))
            for a in range(n*m):
                for c in range(p*n):
                    M_V[a*p*n + c, :] = U[a, :] * W[c, :]
            
            for b in range(m*p):
                t_b = T[:, b, :].flatten()
                V[b, :], _, _, _ = np.linalg.lstsq(M_V, t_b, rcond=None)
            
            # Update W
            M_W = np.zeros((n*m*m*p, r))
            for a in range(n*m):
                for b in range(m*p):
                    M_W[a*m*p + b, :] = U[a, :] * V[b, :]
            
            for c in range(p*n):
                t_c = T[:, :, c].flatten()
                W[c, :], _, _, _ = np.linalg.lstsq(M_W, t_c, rcond=None)
            
            # Check convergence
            U_rounded = np.round(U * 2) / 2.0
            V_rounded = np.round(V * 2) / 2.0
            W_rounded = np.round(W * 2) / 2.0
            
            if verify_halfint(U_rounded, V_rounded, W_rounded):
                return U_rounded, V_rounded, W_rounded
            
            # Track best by residual
            if it % 5 == 0:
                T_calc = np.einsum('ak,bk,ck->abc', U, V, W)
                residual = np.linalg.norm(T_calc - T)
                if residual < best_residual:
                    best_residual = residual
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Final check
        U_rounded = np.round(U * 2) / 2.0
        V_rounded = np.round(V * 2) / 2.0
        W_rounded = np.round(W * 2) / 2.0
        if verify_halfint(U_rounded, V_rounded, W_rounded):
            return U_rounded, V_rounded, W_rounded
        
        return None
    
    # Try ranks from low to high, with multiple restarts
    # Known: schoolbook is 63. Lower bound is at least 50 for 3x3x7.
    for r in range(50, 64):
        if time.time() > deadline:
            break
        
        n_restarts = max(1, int((deadline - time.time()) / 5))
        n_restarts = min(n_restarts, 8)
        
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if restart == 0:
                # Sparse-like init
                U = np.zeros((n*m, r))
                V = np.zeros((m*p, r))
                W = np.zeros((p*n, r))
                for col in range(r):
                    i = col % n
                    j = (col // n) % m
                    k = (col // (n*m)) % p
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                U += np.random.randn(n*m, r) * 0.1
                V += np.random.randn(m*p, r) * 0.1
                W += np.random.randn(p*n, r) * 0.1
            elif restart == 1:
                # Random
                U = np.random.randn(n*m, r) * 0.5
                V = np.random.randn(m*p, r) * 0.5
                W = np.random.randn(p*n, r) * 0.5
            else:
                # Random with different scale
                scale = 0.3 + 0.2 * restart
                U = np.random.randn(n*m, r) * scale
                V = np.random.randn(m*p, r) * scale
                W = np.random.randn(p*n, r) * scale
            
            res = solve_als(r, deadline - 2, U, V, W)
            if res is not None:
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 63
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
