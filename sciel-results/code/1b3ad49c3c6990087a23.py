import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    n, m, p = DIMENSIONS
    # Target tensor: C[i,k] = sum_j A[i,j]*B[j,k]
    # T[a,b,c] = 1 if (i,j,k) consistent, else 0
    # a = i*m+j, b = j*p+k, c = k*n+i
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Precompute coefficient matrices for ALS
    # For updating U[a,:]: need V[b,:] * W[c,:] for all b,c
    # For updating V[b,:]: need U[a,:] * W[c,:] for all a,c
    # For updating W[c,:]: need U[a,:] * V[b,:] for all a,b
    
    # Precompute the structure
    # For U update at row a: A_U[a] has rows indexed by (b,c), columns by t
    # A_U[a][(b,c), t] = V[b,t] * W[c,t]
    # This depends on V and W, so we build it each iteration but can precompute indices
    
    # Actually, let's precompute the (b,c) pairs that matter
    # For each a, which (b,c) have T[a,b,c] != 0?
    # T[a,b,c] = 1 when a=i*m+j, b=j*p+k, c=k*n+i
    # So for fixed a=i*m+j: b must be j*p+k for some k, c must be k*n+i for same k
    # So only p values of (b,c) are nonzero for each a
    
    # Similarly for V update at row b=j*p+k: a must be i*m+j for some i, c must be k*n+i for same i
    # So only n values of (a,c) are nonzero
    
    # For W update at row c=k*n+i: a must be i*m+j for some j, b must be j*p+k for same j
    # So only m values of (a,b) are nonzero
    
    # This means the linear systems are much smaller than the full cross product!
    
    # For U[a,:]: only p equations (one per k)
    # For V[b,:]: only n equations (one per i)  
    # For W[c,:]: only m equations (one per j)
    
    # But we still need the full system to ensure zero entries stay zero.
    # Actually, the equations for zero entries are: sum_t U[a,t]*V[b,t]*W[c,t] = 0
    # These are automatically satisfied if the nonzero equations are satisfied? No.
    # We need ALL equations.
    
    # Let me reconsider. The full system for U[a,:] has m*p*p*n equations.
    # But many are redundant or zero.
    
    # Key insight: T[a,b,c] is nonzero only when (a,b,c) correspond to valid (i,j,k).
    # For invalid (a,b,c), T=0, so we need sum_t U[a,t]*V[b,t]*W[c,t] = 0.
    
    # This is a hard constraint. Let me just use the full system but be smarter.
    
    # Alternative: use a different optimization approach.
    # Minimize ||U ⊙ V ⊙ W - T||_F^2 using gradient descent or ALS with proper handling.
    
    def try_rank(r, max_iters=200, tol=1e-8):
        """Try to find a rank-r decomposition using ALS with precomputed structure."""
        if time.time() > deadline - 5:
            return None
        
        # Multiple random restarts
        best_err = np.inf
        best_factors = None
        
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            np.random.seed(seed)
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # Precompute the index mappings
            # For each a, list of (b, c) pairs
            a_to_bc = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        if a not in a_to_bc:
                            a_to_bc[a] = []
                        a_to_bc[a].append((b, c))
            
            # For each b, list of (a, c) pairs
            b_to_ac = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        if b not in b_to_ac:
                            b_to_ac[b] = []
                        b_to_ac[b].append((a, c))
            
            # For each c, list of (a, b) pairs
            c_to_ab = {}
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        if c not in c_to_ab:
                            c_to_ab[c] = []
                        c_to_ab[c].append((a, b))
            
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each a, solve least squares
                # Full system: for all (b,c), sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
                # A has shape (m*p*p*n, r), A[(b,c),t] = V[b,t]*W[c,t]
                # But this is expensive. Use the fact that only p entries are 1.
                
                # Actually, let's build the full A matrix but do it efficiently
                # A for row a: rows are (b,c) pairs, A[(b,c),t] = V[b,t]*W[c,t]
                # We can build this as: for each t, the column is V[:,t] outer W[:,t]
                
                # More efficient: A = kron(V, W) reshaped
                # Actually, A[(b,c),t] = V[b,t]*W[c,t]
                # This is a 3D array V[:, :, t] * W[:, :, t] ... no
                # A is a 2D matrix of shape (m*p*p*n, r)
                # A[idx(b,c), t] = V[b,t] * W[c,t]
                
                # We can compute: A = np.einsum('bt,ct->bct', V, W).reshape(m*p*p*n, r)
                # But this is the same for all a! So we precompute it.
                
                A_U = np.einsum('bt,ct->bct', V, W).reshape(m*p, p*n, r)
                # A_U[b, c, t] = V[b,t] * W[c,t]
                
                for a in range(n*m):
                    # rhs = T[a, :, :].reshape(-1)
                    # A_a = A_U.reshape(m*p*p*n, r)
                    # But A_U is the same for all a!
                    A_full = A_U.reshape(m*p*p*n, r)
                    rhs = T[a].reshape(-1)
                    U[a, :], residuals, rank_a, sv = np.linalg.lstsq(A_full, rhs, rcond=None)
                
                # Update V
                A_V = np.einsum('at,ct->act', U, W).reshape(n*m, p*n, r)
                for b in range(m*p):
                    A_full = A_V.reshape(n*m*p*n, r)
                    rhs = T[:, b, :].reshape(-1)
                    V[b, :], residuals, rank_b, sv = np.linalg.lstsq(A_full, rhs, rcond=None)
                
                # Update W
                A_W = np.einsum('at,bt->abt', U, V).reshape(n*m, m*p, r)
                for c in range(p*n):
                    A_full = A_W.reshape(n*m*m*p, r)
                    rhs = T[:, :, c].reshape(-1)
                    W[c, :], residuals, rank_c, sv = np.linalg.lstsq(A_full, rhs, rcond=None)
                
                # Check convergence
                recon = np.einsum('at,bt,ct->abc', U, V, W)
                err = np.abs(recon - T).max()
                if err < tol:
                    break
            
            recon = np.einsum('at,bt,ct->abc', U, V, W)
            err = np.abs(recon - T).max()
            if err < best_err:
                best_err = err
                best_factors = (U.copy(), V.copy(), W.copy())
        
        if best_err > 1e-4:
            return None
        
        U, V, W = best_factors
        
        # Try to round to rational values with small denominators
        for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]:
            U_r = np.round(U * denom) / denom
            V_r = np.round(V * denom) / denom
            W_r = np.round(W * denom) / denom
            recon_r = np.einsum('at,bt,ct->abc', U_r, V_r, W_r)
            if np.abs(recon_r - T).max() < 1e-6:
                return (U_r, V_r, W_r)
        
        # If we can't round cleanly, try a more aggressive approach
        # Use the fact that the tensor is sparse and try to find a sparse solution
        return None
    
    # Try ranks from low to high
    best_result = None
    for r in range(20, 37):
        if time.time() > deadline - 10:
            break
        result = try_rank(r)
        if result is not None:
            best_result = (r, result)
            break
    
    if best_result is None:
        # Fallback to schoolbook
        r = n * m * p
        U = np.zeros((n*m, r))
        V = np.zeros((m*p, r))
        W = np.zeros((p*n, r))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_result = (r, (U, V, W))
    
    r, (U, V, W) = best_result
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
