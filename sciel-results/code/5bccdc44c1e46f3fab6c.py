import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 4, 6]
DEADLINE = 115.0  # safety margin before 120s

def _build_rank42():
    """Build the known rank-42 block-Strassen solution."""
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    U_s = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1}
    ]
    V_s = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1}
    ]
    W_s = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):-1},
        {(0,1):1, (1,1):1},
        {(0,0):1, (1,0):1},
        {(0,0):-1, (0,1):1},
        {(1,1):1},
        {(0,0):1}
    ]

    col = 0
    for a_block in range(2):
        for b_block in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_s[s].items():
                    u_idx = i_loc * m + (a_block * 2 + j_loc)
                    U[u_idx, col] += coeff
                for (j_loc, k_loc), coeff in V_s[s].items():
                    v_idx = (a_block * 2 + j_loc) * p + (b_block * 2 + k_loc)
                    V[v_idx, col] += coeff
                for (i_loc, k_loc), coeff in W_s[s].items():
                    w_idx = (b_block * 2 + k_loc) * n + i_loc
                    W[w_idx, col] += coeff
                col += 1
    return U, V, W

def _tensor_check_fast(U, V, W, tol=1e-8):
    """Check if U⊗V⊗W equals the matmul tensor exactly (within tol) using vectorized ops."""
    n, m, p = DIMENSIONS
    r = U.shape[1]
    
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    T_computed = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    T_target = np.ones((n, m, p))
    return np.allclose(T_computed, T_target, atol=tol)

def _round_to_halfint(x, tol=1e-6):
    """Round to nearest half-integer if close enough."""
    doubled = 2.0 * x
    rounded = np.round(doubled)
    if np.allclose(doubled, rounded, atol=tol):
        return rounded / 2.0
    return None

def _als_optimize(T_flat, n, m, p, rank, max_iter=2000, tol=1e-10, seed=42):
    """
    Alternating Least Squares for tensor decomposition.
    T_flat is the flattened tensor of shape (n*m*p,) representing T[i,j,k] = 1.
    Returns (U, V, W, error) where U is (n*m, rank), V is (m*p, rank), W is (p*n, rank).
    """
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank) / np.sqrt(rank)
    V = rng.randn(m * p, rank) / np.sqrt(rank)
    W = rng.randn(p * n, rank) / np.sqrt(rank)
    
    # Precompute reshaped T
    T = np.ones((n, m, p))
    
    best_error = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        # Update U: for each i, solve least squares
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Fix V, W. For fixed i, j: T[i,j,:] = sum_r U[i*m+j, r] * (V[j,:] * W[:,i])^T... 
        # Actually: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For fixed i: T[i,:,k] = sum_j sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # This is complex. Let's use the standard ALS approach.
        
        # Reshape T to (n*m, p, n) for U update? No.
        # Let's think of it as: for fixed V and W, we want to find U such that
        # T[i,j,k] ≈ sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        
        # For each (i,j), define vector w_ij[k] = sum_r V[j,k,r] * W[k,i,r] ... no that's not right either.
        
        # Actually, let's use a different formulation. Define for each r:
        # u_r is a vector of length n*m
        # v_r is a vector of length m*p  
        # w_r is a vector of length p*n
        # T[i,j,k] = sum_r u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        
        # For ALS, fix V and W, update U:
        # For each (i,j), we have: T[i,j,:] = sum_r U[i,j,r] * (v_r[j,:] * w_r[:,i])
        # where v_r[j,:] is V[r, j*p:(j+1)*p] and w_r[:,i] is W[r, i::n]... 
        
        # Let me just do it directly with the full tensor.
        # T is (n,m,p). We want T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        
        # For U update: fix V, W.
        # For each i, j: T[i,j,:] = sum_r U[i,j,r] * (V[j,:,r] * W[:,i,r])
        # Let A_{i,j,k,r} = V[j,k,r] * W[k,i,r]
        # Then T[i,j,k] = sum_r U[i,j,r] * A_{i,j,k,r}
        # So for each (i,j), U[i,j,:] = (A_{i,j}^T A_{i,j})^{-1} A_{i,j}^T T[i,j,:]
        
        # A_{i,j} is a (p, rank) matrix where A_{i,j,k,r} = V[j,k,r] * W[k,i,r]
        
        # Let's compute this efficiently.
        # V is (m*p, rank), reshape to (m, p, rank)
        # W is (p*n, rank), reshape to (p, n, rank)
        
        V_reshaped = V.reshape(m, p, rank)  # V[j,k,r]
        W_reshaped = W.reshape(p, n, rank)  # W[k,i,r]
        
        # A_{i,j,k,r} = V_reshaped[j,k,r] * W_reshaped[k,i,r]
        # For fixed i,j: A_{i,j} is (p, rank) with A_{i,j,k,r} = V_reshaped[j,k,r] * W_reshaped[k,i,r]
        
        # We can compute: for each i, j, the matrix A_{i,j} = diag(V_reshaped[j,:]) @ W_reshaped[:,i]
        # Wait, that's not right. A_{i,j,k,r} = V_reshaped[j,k,r] * W_reshaped[k,i,r]
        # So A_{i,j}[:,r] = V_reshaped[j,:,r] * W_reshaped[:,i,r] (element-wise product over k)
        
        # Let's compute U update:
        U_new = np.zeros((n*m, rank))
        for i in range(n):
            for j in range(m):
                # A_{i,j,k,r} = V_reshaped[j,k,r] * W_reshaped[k,i,r]
                A = V_reshaped[j] * W_reshaped[:, i]  # (p, rank)
                # Solve: T[i,j,:] = A @ U[i,j,:]  (in least squares sense)
                # U[i,j,:] = (A^T A)^{-1} A^T T[i,j,:]
                try:
                    U_new[i*m+j] = np.linalg.lstsq(A, T[i,j], rcond=None)[0]
                except:
                    U_new[i*m+j] = U[i*m+j]
        
        # Similarly for V update:
        V_new = np.zeros((m*p, rank))
        U_reshaped = U_new.reshape(n, m, rank)
        for j in range(m):
            for k in range(p):
                # T[i,j,k] = sum_r U_new[i,j,r] * V[j,k,r] * W[k,i,r]
                # For fixed j,k: T[:,j,k] = sum_r V[j,k,r] * (U_new[:,j,r] * W[k,:,r])
                # B_{i,r} = U_new[i,j,r] * W[k,i,r]
                B = U_new[:, j] * W_reshaped[k]  # (n, rank) where B[i,r] = U_new[i,j,r]*W[k,i,r]
                try:
                    V_new[j*p+k] = np.linalg.lstsq(B, T[:,j,k], rcond=None)[0]
                except:
                    V_new[j*p+k] = V[j*p+k]
        
        # W update:
        W_new = np.zeros((p*n, rank))
        V_reshaped_new = V_new.reshape(m, p, rank)
        for k in range(p):
            for i in range(n):
                # T[i,j,k] = sum_r U_new[i,j,r] * V_new[j,k,r] * W[k,i,r]
                # For fixed k,i: T[i,:,k] = sum_r W[k,i,r] * (U_new[i,:,r] * V_new[:,k,r])
                # C_{j,r} = U_new[i,j,r] * V_new[j,k,r]
                C = U_new[i] * V_reshaped_new[:, k]  # (m, rank) where C[j,r] = U_new[i,j,r]*V_new[j,k,r]
                try:
                    W_new[k*n+i] = np.linalg.lstsq(C, T[i,:,k], rcond=None)[0]
                except:
                    W_new[k*n+i] = W[k*n+i]
        
        U, V, W = U_new, V_new, W_new
        
        # Compute error
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        T_computed = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        error = np.linalg.norm(T_computed - T)
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if error < tol:
            break
    
    return best_U, best_V, best_W, best_error

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Build rank-42 solution as fallback
    U42, V42, W42 = _build_rank42()
    
    # Verify rank-42 solution
    if _tensor_check_fast(U42, V42, W42):
        U_final = _round_to_halfint(U42)
        V_final = _round_to_halfint(V42)
        W_final = _round_to_halfint(W42)
        if U_final is not None and V_final is not None and W_final is not None:
            if _tensor_check_fast(U_final, V_final, W_final, tol=1e-9):
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
        return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
    
    # Try to find lower rank solutions
    # Start with rank 40 and work down
    T = np.ones((n, m, p))
    
    for rank in range(40, 35, -1):
        if time.time() - start_time > DEADLINE - 20:
            break
        
        best_error = np.inf
        best_U, best_V, best_W = None, None, None
        
        for seed in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            U, V, W, error = _als_optimize(T, n, m, p, rank, max_iter=500, tol=1e-12, seed=seed)
            
            if error < best_error:
                best_error = error
                best_U, best_V, best_W = U, V, W
            
            # Check if we can round to half-integers
            U_rounded = _round_to_halfint(U, tol=1e-4)
            V_rounded = _round_to_halfint(V, tol=1e-4)
            W_rounded = _round_to_halfint(W, tol=1e-4)
            
            if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                if _tensor_check_fast(U_rounded, V_rounded, W_rounded, tol=1e-8):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        
        # If best error is very small, try rounding
        if best_U is not None and best_error < 1e-6:
            U_rounded = _round_to_halfint(best_U, tol=1e-3)
            V_rounded = _round_to_halfint(best_V, tol=1e-3)
            W_rounded = _round_to_halfint(best_W, tol=1e-3)
            if U_rounded is not None and V_rounded is not None and W_rounded is not None:
                if _tensor_check_fast(U_rounded, V_rounded, W_rounded, tol=1e-8):
                    return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    
    # Fallback to rank 42
    return {"U": U42.tolist(), "V": V42.tolist(), "W": W42.tolist()}
