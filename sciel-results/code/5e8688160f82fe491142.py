"""Program for matrix multiplication 2,3,9 tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140  # 10s safety margin before 150s kill

def _check_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition."""
    rank = U.shape[1]
    # For all i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _als_decompose(n, m, p, rank, max_iters=500, tol=1e-12):
    """
    Find a rank-R decomposition of the all-ones tensor T[i,j,k]=1
    using Alternating Least Squares (ALS).
    """
    nm, mp, pn = n*m, m*p, p*n
    
    # Initialize
    np.random.seed(42)
    U = np.random.randn(nm, rank) * 0.1
    V = np.random.randn(mp, rank) * 0.1
    W = np.random.randn(pn, rank) * 0.1
    
    # Precompute indices for efficiency
    # We need to solve linear systems for each column of U, V, W.
    # The objective is min || T - sum_r u_r v_r w_r ||^2
    # T is all 1s.
    
    prev_loss = float('inf')
    
    for it in range(max_iters):
        if time.time() > _DEADLINE:
            return None
        
        # --- Update U ---
        # For each r, we want to minimize sum_{i,j,k} (1 - sum_{r'} U[i,j,r'] V[j,k,r'] W[k,i,r'])^2
        # The term linear in U[i,j,r] is:
        # U[i,j,r] * sum_k V[j,k,r] W[k,i,r]
        # The residual for r is: 1 - sum_{r' != r} U[i,j,r'] V[j,k,r'] W[k,i,r']
        # This is a linear system for each r.
        # Let's vectorize.
        
        # Build the "other" tensors
        # We can update all columns of U simultaneously by solving a block system,
        # but since rank is small, we can do it column by column or use a trick.
        
        # Efficient update:
        # Let M_VW be a tensor where M_VW[i,j,k] = sum_{r'} V[j,k,r'] W[k,i,r']
        # This is not quite right because the sum is over r' but U[i,j,r'] multiplies it.
        
        # Standard ALS step:
        # Fix V and W.
        # The tensor T_approx = sum_{r'} U[:,r'] * V[:,r'] * W[:,r'] (outer product sum)
        # We want to replace column r of U with new values.
        # This is equivalent to solving:
        # min_{u_r} || T - sum_{r' != r} u_{r'} v_{r'} w_{r'} - u_r v_r w_r ||^2
        
        # Let R_r = T - sum_{r' != r} u_{r'} v_{r'} w_{r'}
        # We want min_{u_r} || R_r - u_r v_r w_r ||^2
        # This is a least squares problem:
        # min_{u_r} || R_r - u_r \otimes v_r \otimes w_r ||_F^2
        
        # We can flatten this.
        # Let A_r be a matrix where the (i,j)-th row is the vector (v_r[j,k] * w_r[k,i])_k? No.
        # The term is sum_k u_r[i,j] v_r[j,k] w_r[k,i].
        # This is linear in u_r[i,j].
        # For fixed r, the variables are u_r[i,j] for all i,j.
        # The coefficient for u_r[i,j] in the expansion of the square is:
        # sum_k (v_r[j,k] w_r[k,i])^2
        # The cross term with other u_r[i',j] is zero because u_r is only in one column.
        # Wait, u_r is a vector of length nm.
        # The objective is sum_{i,j,k} (R_r[i,j,k] - u_r[i,j] v_r[j,k] w_r[k,i])^2
        # Expanding:
        # sum_{i,j,k} R_r[i,j,k]^2 - 2 sum_{i,j,k} R_r[i,j,k] u_r[i,j] v_r[j,k] w_r[k,i] + sum_{i,j,k} u_r[i,j]^2 v_r[j,k]^2 w_r[k,i]^2
        
        # This is a separable least squares problem for each u_r[i,j]?
        # No, u_r[i,j] appears in terms for all k.
        # For a fixed (i,j), u_r[i,j] appears in:
        # sum_k (R_r[i,j,k] - u_r[i,j] v_r[j,k] w_r[k,i])^2
        # = sum_k R_r[i,j,k]^2 - 2 u_r[i,j] sum_k R_r[i,j,k] v_r[j,k] w_r[k,i] + u_r[i,j]^2 sum_k (v_r[j,k] w_r[k,i])^2
        
        # So we can solve for each u_r[i,j] independently!
        # u_r[i,j] = (sum_k R_r[i,j,k] v_r[j,k] w_r[k,i]) / (sum_k (v_r[j,k] w_r[k,i])^2)
        
        # This is extremely fast. O(nm * rank * p) per update.
        
        # Calculate R_r = T - sum_{r' != r} u_{r'} v_{r'} w_{r'}
        # We can maintain the sum S = sum_{r'} u_{r'} v_{r'} w_{r'}
        # Then R_r = 1 - (S - u_r v_r w_r) = 1 - S + u_r v_r w_r
        
        # Let's implement this.
        
        # Precompute V and W arrays for access
        V_arr = V
        W_arr = W
        
        # Current sum tensor S
        # S[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We can compute this on the fly or maintain it.
        # Maintaining it is O(nm*mp*pn*rank) which is too slow.
        # But the update formula requires R_r.
        
        # Alternative: Just compute the numerator and denominator.
        # Denom[i,j] for column r: sum_k (V[j*p+k, r] * W[k*n+i, r])^2
        # Num[i,j] for column r: sum_k (1 - S_excl_r[i,j,k]) * V[j*p+k, r] * W[k*n+i, r]
        
        # This is getting complex to optimize perfectly in Python.
        # Let's stick to a simpler but still fast approach:
        # Randomized SVD or just simple gradient descent with L-BFGS?
        # No, let's try the explicit formula for u_r[i,j].
        
        # To make it fast, we precompute the "basis" tensors for each r.
        # Basis_r[i,j,k] = V[j*p+k, r] * W[k*n+i, r]
        
        # We need S = sum_r U[:,r] * Basis_r
        # We need S_excl_r = S - U[:,r] * Basis_r
        
        # Let's just do a simple gradient step with a good step size, 
        # but using vectorized operations.
        
        # Compute S
        S = np.zeros((n, m, p))
        for r in range(rank):
            # u is (nm,), v is (mp,), w is (pn,)
            # We need to broadcast to (n,m,p)
            # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # This is a bit tricky to vectorize without loops over r.
            # But rank is small (~45).
            
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            
            # S[i,j,k] += u_col[i,j] * v_col[j,k] * w_col[k,i]
            # This is a sum over k? No, k is an index.
            # For each i,j:
            # S[i,j,:] += u_col[i,j] * v_col[j,:] * w_col[:,i]
            
            # Vectorized:
            # term = u_col[:, :, None] * v_col[None, :, :] * w_col[None, :, 0] ? No.
            # w_col is (p, n). We need w_col[k, i].
            # So for fixed i, w_col[:, i] is a vector of length p.
            
            # S[:, :, :] += u_col[:, :, None] * v_col[None, :, :] * w_col[None, :, 0] is wrong.
            # w_col[k, i] means index k varies, i is fixed.
            # So w_col.T[i, :] is the vector for fixed i.
            
            # Let's do:
            # term[i, j, k] = u_col[i, j] * v_col[j, k] * w_col[k, i]
            
            # We can compute this as:
            # tmp1 = u_col[:, :, None] * v_col[None, :, :]  # shape (n, m, p)
            # tmp2 = w_col.T  # shape (n, p) -> w_col.T[i, k] = w_col[k, i]
            # S += tmp1 * tmp2[:, None, :]  # broadcast (n, p) to (n, m, p)
            
            tmp1 = u_col[:, :, None] * v_col[None, :, :]
            tmp2 = w_col.T  # (n, p)
            S += tmp1 * tmp2[:, None, :]
            
        # Now S is computed.
        # Update U
        for r in range(rank):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            
            # Basis for r
            basis = v_col[None, :, :] * w_col.T[:, None, :] # (n, m, p) ?
            # w_col.T[i, k] = w_col[k, i].
            # So basis[i, j, k] = v_col[j, k] * w_col[k, i].
            
            # S_excl = S - u_col[:, :, None] * basis
            S_excl = S - u_col[:, :, None] * basis
            
            # Numerator: sum_k S_excl[i,j,k] * basis[i,j,k]
            # Denominator: sum_k basis[i,j,k]^2
            
            num = np.sum(S_excl * basis, axis=2) # (n, m)
            den = np.sum(basis**2, axis=2) # (n, m)
            
            # Avoid division by zero
            den[den < 1e-12] = 1e-12
            
            new_u = num / den
            U[:, r] = new_u.flatten()
            
            # Update S
            # S = S_excl + new_u[:, :, None] * basis
            S = S_excl + new_u[:, :, None] * basis
            
        # Similar updates for V and W would be symmetric.
        # For time efficiency, let's just do one pass of U, V, W updates per iteration.
        
        # Update V
        for r in range(rank):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            
            # Basis for V: basis[i, j, k] = u_col[i, j] * w_col[k, i]
            # u_col[i,j] * w_col[k,i]
            # tmp1 = u_col[:, :, None] # (n, m, 1)
            # tmp2 = w_col.T[:, None, :] # (n, 1, p) -> w_col[k, i] is at [i, k] in transposed?
            # w_col.T[i, k] = w_col[k, i].
            # So basis[i, j, k] = u_col[i, j] * w_col.T[i, k]
            
            basis = u_col[:, :, None] * w_col.T[:, None, :] # (n, m, p)
            
            S_excl = S - v_col[None, :, :] * basis # v_col[j,k] is at [None, j, k]
            
            num = np.sum(S_excl * basis, axis=0) # (m, p)
            den = np.sum(basis**2, axis=0) # (m, p)
            den[den < 1e-12] = 1e-12
            
            new_v = num / den
            V[:, r] = new_v.flatten()
            
            S = S_excl + new_v[None, :, :] * basis
            
        # Update W
        for r in range(rank):
            u_col = U[:, r].reshape(n, m)
            v_col = V[:, r].reshape(m, p)
            w_col = W[:, r].reshape(p, n)
            
            # Basis for W: basis[i, j, k] = u_col[i, j] * v_col[j, k]
            basis = u_col[:, :, None] * v_col[None, :, :] # (n, m, p)
            
            S_excl = S - w_col.T[:, None, :] * basis # w_col[k, i] is at [i, k] in transposed
            # Wait, w_col.T[i, k] = w_col[k, i].
            # So w_col.T[:, None, :] is (n, 1, p).
            # basis is (n, m, p).
            # Product is (n, m, p).
            
            num = np.sum(S_excl * basis, axis=1) # (n, p)
            den = np.sum(basis**2, axis=1) # (n, p)
            den[den < 1e-12] = 1e-12
            
            new_w = num / den
            # new_w is (n, p). w_col is (p, n).
            # w_col[k, i] = new_w[i, k]
            W[:, r] = new_w.T.flatten()
            
            # new_w[i, k] corresponds to W[k*n+i, r]
            # We need to broadcast new_w (n, p) to (n, m, p)
            # new_w[:, None, :] is (n, 1, p)
            S = S_excl + new_w[:, None, :] * basis
            
        # Check error
        err = np.sum((S - 1.0)**2)
        if err < tol:
            return U, V, W
            
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank = 54
    
    # Try to find lower rank decompositions
    # Theoretical lower bound is around 27, but structural constraints may be higher.
    # Try ranks from 45 to 53.
    for target_rank in range(45, best_rank):
        if time.time() > _DEADLINE:
            break
            
        # Try multiple random initializations
        found = False
        for attempt in range(10):
            if time.time() > _DEADLINE:
                break
                
            # Set seed for reproducibility in attempts
            np.random.seed(attempt * 1000 + target_rank)
            
            result = _als_decompose(n, m, p, target_rank, max_iters=200)
            if result is not None:
                U, V, W = result
                if _check_decomposition(U, V, W, n, m, p):
                    best_rank = target_rank
                    best_result = (U, V, W)
                    found = True
                    break
        
        if found:
            break
            
    if best_result is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_result
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
