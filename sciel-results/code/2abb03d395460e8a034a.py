import time
import random
import numpy as np
from scipy.optimize import least_squares
from numpy import linalg as LA

DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145
    
    def schoolbook():
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    U[i*m + k][col] = 1
                    V[k*p + j][col] = 1
                    W[j*n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_sol = schoolbook()
    
    def verify(U_mat, V_mat, W_mat):
        """Check if sum_c U[i*m+k,c]*V[k*p+j,c]*W[j*n+i,c] == 1 for all i,k,j"""
        r = U_mat.shape[1]
        u_r = U_mat.T.reshape(n, m, r)
        v_r = V_mat.T.reshape(m, p, r)
        w_r = W_mat.T.reshape(p, n, r)
        T = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
        return np.allclose(T, 1.0, atol=1e-9)

    def is_half_integer(mat):
        """Check if all entries are multiples of 0.5"""
        return np.allclose(mat * 2, np.round(mat * 2), atol=1e-9)

    def try_optimize(rank, max_iters=50):
        """
        Strategy: Continuous optimization of U, V, W to match the tensor T, 
        then rounding to half-integers.
        """
        if time.time() > deadline:
            return None
            
        # Target tensor T[i,k,j] = 1
        # We flatten U, V, W into a single vector x
        # dim_U = n*m*rank, dim_V = m*p*rank, dim_W = p*n*rank
        dim_U = n * m * rank
        dim_V = m * p * rank
        dim_W = p * n * rank
        total_dim = dim_U + dim_V + dim_W
        
        # Initialize with random sparse structure or schoolbook subset
        # Using random initialization is often better for global search
        np.random.seed(int(time.time() * 1000) % 10000)
        
        for attempt in range(max_iters):
            if time.time() > deadline:
                return None
                
            # Initialize x
            # We can initialize with the first 'rank' terms of schoolbook to give a good start
            # Or purely random. Let's try random with small values first, then schoolbook-based.
            if attempt % 2 == 0:
                # Schoolbook-based init: take 'rank' random columns from the 28 schoolbook terms
                # This preserves the "structure" of valid terms
                sb = schoolbook()
                U_sb = np.array(sb['U']).T # (28, 4)
                V_sb = np.array(sb['V']).T # (28, 14)
                W_sb = np.array(sb['W']).T # (28, 14)
                
                idx = np.random.choice(28, rank, replace=False)
                U_init = U_sb[idx, :].T.reshape(-1)
                V_init = V_sb[idx, :].T.reshape(-1)
                W_init = W_sb[idx, :].T.reshape(-1)
            else:
                # Random init
                U_init = np.random.randn(dim_U) * 0.5
                V_init = np.random.randn(dim_V) * 0.5
                W_init = np.random.randn(dim_W) * 0.5
                
            x0 = np.concatenate([U_init, V_init, W_init])
            
            def residuals(x):
                U_flat = x[:dim_U]
                V_flat = x[dim_U:dim_U+dim_V]
                W_flat = x[dim_U+dim_V:]
                
                U_mat = U_flat.reshape(n*m, rank)
                V_mat = V_flat.reshape(m*p, rank)
                W_mat = W_flat.reshape(p*n, rank)
                
                u_r = U_mat.T.reshape(n, m, rank)
                v_r = V_mat.T.reshape(m, p, rank)
                w_r = W_mat.T.reshape(p, n, rank)
                
                T_calc = np.einsum('ikc,kjc,jic->ijk', u_r, v_r, w_r)
                
                # Residual is T_calc - 1
                return (T_calc - 1.0).flatten()
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=200)
            except Exception:
                continue
                
            x_opt = result.x
            U_opt = x_opt[:dim_U].reshape(n*m, rank)
            V_opt = x_opt[dim_U:dim_U+dim_V].reshape(m*p, rank)
            W_opt = x_opt[dim_U+dim_V:].reshape(p*n, rank)
            
            # Check if the solution is close to half-integers
            # If not, we can try to snap to nearest half-integer and re-verify
            U_snap = np.round(U_opt * 2) / 2.0
            V_snap = np.round(V_opt * 2) / 2.0
            W_snap = np.round(W_opt * 2) / 2.0
            
            if verify(U_snap, V_snap, W_snap):
                # Convert to lists
                return {
                    "U": U_snap.astype(int).tolist() if np.allclose(U_snap, np.round(U_snap)) else (U_snap*2).astype(int).tolist(),
                    "V": V_snap.astype(int).tolist() if np.allclose(V_snap, np.round(V_snap)) else (V_snap*2).astype(int).tolist(),
                    "W": W_snap.astype(int).tolist() if np.allclose(W_snap, np.round(W_snap)) else (W_snap*2).astype(int).tolist(),
                    "rank": rank
                }
            
            # Also check if the raw solution is exact (unlikely but possible)
            if verify(U_opt, V_opt, W_opt) and is_half_integer(U_opt) and is_half_integer(V_opt) and is_half_integer(W_opt):
                 return {
                    "U": (U_opt*2).astype(int).tolist(),
                    "V": (V_opt*2).astype(int).tolist(),
                    "W": (W_opt*2).astype(int).tolist(),
                    "rank": rank
                }

        return None

    # Try ranks from 18 upwards (known lower bound for 2x2x7 is often cited around 18-20, schoolbook is 28)
    # We start from a low rank to try to find a better solution.
    # If we fail, we fall back to schoolbook (28).
    
    # Note: The score is seed/value. Seed is 28.
    # If we find rank 27, score = 28/27 > 1.
    # If we find rank 18, score = 28/18 = 1.55.
    
    for r in range(18, 28):
        if time.time() > deadline:
            break
        sol = try_optimize(r)
        if sol is not None:
            sol.pop("rank", None)
            return sol
            
    return best_sol
# EVOLVE-BLOCK-END
