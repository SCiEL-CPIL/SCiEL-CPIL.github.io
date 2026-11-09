"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140  # safety margin before 150s kill
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    # Try to find better decomposition
    # The constraint: for all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Strategy: try various ranks, use optimization
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if decomposition is valid. U:(nm x R), V:(mp x R), W:(pn x R)"""
        R = U.shape[1]
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def try_rank(R, max_iters=2000, seed=None):
        """Try to find a valid decomposition of rank R."""
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()
        
        # Initialize
        U = rng.standard_normal((n*m, R)) * 0.5
        V = rng.standard_normal((m*p, R)) * 0.5
        W = rng.standard_normal((p*n, R)) * 0.5
        
        # Use alternating least squares-like approach
        # For each r, we have U_col * V_col * W_col contributing
        # We want sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = 1
        
        # Reformulate: for each (i,j,k), the constraint is:
        # sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
        # Let A_{ij, r} = U[i*m+j, r] * V[j*p+k, r] ... this is getting complex
        
        # Alternative: treat it as a factorization problem
        # Build the tensor T of shape (n*m, m*p, p*n) where T[a,b,c] = 1 if 
        # a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, else 0
        # Then we need T = sum_r u_r ⊗ v_r ⊗ w_r
        
        # Actually T[a,b,c] = 1 iff there exist i,j,k with a=i*m+j, b=j*p+k, c=k*n+i
        # For given a=i*m+j, b=j*p+k, c=k*n+i, we need consistency: j from a, k from b, i from c
        # j = a % m, i = a // m, k = b % p, and c should equal k*n + i
        
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    T[a, b, c] = 1.0
        
        # Now find rank-R decomposition: T = sum_{r=0}^{R-1} u_r ⊗ v_r ⊗ w_r
        # Use iterative method: fix two, solve for third
        
        U = rng.standard_normal((n*m, R))
        V = rng.standard_normal((m*p, R))
        W = rng.standard_normal((p*n, R))
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
            
            # Update U: for each r, minimize ||T - sum_r' u_r'⊗v_r'⊗w_r'||_F
            # Fix V, W. For each r, the contribution is u_r ⊗ V[:,r] ⊗ W[:,r]
            # Residual R = T - sum_{r'} u_{r'}⊗V_{:,r'}⊗W_{:,r'}
            # For updating u_r: we want to minimize ||R + u_r⊗V_r⊗W_r - u_old_r⊗V_r⊗W_r||
            # This is a linear least squares in u_r
            
            # Compute residual
            Res = T.copy()
            for r in range(R):
                Res -= np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            # Update each U column
            for r in range(R):
                # Remove current contribution
                Res += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                # We want to find u_r minimizing ||Res - u_r ⊗ V_r ⊗ W_r||_F^2
                # = ||Res - outer(u_r, outer(V_r, W_r))||_F^2
                # Let M = Res reshaped... Actually:
                # (u⊗v⊗w)[a,b,c] = u[a]*v[b]*w[c]
                # So for fixed v,w, we're solving: minimize ||Res - u ⊗ (v⊗w)||^2
                # This is: for each (b,c), Res[:,b,c] ≈ u * (v[b]*w[c])
                # So u[a] = sum_{b,c} Res[a,b,c] * v[b]*w[c] / sum_{b,c} (v[b]*w[c])^2
                vw = np.outer(V[:,r], W[:,r])  # shape (m*p, p*n)
                denom = np.sum(vw**2)
                if denom > 1e-12:
                    U[:,r] = np.sum(Res * vw[np.newaxis, :, :], axis=(1,2)) / denom
                Res -= np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            # Update V
            Res = T.copy()
            for r in range(R):
                Res -= np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            for r in range(R):
                Res += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                uw = np.outer(U[:,r], W[:,r])  # shape (n*m, p*n)
                denom = np.sum(uw**2)
                if denom > 1e-12:
                    V[:,r] = np.sum(Res * uw[:, np.newaxis, :], axis=(0,2)) / denom
                Res -= np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            # Update W
            Res = T.copy()
            for r in range(R):
                Res -= np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            for r in range(R):
                Res += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
                uv = np.outer(U[:,r], V[:,r])  # shape (n*m, m*p)
                denom = np.sum(uv**2)
                if denom > 1e-12:
                    W[:,r] = np.sum(Res * uv[:, :, np.newaxis], axis=(0,1)) / denom
                Res -= np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
            
            # Check convergence
            if it % 100 == 0:
                err = np.sum(Res**2)
                if err < 1e-12:
                    break
        
        # Check if solution is valid
        if check_decomposition(U, V, W, tol=1e-6):
            # Try to clean up to integers/half-integers
            U_clean = np.round(U * 2) / 2
            V_clean = np.round(V * 2) / 2
            W_clean = np.round(W * 2) / 2
            if check_decomposition(U_clean, V_clean, W_clean, tol=1e-6):
                U_out = [[float(x) for x in row] for row in U_clean]
                V_out = [[float(x) for x in row] for row in V_clean]
                W_out = [[float(x) for x in row] for row in W_clean]
                return U_out, V_out, W_out
            else:
                # Try integer rounding
                U_clean = np.round(U)
                V_clean = np.round(V)
                W_clean = np.round(W)
                if check_decomposition(U_clean, V_clean, W_clean, tol=1e-6):
                    U_out = [[float(x) for x in row] for row in U_clean]
                    V_out = [[float(x) for x in row] for row in V_clean]
                    W_out = [[float(x) for x in row] for row in W_clean]
                    return U_out, V_out, W_out
        return None
    
    # Try ranks from low to high
    for R in range(20, min(60, best_rank)):
        if time.time() > deadline - 20:
            break
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            result = try_rank(R, max_iters=500, seed=R*100+attempt)
            if result is not None:
                U, V, W = result
                best = {"U": U, "V": V, "W": W}
                best_rank = R
                break
    
    return best
# EVOLVE-BLOCK-END
