"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 24
    
    # Build the tensor T[i,j,k] for C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T is in R^{n*m x m*p x p*n}
    # T[(i,j),(j,k),(k,i)] = 1
    
    # We want to find rank r such that T = sum_{c=1}^r U[:,c] V[:,c] W[:,c] (outer product)
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    
    # The schoolbook decomposition has rank 24.
    # Can we do better?
    # For 2x3 * 3x4, the minimum rank is known to be at most 24 (schoolbook).
    # Let's check if there's a known lower rank.
    # Actually, for matrix multiplication, the tensor rank is the bilinear complexity.
    # For 2x3 and 3x4, this is a specific case.
    # Let's try to find a decomposition with rank < 24.
    
    # The tensor T has entries T[a,b,c] where:
    # a = i*m + j (i in 0..n-1, j in 0..m-1)
    # b = j*p + k (j in 0..m-1, k in 0..p-1)
    # c = k*n + i (k in 0..p-1, i in 0..n-1)
    # T[a,b,c] = 1 if the indices are consistent (same j for a,b and same i,k for a,c)
    
    # Let's construct T as a 3D array
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition
    U_school = np.zeros((n*m, rank_schoolbook), dtype=np.float64)
    V_school = np.zeros((m*p, rank_schoolbook), dtype=np.float64)
    W_school = np.zeros((p*n, rank_schoolbook), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1.0
                V_school[j*p + k, col] = 1.0
                W_school[k*n + i, col] = 1.0
                col += 1
    
    # Try to find a lower rank decomposition using random initialization and optimization
    # We'll use a simple approach: try ranks from 23 down to some minimum
    
    deadline = time.time() + 140.0  # Leave 10s margin
    
    best_rank = rank_schoolbook
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find rank 23 decomposition
    # Use random initialization and gradient-free optimization (e.g., alternating least squares)
    
    def check_decomposition(U, V, W, T, tol=1e-8):
        """Check if U, V, W decompose T."""
        r = U.shape[1]
        for c in range(r):
            pass
        # Compute T_decomposed = sum_c U[:,c] outer V[:,c] outer W[:,c]
        # This is expensive, so we'll do it efficiently
        T_dec = np.zeros_like(T)
        for c in range(r):
            T_dec += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
        return np.allclose(T_dec, T, atol=tol)
    
    def als_decompose(T, rank, max_iter=500, tol=1e-10, seed=42):
        """Alternating Least Squares for tensor decomposition."""
        rng = np.random.RandomState(seed)
        n_a, n_b, n_c = T.shape
        U = rng.randn(n_a, rank) * 0.1
        V = rng.randn(n_b, rank) * 0.1
        W = rng.randn(n_c, rank) * 0.1
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            # Update U
            for c in range(rank):
                # T = sum_c U[:,c] outer V[:,c] outer W[:,c]
                # For fixed c, we want to minimize ||T - sum_d U_d V_d W_d||
                # This is a linear least squares problem in U[:,c]
                # T = U[:,c] outer (V[:,c] outer W[:,c]) + rest
                # rest = sum_{d!=c} U[:,d] outer V[:,d] outer W[:,d]
                
                # Compute the residual
                T_rest = np.zeros_like(T)
                for d in range(rank):
                    if d != c:
                        T_rest += np.einsum('a,b,c->abc', U[:,d], V[:,d], W[:,d])
                T_target = T - T_rest
                
                # Now we need U[:,c] such that U[:,c] outer (V[:,c] outer W[:,c]) = T_target
                # Let v_w = V[:,c] outer W[:,c], shape (n_b, n_c)
                # Then U[:,c] outer v_w = T_target
                # This means T_target[a,b,c] = U[a,c] * v_w[b,c]
                # So U[a,c] = sum_{b,c} T_target[a,b,c] * v_w[b,c] / ||v_w||^2
                # More precisely: for each a, sum_{b,c} T_target[a,b,c] * v_w[b,c] = U[a,c] * sum_{b,c} v_w[b,c]^2
                
                v_w = np.outer(V[:,c], W[:,c])
                denom = np.sum(v_w**2)
                if denom < 1e-12:
                    continue
                # T_target reshaped as (n_a, n_b*n_c)
                T_target_2d = T_target.reshape(n_a, n_b * n_c)
                v_w_2d = v_w.reshape(n_b * n_c)
                # U[:,c] = T_target_2d @ v_w_2d / denom
                U[:,c] = (T_target_2d @ v_w_2d) / denom
            
            # Update V
            for c in range(rank):
                T_rest = np.zeros_like(T)
                for d in range(rank):
                    if d != c:
                        T_rest += np.einsum('a,b,c->abc', U[:,d], V[:,d], W[:,d])
                T_target = T - T_rest
                
                u_w = np.outer(U[:,c], W[:,c])
                denom = np.sum(u_w**2)
                if denom < 1e-12:
                    continue
                T_target_2d = T_target.reshape(n_a * n_b, n_c)
                u_w_2d = u_w.reshape(n_a * n_b)
                V[:,c] = (T_target_2d @ u_w_2d) / denom
            
            # Update W
            for c in range(rank):
                T_rest = np.zeros_like(T)
                for d in range(rank):
                    if d != c:
                        T_rest += np.einsum('a,b,c->abc', U[:,d], V[:,d], W[:,d])
                T_target = T - T_target if False else T - T_rest
                
                u_v = np.outer(U[:,c], V[:,c])
                denom = np.sum(u_v**2)
                if denom < 1e-12:
                    continue
                T_target_2d = T_target.reshape(n_a * n_b, n_c)
                u_v_2d = u_v.reshape(n_a * n_b)
                W[:,c] = (T_target_2d @ u_v_2d) / denom
            
            # Check convergence
            T_dec = np.zeros_like(T)
            for c in range(rank):
                T_dec += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
            err = np.linalg.norm(T_dec - T) / (np.linalg.norm(T) + 1e-12)
            if err < tol:
                break
        
        return U, V, W, err
    
    # Try ranks from 23 down to 20
    for target_rank in range(rank_schoolbook - 1, 19, -1):
        if time.time() > deadline - 10:
            break
        
        found = False
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            U, V, W, err = als_decompose(T, target_rank, max_iter=300, tol=1e-10, seed=42 + attempt)
            
            if err < 1e-8:
                # Verify more carefully
                T_dec = np.zeros_like(T)
                for c in range(target_rank):
                    T_dec += np.einsum('a,b,c->abc', U[:,c], V[:,c], W[:,c])
                if np.allclose(T_dec, T, atol=1e-8):
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    found = True
                    break
        
        if found:
            # Try to go lower
            continue
        else:
            # If we can't find rank target_rank, we might still try lower
            # But for now, let's stop if we fail at the first lower rank
            # Actually, let's keep trying lower ranks
            pass
    
    # Convert to lists with integer/half-integer entries if possible
    # The problem says "integer/half-integer entries"
    # Let's round to nearest 0.5 if close
    
    def to_half_int(x):
        """Round to nearest half-integer if close."""
        rounded = np.round(x * 2) / 2
        if np.allclose(x, rounded, atol=1e-6):
            return rounded
        return x
    
    U_out = to_half_int(best_U)
    V_out = to_half_int(best_V)
    W_out = to_half_int(best_W)
    
    # Convert to Python lists
    U_list = U_out.tolist()
    V_list = V_out.tolist()
    W_list = W_out.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
