import numpy as np
import time
from itertools import product
from fractions import Fraction

def construct():
    n, m, p = 2, 4, 5
    deadline = time.time() + 135  # safety margin
    
    # Build the target tensor T[a,b,c]
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a,b,c] = 1.0
    
    # Verify function
    def verify(U, V, W, tol=1e-8):
        n, m, p = 2, 4, 5
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    val = np.sum(U[a,:] * V[b,:] * W[c,:])
                    if abs(val - 1.0) > tol:
                        return False
        # Check all zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if not (i_a == i_c and j_a == j_b and k_b == k_c):
                        val = np.sum(U[a,:] * V[b,:] * W[c,:])
                        if abs(val) > tol:
                            return False
        return True
    
    # Schoolbook decomposition (rank = n*m*p = 40)
    def get_schoolbook():
        rank = n*m*p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_school, V_school, W_school = get_schoolbook()
    
    # Try to find lower rank decompositions using structured search
    # The key insight: we can use symmetries and structure of the problem
    
    # Strategy 1: Try rank 20-39 with randomized ALS + rounding
    best_rank = n*m*p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try different ranks from low to high
    for target_rank in range(20, 40):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 20:
                break
            
            # Initialize with random factors
            np.random.seed(hash((target_rank, attempt)) % 2**32)
            U_init = np.random.randn(n*m, target_rank) * 0.1
            V_init = np.random.randn(m*p, target_rank) * 0.1
            W_init = np.random.randn(p*n, target_rank) * 0.1
            
            U_als = U_init.copy()
            V_als = V_init.copy()
            W_als = W_init.copy()
            
            # Run ALS iterations
            m_dim = m * p  # 20
            n_dim = p * n  # 10
            r = target_rank
            
            for iteration in range(50):
                if time.time() > deadline - 20:
                    break
                
                # Update U
                A_u = np.zeros((m_dim * n_dim, r))
                for rr in range(r):
                    A_u[:, rr] = np.outer(V_als[:, rr], W_als[:, rr]).flatten()
                
                for a in range(n * m):
                    rhs = T[a].flatten()
                    try:
                        u_a, _, _, _ = np.linalg.lstsq(A_u, rhs, rcond=None)
                        U_als[a, :] = u_a
                    except:
                        pass
                
                # Update V
                A_v = np.zeros((n * m * n_dim, r))
                for rr in range(r):
                    A_v[:, rr] = np.outer(U_als[:, rr], W_als[:, rr]).flatten()
                
                for b in range(m * p):
                    rhs = T[:, b, :].flatten()
                    try:
                        v_b, _, _, _ = np.linalg.lstsq(A_v, rhs, rcond=None)
                        V_als[b, :] = v_b
                    except:
                        pass
                
                # Update W
                A_w = np.zeros((n * m * m_dim, r))
                for rr in range(r):
                    A_w[:, rr] = np.outer(U_als[:, rr], V_als[:, rr]).flatten()
                
                for c in range(p * n):
                    rhs = T[:, :, c].flatten()
                    try:
                        w_c, _, _, _ = np.linalg.lstsq(A_w, rhs, rcond=None)
                        W_als[c, :] = w_c
                    except:
                        pass
            
            # Try to round to simple rational values
            def try_round(mat, scale=2):
                mat_r = mat.copy()
                for i in range(mat_r.shape[0]):
                    for j in range(mat_r.shape[1]):
                        val = mat_r[i,j]
                        # Try rounding to multiples of 1/scale
                        for denom in [1, scale, 4, 8]:
                            rounded = round(val * denom) / denom
                            if abs(val - rounded) < 1e-5:
                                mat_r[i,j] = rounded
                                break
                return mat_r
            
            U_rounded = try_round(U_als)
            V_rounded = try_round(V_als)
            W_rounded = try_round(W_als)
            
            if verify(U_rounded, V_rounded, W_rounded):
                best_rank = target_rank
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
                found = True
                break
            elif verify(U_als, V_als, W_als):
                # Even if not rounded, if it's valid, use it
                best_rank = target_rank
                best_U, best_V, best_W = U_als, V_als, W_als
                found = True
                break
        
        if found:
            # Try to go lower
            continue
    
    # If we found something better than schoolbook, return it
    if best_rank < n*m*p:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Otherwise return schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
