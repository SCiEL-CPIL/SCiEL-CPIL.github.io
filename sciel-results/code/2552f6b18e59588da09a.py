"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 5]

import time
import random
import numpy as np

def _verify(U, V, W, n, m, p):
    """Check whether U,V,W (as lists of lists) exactly decompose the matmul tensor."""
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Build the matmul tensor T[i,m+k, j,n+l]? Actually the tensor is T[i, j, k] = delta.
    # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Check for a sample of entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = 0.0
                for c in range(r):
                    lhs += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                expected = 1.0 if True else 0.0
                # Actually T[i,j,k] = 1 if the matmul has A[i,j]*B[j,k], so the tensor entry is 1 for all (i,j,k)
                # No wait: the matmul tensor T is defined by T[i,j,k] = delta (no delta, it's just 1 for the product)
                # The standard tensor: T[i,j,k] = 1 for all i,j,k? No.
                # T_{ijk} = 1 if the (i,j) entry of A times (j,k) entry of B contributes to C[i,k].
                # So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
                # Actually yes, the matmul tensor has T[i,j,k] = 1 for all valid indices.
                if abs(lhs - 1.0) > 1e-9:
                    return False
    return True

def _verify_full(U, V, W, n, m, p):
    """Full verification using tensor contraction."""
    r = len(U[0]) if U else 0
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # Reshape
    U_tensor = Ua.reshape(n, m, r)
    V_tensor = Va.reshape(m, p, r)
    W_tensor = Wa.reshape(p, n, r)
    # T[i,j,k] = sum_c U_tensor[i,j,c] * V_tensor[j,k,c] * W_tensor[k,i,c]
    # Build T
    T = np.zeros((n, m, p), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.sum(U_tensor[i,j,:] * V_tensor[j,k,:] * W_tensor[k,i,:])
    # Expected: all ones
    if not np.allclose(T, np.ones((n,m,p)), atol=1e-9):
        return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W

def _try_random_rank(n, m, p, target_rank, n_iter=2000, seed=42):
    """Try to find a decomposition of rank target_rank via random initialization + local search."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    best_U, best_V, best_W = None, None, None
    
    for it in range(n_iter):
        r = target_rank
        U = np_rng.normal(size=(n*m, r))
        V = np_rng.normal(size=(m*p, r))
        W = np_rng.normal(size=(p*n, r))
        
        # Normalize to avoid numerical issues
        for c in range(r):
            norm = np.sqrt(np.sum(U[:,c]**2) * np.sum(V[:,c]**2) * np.sum(W[:,c]**2))
            if norm > 1e-12:
                scale = (1.0 / norm) ** (1.0/3.0)
                U[:,c] *= scale
                V[:,c] *= scale
                W[:,c] *= scale
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        if _verify_full(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list
        
        # Simple local search: perturb randomly
        for sub_it in range(50):
            c = rng.randrange(r)
            delta = 0.1 * np_rng.normal(size=(n*m,))
            U[:,c] += delta
            delta_v = 0.1 * np_rng.normal(size=(m*p,))
            V[:,c] += delta_v
            delta_w = 0.1 * np_rng.normal(size=(p*n,))
            W[:,c] += delta_w
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify_full(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
            # Undo
            U[:,c] -= delta
            V[:,c] -= delta_v
            W[:,c] -= delta_w
    
    return None

def construct():
    n, m, p = DIMENSIONS
    # Deadline guard
    start_time = time.time()
    deadline = 130.0  # seconds
    
    # Start with schoolbook
    U, V, W = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to improve from rank 44 downwards
    for target_rank in range(n*m*p - 1, 25, -1):
        if time.time() - start_time > deadline - 10:
            break
        result = _try_random_rank(n, m, p, target_rank, n_iter=200, seed=target_rank)
        if result is not None:
            U, V, W = result
            best_rank = target_rank
            break
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
