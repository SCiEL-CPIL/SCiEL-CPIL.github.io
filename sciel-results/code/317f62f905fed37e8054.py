import numpy as np
import time

DIMENSIONS = [3, 3, 7]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - START_TIME

def _time_remaining():
    return TIME_LIMIT - SAFETY_MARGIN - _elapsed()

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
    equals 1 if (u,v,w) = (i*m+j, j*p+k, k*n+i) for some (i,j,k), else 0.
    """
    rank = U.shape[1]
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if abs(T[u, v, w] - 1.0) > tol:
                    return False
    
    # Check all other entries are zero
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                i = u // m
                j = u % m
                j_prime = v // p
                k_prime = v % p
                k_double_prime = w // n
                i_double_prime = w % n
                if i == i_double_prime and j == j_prime and k_prime == k_double_prime:
                    continue
                if abs(T[u, v, w]) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _build_flattening_matrix(n, m, p):
    """
    Build the matrix F of shape (n*m*p, (n*m)*(m*p) + (m*p)*(p*n) + (p*n)*(n*m))
    such that for a candidate rank-r decomposition with vectors u_r, v_r, w_r:
    The flattening along mode-1 gives: for each (v,w) pair, sum_r u_r * (v_r * w_r) = target.
    
    Actually, let's use the standard approach:
    T[u,v,w] = sum_r U[u,r] V[v,r] W[w,r]
    
    Fix (v,w). Then T[:,v,w] = U * (V[v,:] * W[w,:]) where * is element-wise.
    So for each (v,w), we have: U @ diag(V[v,:] * W[w,:]) = T[:,v,w]
    
    Equivalently, let's vectorize. Let U be (nm x r), V be (mp x r), W be (pn x r).
    For a given (v,w), the vector T[:,v,w] (length nm) must equal U @ (V[v,:] * W[w,:]).
    
    This is linear in the entries of U, V, W. Let's set up the system.
    
    Let's use the mode-1 flattening: 
    Flatten T along mode 1: T_flat1 has shape (nm, mp*pn).
    T_flat1[u, v*pn+w] = T[u,v,w]
    
    T_flat1 = U @ (V ⊙ W) where ⊙ is the Khatri-Rao product (column-wise outer product).
    Specifically, (V ⊙ W)[(v*pn+w), r] = V[v,r] * W[w,r]
    
    So T_flat1 = U @ (V ⊙ W)
    
    Similarly, T_flat2 = V @ (W ⊙ U) and T_flat3 = W @ (U ⊙ V)
    
    For rank r, we need to find U (nm x r), V (mp x r), W (pn x r) such that:
    U @ (V ⊙ W) = T_flat1
    
    This is a bilinear system. Let's use the approach from inspiration 2:
    Fix two matrices and solve for the third, or use iterative methods.
    
    Actually, a cleaner approach: for a given rank r, we can try to find a decomposition
    by setting up the linear system for each "slice" and using the fact that the columns
    of U, V, W span the appropriate subspaces.
    
    Let me use a different strategy: 
    1. Start with schoolbook (rank = n*m*p = 63)
    2. Try to reduce rank by checking if the tensor has lower rank
    3. Use the fact that rank(T) <= r iff there exist U,V,W of rank r
    
    For small problems like this, we can use the following approach:
    - The tensor T has a unique flattening T_flat1 of shape (9, 441)
    - rank(T) >= rank(T_flat1) = 9
    - We can try ranks from the lower bound up to schoolbook
    
    To test if rank r is achievable, we can use the following:
    - Assume W is fixed (or parameterized)
    - Then U @ (V ⊙ W) = T_flat1 is linear in U and V
    - We can iterate: fix V,W, solve for U; fix U,W, solve for V; fix U,V, solve for W
    
    Let's implement this iterative approach.
    """
    pass

def _try_rank_decomposition(n, m, p, r, max_iters=200, tol=1e-8, seed=42):
    """
    Try to find a rank-r decomposition using iterative linear algebra.
    Returns (U, V, W) if successful, None otherwise.
    """
    if _time_remaining() < 5:
        return None
    
    np.random.seed(seed)
    
    # Initialize with random matrices
    U = np.random.randn(n*m, r)
    V = np.random.randn(m*p, r)
    W = np.random.randn(p*n, r)
    
    # Build target tensors
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1.0
    
    # Flattening along mode 1: T1[u, v*pn + w] = T[u,v,w]
    T1 = T.reshape(n*m, m*p*p*n)
    # Flattening along mode 2: T2[v, u*pn + w] = T[u,v,w]  
    T2 = np.zeros((m*p, n*m*p*n))
    # Flattening along mode 3: T3[w, u*mp + v] = T[u,v,w]
    T3 = np.zeros((p*n, n*m*m*p))
    
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                val = T[u, v, w]
                T1[u, v * p*n + w] = val
                T2[v, u * p*n + w] = val
                T3[w, u * m*p + v] = val
    
    for it in range(max_iters):
        if _time_remaining() < 3:
            return None
        
        # Fix V and W, solve for U: U @ (V ⊙ W) = T1
        # (V ⊙ W) has shape (m*p*p*n, r) where [(v*pn+w), r] = V[v,r]*W[w,r]
        VW = np.zeros((m*p*p*n, r))
        for v in range(m*p):
            for w in range(p*n):
                VW[v * p*n + w, :] = V[v, :] * W[w, :]
        
        # Solve U @ VW = T1 -> U = T1 @ pinv(VW)
        try:
            U_new = T1 @ np.linalg.pinv(VW)
        except Exception:
            return None
        
        # Fix U and W, solve for V: V @ (W ⊙ U) = T2
        # (W ⊙ U) has shape (p*n*n*m, r) where [(w*nm+u), r] = W[w,r]*U[u,r]
        WU = np.zeros((p*n*n*m, r))
        for w in range(p*n):
            for u in range(n*m):
                WU[w * n*m + u, :] = W[w, :] * U[u, :]
        
        try:
            V_new = T2 @ np.linalg.pinv(WU)
        except Exception:
            return None
        
        # Fix U and V, solve for W: W @ (U ⊙ V) = T3
        # (U ⊙ V) has shape (n*m*m*p, r) where [(u*mp+v), r] = U[u,r]*V[v,r]
        UV = np.zeros((n*m*m*p, r))
        for u in range(n*m):
            for v in range(m*p):
                UV[u * m*p + v, :] = U[u, :] * V[v, :]
        
        try:
            W_new = T3 @ np.linalg.pinv(UV)
        except Exception:
            return None
        
        # Check convergence
        diff_U = np.linalg.norm(U_new - U) / (np.linalg.norm(U) + 1e-15)
        diff_V = np.linalg.norm(V_new - V) / (np.linalg.norm(V) + 1e-15)
        diff_W = np.linalg.norm(W_new - W) / (np.linalg.norm(W) + 1e-15)
        
        U, V, W = U_new, V_new, W_new
        
        if max(diff_U, diff_V, diff_W) < tol:
            # Verify the solution
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                return U, V, W
    
    # Final verification
    if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
        return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 7
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 63
    
    # Lower bound on rank
    lower_bound = max(n*m, m*p, p*n)  # max(9, 21, 21) = 21
    
    # Try ranks from lower_bound up to schoolbook_rank
    # Start with a few promising ranks
    ranks_to_try = list(range(lower_bound, schoolbook_rank + 1))
    
    # Try to find the minimal rank
    best_result = None
    best_rank = schoolbook_rank
    
    # First, get the schoolbook decomposition as fallback
    U_sb, V_sb, W_sb = _schoolbook_decomposition(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    
    # Try to find lower rank decompositions
    # Use multiple random seeds for better coverage
    for r in ranks_to_try:
        if _time_remaining() < 10:
            break
        
        if r >= best_rank:
            continue
        
        # Try a few random initializations
        for seed in range(5):
            if _time_remaining() < 5:
                break
            
            result = _try_rank_decomposition(n, m, p, r, max_iters=100, seed=seed)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
                    best_result = (U, V, W)
                    best_rank = r
                    break  # Found rank r, try next lower rank
    
    U, V, W = best_result
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
