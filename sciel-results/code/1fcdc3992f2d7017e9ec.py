import time
import numpy as np
import itertools

DIMENSIONS = [2, 3, 5]

def _build_target_tensor(n, m, p):
    """
    Builds the flattened target tensor T of shape (n*m, m*p, p*n)
    T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _check_solution(U, V, W, T, tol=1e-6):
    """
    Checks if U, V, W exactly decompose T within tolerance.
    Returns error norm.
    """
    # T_decomp[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # We can compute this efficiently using einsum or broadcasting
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Term: U[:, :, None, None] * V[None, :, :, None] * W[None, None, :, :]
    # This creates a large array. 
    # Better: 
    # For each r, outer product? No.
    # Use einsum: 'ar, br, cr -> abc'
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    return np.linalg.norm(T_decomp - T)

def _snap_to_half_int(M):
    """
    Snaps matrix M to nearest half-integer.
    """
    return np.round(M * 2.0) / 2.0

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    target_rank = 25
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook solution as fallback
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Time limit guard
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    # Try to find rank 25 solution
    # We will use ALS (Alternating Least Squares)
    
    for attempt in range(100): # Max 100 restarts
        if time.time() - start_time > timeout:
            break
            
        R = target_rank
        # Random initialization
        # Scale matters. Entries are likely small.
        U = np.random.randn(n * m, R) * 0.5
        V = np.random.randn(m * p, R) * 0.5
        W = np.random.randn(p * n, R) * 0.5
        
        # Precompute dimensions for ALS
        B = m * p
        C = p * n
        A = n * m
        
        # Reshape T for easier access
        # T is (A, B, C)
        
        converged = False
        for iter in range(200): # Max 200 iterations per restart
            if time.time() - start_time > timeout:
                break
                
            # Solve for U
            # For each row a of U, we solve U[a] * M = T[a]
            # M[r, bc] = V[b, r] * W[c, r]
            # T[a] is flattened T[a, :, :]
            
            # Construct M: (R, B*C)
            # M[r, b*C + c] = V[b, r] * W[c, r]
            # This can be done by:
            # V is (B, R), W is (C, R)
            # M is (R, B*C)
            # M = np.einsum('br,cr->rbc', V, W).reshape(R, B*C)
            M_U = np.einsum('br,cr->rbc', V, W).reshape(R, B * C)
            
            # T_flat is (A, B*C)
            T_flat = T.reshape(A, B * C)
            
            # Solve U[a] M_U = T_flat[a] for each a
            # U is (A, R)
            # We want U[a] such that U[a] @ M_U.T = T_flat[a]
            # So U[a] = T_flat[a] @ M_U @ (M_U.T @ M_U)^-1 ?
            # Use lstsq: min || X M_U.T - Y ||
            # X is (R,), Y is (B*C,)
            # np.linalg.lstsq(M_U.T, Y) gives X
            for a in range(A):
                # Solve for U[a]
                try:
                    U[a], _, _, _ = np.linalg.lstsq(M_U.T, T_flat[a], rcond=None)
                except:
                    pass
                    
            # Solve for V
            # For each row b of V, we solve V[b] * M_V = T[:, b, :]
            # M_V[r, ac] = U[a, r] * W[c, r]
            M_V = np.einsum('ar,cr->rac', U, W).reshape(R, A * C)
            T_flat_V = T.reshape(A, B, C).transpose(1, 0, 2).reshape(B, A * C)
            
            for b in range(B):
                try:
                    V[b], _, _, _ = np.linalg.lstsq(M_V.T, T_flat_V[b], rcond=None)
                except:
                    pass
                    
            # Solve for W
            # For each row c of W, we solve W[c] * M_W = T[:, :, c]
            # M_W[r, ab] = U[a, r] * V[b, r]
            M_W = np.einsum('ar,br->rab', U, V).reshape(R, A * B)
            T_flat_W = T.reshape(A, B, C).transpose(2, 0, 1).reshape(C, A * B)
            
            for c in range(C):
                try:
                    W[c], _, _, _ = np.linalg.lstsq(M_W.T, T_flat_W[c], rcond=None)
                except:
                    pass
            
            # Check error
            err = _check_solution(U, V, W, T)
            if err < 1e-4:
                # Snap to half-integers
                U_snap = _snap_to_half_int(U)
                V_snap = _snap_to_half_int(V)
                W_snap = _snap_to_half_int(W)
                
                err_snap = _check_solution(U_snap, V_snap, W_snap, T)
                if err_snap < 1e-6:
                    # Found valid rank 25 solution
                    best_rank = R
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    converged = True
                    break
            elif err < 1e-8:
                # Very low error, might be exact but floating point issues
                U_snap = _snap_to_half_int(U)
                V_snap = _snap_to_half_int(V)
                W_snap = _snap_to_half_int(W)
                err_snap = _check_solution(U_snap, V_snap, W_snap, T)
                if err_snap < 1e-6:
                    best_rank = R
                    best_U = U_snap
                    best_V = V_snap
                    best_W = W_snap
                    converged = True
                    break
                    
        if converged:
            break
            
    # If we found a better rank, return it. Otherwise return schoolbook.
    if best_rank < rank_schoolbook:
        # Convert to lists of lists if required? 
        # The seed returns lists of lists.
        # "factor matrices of shapes ... with integer/half-integer entries"
        # The interface example in seed uses lists.
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    else:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
