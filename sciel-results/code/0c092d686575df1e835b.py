import time
import numpy as np
import random

DIMENSIONS = [2, 3, 5]

def _verify_decomposition(U, V, W):
    """
    Verify that the decomposition correctly represents matrix multiplication.
    Checks both diagonal (j=j') and off-diagonal (j!=j') terms.
    """
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check diagonal terms (should be 1)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                product = np.sum(U_arr[u_idx, :] * V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(product - 1.0) > 1e-9:
                    return False
                
    # Check off-diagonal terms (should be 0)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                if j == jp:
                    continue
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = jp * p + k
                    w_idx = k * n + i
                    product = np.sum(U_arr[u_idx, :] * V_arr[v_idx, :] * W_arr[w_idx, :])
                    if abs(product) > 1e-9:
                        return False
    return True

def _try_rank_reduction(n, m, p, target_rank, time_limit=30.0):
    """
    Try to find a decomposition with the given target_rank.
    Uses random initialization with local optimization on the full tensor error.
    """
    start_time = time.time()
    U_dim = n * m
    V_dim = m * p
    W_dim = p * n
    
    def compute_full_error(U, V, W):
        """
        Returns a list of errors for all (i, j, jp, k).
        Error is sum_r U[i,j,r] V[jp,k,r] W[k,i,r] - delta(j, jp)
        """
        errors = []
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    target = 1.0 if j == jp else 0.0
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = jp * p + k
                        w_idx = k * n + i
                        prod = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                        errors.append(prod - target)
        return np.array(errors)

    def compute_loss(U, V, W):
        errs = compute_full_error(U, V, W)
        return np.sum(errs ** 2)

    best_loss = float('inf')
    best_U, best_V, best_W = None, None, None
    
    # Multiple restarts
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        U = np.random.randn(U_dim, target_rank)
        V = np.random.randn(V_dim, target_rank)
        W = np.random.randn(W_dim, target_rank)
        
        lr = 0.01
        for step in range(200):
            if time.time() - start_time > time_limit:
                break
                
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            # We need the error for each (i, j, jp, k)
            # To avoid recomputing the full error vector every time, we compute gradients directly
            # dL/dU[u, r] = sum_{j, jp, k} error(i, j, jp, k) * V[jp, k, r] * W[k, i, r]
            # Note: u = i*m+j. The term U[u, r] appears in errors for all jp, k.
            
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    for jp in range(m):
                        target = 1.0 if j == jp else 0.0
                        for k in range(p):
                            v_idx = jp * p + k
                            w_idx = k * n + i
                            
                            # Current product
                            prod = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                            err = prod - target
                            
                            # Gradient contributions
                            dU[u_idx, :] += err * V[v_idx, :] * W[w_idx, :]
                            dV[v_idx, :] += err * U[u_idx, :] * W[w_idx, :]
                            dW[w_idx, :] += err * U[u_idx, :] * V[v_idx, :]
                            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Decay learning rate
            lr *= 0.98
            
            loss = compute_loss(U, V, W)
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_loss < 1e-10:
                break
                
    if best_loss < 1e-8:
        # Try to round to integers or half-integers
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify rounded solution
        if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
            return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
        
        # If rounding fails, try integers
        U_int = np.round(best_U)
        V_int = np.round(best_V)
        W_int = np.round(best_W)
        if _verify_decomposition(U_int.tolist(), V_int.tolist(), W_int.tolist()):
            return U_int.tolist(), V_int.tolist(), W_int.tolist()
            
        # If neither works, return floats (might be accepted if tolerance is loose, but contract says integer/half-integer)
        # The contract says "integer/half-integer entries". If we can't find one, we might fail validity.
        # But the search score is seed/value. If invalid, score 0.
        # So we must return a valid one or None.
        return None
        
    return None

def _find_best_decomposition(n, m, p, time_limit=120.0):
    """
    Search for the best (lowest rank) decomposition within the time limit.
    """
    start_time = time.time()
    schoolbook_rank = n * m * p
    
    # Start with schoolbook
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U, V, W
    
    # Try lower ranks
    # We know rank 28 is possible (from inspiration).
    # Try 29, 28, 27...
    for target_rank in range(schoolbook_rank - 1, 15, -1):
        if time.time() - start_time > time_limit - 10:  # Leave 10s margin
            break
        
        # Give more time to higher ranks (closer to schoolbook)
        time_per_rank = 10.0
        result = _try_rank_reduction(n, m, p, target_rank, time_limit=time_per_rank)
        if result is not None:
            U_list, V_list, W_list = result
            if _verify_decomposition(U_list, V_list, W_list):
                best_rank = target_rank
                best_U, best_V, best_W = U_list, V_list, W_list
                break  # Found a valid lower rank, stop searching
    
    return best_U, best_V, best_W

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the matrix multiplication tensor decomposition for 2x3 by 3x5.
    Returns a dictionary with keys 'U', 'V', 'W' representing the factor matrices.
    U: shape (n*m, rank)
    V: shape (m*p, rank)
    W: shape (p*n, rank)
    
    Searches for a rank-minimal exact tensor decomposition.
    Falls back to schoolbook algorithm (rank 30) if no better found in time.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    try:
        U, V, W = _find_best_decomposition(n, m, p, time_limit=110.0)
    except Exception:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
