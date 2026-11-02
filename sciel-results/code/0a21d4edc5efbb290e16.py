import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 5]

def _build_tensor_indices(n, m, p):
    """Return list of (a, b, c) tuples for the non-zero entries of the matmul tensor."""
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                indices.append((a, b, c))
    return indices

def _verify_decomposition(U, V, W, n, m, p, indices, tol=1e-7):
    """
    Verify that the decomposition exactly reproduces the matmul tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Tensor T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
    Non-zero entries (from indices) should be 1.0
    All other entries should be 0.0
    """
    r = U.shape[1]
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Check non-zero entries
    for (a, b, c) in indices:
        val = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(val - 1.0) > tol:
            return False
    
    # Check zero entries - sample a subset for speed, but check all for correctness
    # Build a set of non-zero (a,b,c) for quick lookup
    nonzero_set = set(indices)
    
    # For efficiency, we check all combinations but skip non-zero ones
    # Total: dim_u * dim_v * dim_w = 12*20*15 = 3600
    # Non-zero: 60
    # This is fast enough
    
    for a in range(dim_u):
        for b in range(dim_v):
            for c in range(dim_w):
                if (a, b, c) in nonzero_set:
                    continue
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
    return True

def _verify_decomposition_fast(U, V, W, n, m, p, indices, tol=1e-7):
    """Faster verification using vectorized operations where possible."""
    r = U.shape[1]
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    nonzero_set = set(indices)
    
    # Pre-compute V*W outer products for each t? No, that's expensive.
    # Instead, for each a, compute U[a,:] and check against all (b,c)
    
    for a in range(dim_u):
        Ua = U[a, :]  # shape (r,)
        # For all b, c: sum_t Ua[t] V[b,t] W[c,t]
        # = (Ua * V[b,:]) dot W[c,:] for each b,c
        # Vectorize over c for each b:
        for b in range(dim_v):
            Vb = V[b, :]  # shape (r,)
            # product = Ua * Vb  shape (r,)
            # Then for each c: dot(product, W[c,:])
            prod = Ua * Vb  # shape (r,)
            # Check all c
            for c in range(dim_w):
                if (a, b, c) in nonzero_set:
                    val = np.dot(prod, W[c, :])
                    if abs(val - 1.0) > tol:
                        return False
                else:
                    val = np.dot(prod, W[c, :])
                    if abs(val) > tol:
                        return False
    return True

def _construct_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition of rank n*m*p."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    rank = n * m * p
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    return U, V, W

def _try_rank_reduction(n, m, p, target_rank, max_iterations=200, timeout=None, start_time=None):
    """
    Try to find a decomposition of rank target_rank.
    Uses alternating least squares with integer snapping.
    """
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    indices = _build_tensor_indices(n, m, p)
    
    if timeout is not None and start_time is not None:
        if time.time() - start_time > timeout:
            return None
    
    # Build target tensor as sparse representation
    # T[a,b,c] = 1 for (a,b,c) in indices, 0 otherwise
    
    # We'll use ALS: fix V,W and solve for U, etc.
    # For each a: sum_t U[a,t] * (V[b,t] * W[c,t]) = T[a,b,c]
    # This is a linear system in U[a,:] for each a.
    
    # Pre-compute the structure
    # For solving U: A_U[b*dim_w + c, t] = V[b,t] * W[c,t]
    # For solving V: A_V[a*dim_w + c, t] = U[a,t] * W[c,t]
    # For solving W: A_W[a*dim_v + b, t] = U[a,t] * V[b,t]
    
    # Initialize randomly
    np.random.seed(12345)
    
    best_result = None
    
    for trial in range(10):
        if timeout is not None and start_time is not None:
            if time.time() - start_time > timeout:
                return best_result
        
        # Random integer initialization
        U = np.random.randint(-1, 2, size=(dim_u, target_rank)).astype(float)
        V = np.random.randint(-1, 2, size=(dim_v, target_rank)).astype(float)
        W = np.random.randint(-1, 2, size=(dim_w, target_rank)).astype(float)
        
        for it in range(max_iterations):
            if timeout is not None and start_time is not None:
                if time.time() - start_time > timeout:
                    return best_result
            
            # Solve for U
            # A_U is (dim_v * dim_w) x target_rank
            # A_U[b*dim_w + c, t] = V[b,t] * W[c,t]
            A_U = np.zeros((dim_v * dim_w, target_rank))
            for b in range(dim_v):
                for c in range(dim_w):
                    A_U[b * dim_w + c, :] = V[b, :] * W[c, :]
            
            # Target for U: T_all[a, b*dim_w+c] = T[a,b,c]
            T_all = np.zeros((dim_u, dim_v * dim_w))
            for (a, b, c) in indices:
                T_all[a, b * dim_w + c] = 1.0
            
            # Solve: A_U @ U[a,:].T = T_all[a,:].T for each a
            # Stack: A_U @ U.T = T_all.T
            try:
                U_sol, residuals, rank_u, sv = np.linalg.lstsq(A_U, T_all.T, rcond=None)
                U_new = U_sol.T  # shape (dim_u, target_rank)
            except np.linalg.LinAlgError:
                U_new = U.copy()
            
            # Snap to integers
            U_snapped = np.round(U_new)
            
            # Check if snapping is close
            if np.max(np.abs(U_new - U_snapped)) < 0.5:
                U = U_snapped
            
            # Solve for V
            A_V = np.zeros((dim_u * dim_w, target_rank))
            for a in range(dim_u):
                for c in range(dim_w):
                    A_V[a * dim_w + c, :] = U[a, :] * W[c, :]
            
            T_all_V = np.zeros((dim_v, dim_u * dim_w))
            for (a, b, c) in indices:
                T_all_V[b, a * dim_w + c] = 1.0
            
            try:
                V_sol, residuals, rank_v, sv = np.linalg.lstsq(A_V, T_all_V.T, rcond=None)
                V_new = V_sol.T
            except np.linalg.LinAlgError:
                V_new = V.copy()
            
            V_snapped = np.round(V_new)
            if np.max(np.abs(V_new - V_snapped)) < 0.5:
                V = V_snapped
            
            # Solve for W
            A_W = np.zeros((dim_u * dim_v, target_rank))
            for a in range(dim_u):
                for b in range(dim_v):
                    A_W[a * dim_v + b, :] = U[a, :] * V[b, :]
            
            T_all_W = np.zeros((dim_w, dim_u * dim_v))
            for (a, b, c) in indices:
                T_all_W[c, a * dim_v + b] = 1.0
            
            try:
                W_sol, residuals, rank_w, sv = np.linalg.lstsq(A_W, T_all_W.T, rcond=None)
                W_new = W_sol.T
            except np.linalg.LinAlgError:
                W_new = W.copy()
            
            W_snapped = np.round(W_new)
            if np.max(np.abs(W_new - W_snapped)) < 0.5:
                W = W_snapped
            
            # Check convergence
            if _verify_decomposition_fast(U, V, W, n, m, p, indices):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # Early exit if no improvement
            if it > 0:
                # Check if U, V, W changed
                pass
    
    return best_result

def _greedy_rank_reduction(n, m, p, max_rank, timeout=100.0):
    """
    Try to reduce rank from max_rank down to max_rank-1, max_rank-2, etc.
    Returns the best decomposition found.
    """
    start_time = time.time()
    schoolbook_rank = n * m * p
    
    # Start from schoolbook and try to reduce
    for target_rank in range(max_rank, 49, -1):
        if time.time() - start_time > timeout:
            break
        
        result = _try_rank_reduction(n, m, p, target_rank, 
                                      max_iterations=100,
                                      timeout=timeout,
                                      start_time=start_time)
        if result is not None:
            return result
    
    return None

def _structured_search(n, m, p, timeout=100.0):
    """
    Try structured decompositions that might achieve lower rank.
    For example, use the identity that matmul can sometimes be done
    with fewer operations using clever linear combinations.
    """
    start_time = time.time()
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    indices = _build_tensor_indices(n, m, p)
    
    # Try rank 50-59 with more diverse initializations
    for target_rank in range(59, 49, -1):
        if time.time() - start_time > timeout:
            break
        
        # Try different random seeds and initialization strategies
        for seed in range(20):
            if time.time() - start_time > timeout:
                break
            
            np.random.seed(seed * 1000 + 42)
            
            # Strategy 1: Random integers
            U = np.random.randint(-2, 3, size=(dim_u, target_rank)).astype(float)
            V = np.random.randint(-2, 3, size=(dim_v, target_rank)).astype(float)
            W = np.random.randint(-2, 3, size=(dim_w, target_rank)).astype(float)
            
            for it in range(50):
                if time.time() - start_time > timeout:
                    break
                
                # Solve for U
                A_U = np.zeros((dim_v * dim_w, target_rank))
                for b in range(dim_v):
                    for c in range(dim_w):
                        A_U[b * dim_w + c, :] = V[b, :] * W[c, :]
                
                T_all = np.zeros((dim_u, dim_v * dim_w))
                for (a, b, c) in indices:
                    T_all[a, b * dim_w + c] = 1.0
                
                try:
                    U_sol, _, _, _ = np.linalg.lstsq(A_U, T_all.T, rcond=None)
                    U_new = U_sol.T
                    U = np.round(U_new)
                except:
                    pass
                
                # Solve for V
                A_V = np.zeros((dim_u * dim_w, target_rank))
                for a in range(dim_u):
                    for c in range(dim_w):
                        A_V[a * dim_w + c, :] = U[a, :] * W[c, :]
                
                T_all_V = np.zeros((dim_v, dim_u * dim_w))
                for (a, b, c) in indices:
                    T_all_V[b, a * dim_w + c] = 1.0
                
                try:
                    V_sol, _, _, _ = np.linalg.lstsq(A_V, T_all_V.T, rcond=None)
                    V_new = V_sol.T
                    V = np.round(V_new)
                except:
                    pass
                
                # Solve for W
                A_W = np.zeros((dim_u * dim_v, target_rank))
                for a in range(dim_u):
                    for b in range(dim_v):
                        A_W[a * dim_v + b, :] = U[a, :] * V[b, :]
                
                T_all_W = np.zeros((dim_w, dim_u * dim_v))
                for (a, b, c) in indices:
                    T_all_W[c, a * dim_v + b] = 1.0
                
                try:
                    W_sol, _, _, _ = np.linalg.lstsq(A_W, T_all_W.T, rcond=None)
                    W_new = W_sol.T
                    W = np.round(W_new)
                except:
                    pass
                
                if _verify_decomposition_fast(U, V, W, n, m, p, indices):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 120.0  # Safety margin before 150s kill
    
    # Get schoolbook as fallback
    U0, V0, W0 = _construct_schoolbook(n, m, p)
    schoolbook_result = {"U": U0.tolist(), "V": V0.tolist(), "W": V0.tolist()}
    
    # Try to find lower rank
    result = None
    
    # First try structured search for ranks 50-59
    if time.time() - start_time < timeout - 20:
        result = _structured_search(n, m, p, timeout=timeout - 20)
    
    # If not found, try greedy reduction
    if result is None and time.time() - start_time < timeout - 10:
        result = _greedy_rank_reduction(n, m, p, n * m * p, timeout=timeout - 10)
    
    if result is not None:
        return result
    
    # Return schoolbook
    return {"U": U0.tolist(), "V": V0.tolist(), "W": W0.tolist()}
