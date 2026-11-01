"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]
import time
import itertools
import random
import numpy as np

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Vectorized verification."""
    r = U.shape[1]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    T_exp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_exp[i*m+j, j*p+k, k*n+i] = 1.0
    return np.allclose(T, T_exp, atol=tol, rtol=0)

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _als_decompose(T, rank, max_iters=300, tol=1e-10, init_U=None, init_V=None, init_W=None, deadline=None):
    """
    Alternating Least Squares for tensor decomposition.
    T: target tensor (a, b, c)
    rank: target rank
    Returns U, V, W if converged, else None.
    """
    dim_a, dim_b, dim_c = T.shape
    
    if init_U is None:
        U = np.random.randn(dim_a, rank)
        V = np.random.randn(dim_b, rank)
        W = np.random.randn(dim_c, rank)
    else:
        U = init_U.copy().astype(float)
        V = init_V.copy().astype(float)
        W = init_W.copy().astype(float)
        
    # Normalize initial vectors for stability
    for r in range(rank):
        nu = np.linalg.norm(U[:, r])
        nv = np.linalg.norm(V[:, r])
        nw = np.linalg.norm(W[:, r])
        if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
            U[:, r] /= nu
            V[:, r] /= nv
            W[:, r] /= nw
    
    prev_error = float('inf')
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Update U
        for a in range(dim_a):
            M = np.einsum('br,cr->bcr', V, W)
            M_flat = M.reshape(dim_b * dim_c, rank)
            T_flat = T[a, :, :].reshape(-1)
            u_a, _, _, _ = np.linalg.lstsq(M_flat, T_flat, rcond=None)
            U[a, :] = u_a
            
        # Update V
        for b in range(dim_b):
            M = np.einsum('ar,cr->acr', U, W)
            M_flat = M.reshape(dim_a * dim_c, rank)
            T_flat = T[:, b, :].reshape(-1)
            v_b, _, _, _ = np.linalg.lstsq(M_flat, T_flat, rcond=None)
            V[b, :] = v_b
            
        # Update W
        for c in range(dim_c):
            M = np.einsum('ar,br->abr', U, V)
            M_flat = M.reshape(dim_a * dim_b, rank)
            T_flat = T[:, :, c].reshape(-1)
            w_c, _, _, _ = np.linalg.lstsq(M_flat, T_flat, rcond=None)
            W[c, :] = w_c
            
        # Calculate error
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        
        if abs(prev_error - error) < tol:
            if error < 1e-6:
                return U, V, W
            else:
                return None
        prev_error = error
        
    # Final check
    T_approx = np.einsum('ar,br,cr->abc', U, V, W)
    error = np.linalg.norm(T_approx - T)
    if error < 1e-6:
        return U, V, W
    return None

def _polish(U, V, W, T, max_iters=100, tol=1e-12, deadline=None):
    """High-precision polish pass with tighter tolerance."""
    dim_a, dim_b, dim_c = T.shape
    rank = U.shape[1]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        for a in range(dim_a):
            M = np.einsum('br,cr->bcr', V, W)
            M_flat = M.reshape(dim_b * dim_c, rank)
            T_flat = T[a, :, :].reshape(-1)
            u_a, _, _, _ = np.linalg.lstsq(M_flat, T_flat, rcond=None)
            U[a, :] = u_a
            
        for b in range(dim_b):
            M = np.einsum('ar,cr->acr', U, W)
            M_flat = M.reshape(dim_a * dim_c, rank)
            T_flat = T[:, b, :].reshape(-1)
            v_b, _, _, _ = np.linalg.lstsq(M_flat, T_flat, rcond=None)
            V[b, :] = v_b
            
        for c in range(dim_c):
            M = np.einsum('ar,br->abr', U, V)
            M_flat = M.reshape(dim_a * dim_b, rank)
            T_flat = T[:, :, c].reshape(-1)
            w_c, _, _, _ = np.linalg.lstsq(M_flat, T_flat, rcond=None)
            W[c, :] = w_c
            
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        error = np.linalg.norm(T_approx - T)
        if error < tol:
            break
    
    return U, V, W

def _structured_init_rank23(n, m, p):
    """
    Create structured initializations for rank-23 decomposition.
    
    Strategy: Start from schoolbook (rank 27), remove 4 terms based on
    algebraic structure (Strassen-like block decomposition), and let ALS
    fill in the correct values.
    
    For 3x3, we can think of it as:
    - 2x2 Strassen core: 7 multiplications
    - Cross terms with 3rd row/col: need additional terms
    - Total: 23 terms
    
    We try multiple structured removals of 4 schoolbook terms.
    """
    U0, V0, W0 = _schoolbook(n, m, p)
    total = n * m * p  # 27
    
    # Generate candidate sets of 4 indices to remove
    # Based on algebraic structure: try removing terms that correspond
    # to redundant computations in the block decomposition
    
    candidates = []
    
    # Strategy 1: Remove terms from the "diagonal" of the 3x3 grid
    # where Strassen can combine them
    diag_terms = [(i, i, i) for i in range(3)]
    off_diag = [(i, j, k) for i in range(3) for j in range(3) for k in range(3) 
                if not (i == j == k)]
    
    # Try various structured removals
    removal_sets = []
    
    # Remove 4 terms from different structural positions
    # The idea: in Strassen for 2x2, we save 1 multiplication.
    # For 3x3, we can potentially save 4 by combining cross terms.
    
    # Try all combinations of 4 from a subset of "redundant" looking terms
    # Focus on terms where the indices have some symmetry
    
    # Group terms by their "type"
    for i in range(3):
        for j in range(3):
            for k in range(3):
                idx = i * 9 + j * 3 + k
                removal_sets.append(idx)
    
    # Try specific structured removals
    # Removal set A: remove terms (0,0,0), (1,1,1), (2,2,2), and one more
    structured_sets = [
        [0, 10, 20, 9],    # diagonal + one
        [0, 10, 20, 18],   # diagonal + another
        [0, 10, 20, 27-1], # diagonal + last
        [1, 11, 21, 0],    # shifted
        [2, 12, 22, 0],    # shifted more
        [0, 10, 20, 5],
        [0, 10, 20, 15],
        [0, 10, 20, 25],
        [3, 13, 23, 0],
        [6, 16, 26, 0],
        [0, 9, 18, 27-1],
        [1, 10, 19, 27-1],
        [2, 11, 20, 27-1],
        [3, 12, 21, 27-1],
        [4, 13, 22, 27-1],
        [5, 14, 23, 27-1],
        [6, 15, 24, 27-1],
        [7, 16, 25, 27-1],
        [8, 17, 26, 27-1],
    ]
    
    # Also try random removals
    random.seed(42)
    for _ in range(30):
        removal_set = random.sample(range(total), 4)
        structured_sets.append(removal_set)
    
    return U0, V0, W0, structured_sets

def _try_rank23(n, m, p, deadline):
    """
    Attempt to find a rank-23 decomposition using structured ALS.
    """
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = _build_target_tensor(n, m, p)
    
    U0, V0, W0, removal_sets = _structured_init_rank23(n, m, p)
    total_terms = n * m * p
    
    best_result = None
    best_error = float('inf')
    
    for rem_set in removal_sets:
        if time.time() > deadline:
            break
        
        # Ensure valid indices
        rem_set = sorted(set(rem_set))
        if len(rem_set) != 4:
            continue
        
        keep_indices = [i for i in range(total_terms) if i not in rem_set]
        if len(keep_indices) != 23:
            continue
        
        init_U = U0[:, keep_indices]
        init_V = V0[:, keep_indices]
        init_W = W0[:, keep_indices]
        
        # Run ALS
        result = _als_decompose(T, 23, max_iters=200, tol=1e-10,
                                init_U=init_U, init_V=init_V, init_W=init_W,
                                deadline=deadline)
        
        if result is not None:
            U, V, W = result
            # Verify
            if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                # Polish for higher precision
                U, V, W = _polish(U, V, W, T, max_iters=50, tol=1e-12, deadline=deadline)
                if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                    return U, V, W
                else:
                    return U, V, W
    
    # If no exact solution found, try more random restarts with pure random init
    if time.time() < deadline:
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            U_init = np.random.randn(dim_a, 23)
            V_init = np.random.randn(dim_b, 23)
            W_init = np.random.randn(dim_c, 23)
            
            result = _als_decompose(T, 23, max_iters=300, tol=1e-10,
                                    init_U=U_init, init_V=V_init, init_W=W_init,
                                    deadline=deadline)
            
            if result is not None:
                U, V, W = result
                if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                    U, V, W = _polish(U, V, W, T, max_iters=50, tol=1e-12, deadline=deadline)
                    if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                        return U, V, W
                    else:
                        return U, V, W
    
    return None

def _try_rank24(n, m, p, deadline):
    """Try rank 24 as a fallback if 23 fails."""
    dim_a, dim_b, dim_c = n*m, m*p, p*n
    T = _build_target_tensor(n, m, p)
    U0, V0, W0 = _schoolbook(n, m, p)
    total_terms = n * m * p
    
    # Remove 3 terms from schoolbook
    random.seed(123)
    for _ in range(20):
        if time.time() > deadline:
            break
        
        rem_set = random.sample(range(total_terms), 3)
        keep_indices = [i for i in range(total_terms) if i not in rem_set]
        
        init_U = U0[:, keep_indices]
        init_V = V0[:, keep_indices]
        init_W = W0[:, keep_indices]
        
        result = _als_decompose(T, 24, max_iters=200, tol=1e-10,
                                init_U=init_U, init_V=init_V, init_W=init_W,
                                deadline=deadline)
        
        if result is not None:
            U, V, W = result
            if _verify_fast(U, V, W, n, m, p, tol=1e-9):
                return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try rank 23 first (known minimum)
    if time.time() < deadline:
        result = _try_rank23(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 24
    if time.time() < deadline:
        result = _try_rank24(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
