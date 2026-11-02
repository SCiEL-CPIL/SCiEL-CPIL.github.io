import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 5, 6]

def _schoolbook(n, m, p):
    """Return schoolbook decomposition with rank n*m*p."""
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_target_tensor(n, m, p):
    """Build the dense target tensor for matrix multiplication."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    return T

def _verify_decomposition(U, V, W, T_target, tol=1e-8):
    """Verify that U, V, W exactly decompose the target tensor."""
    T_calc = np.einsum('ar,br,dr->abd', U, V, W)
    return np.allclose(T_calc, T_target, atol=tol)

def _optimize_rank(n, m, p, rank, T_target, deadline, n_restarts=5):
    """
    Attempt to find a decomposition of the given rank using gradient descent.
    Returns (U, V, W) if successful (low error), else None.
    """
    nm, mp, pn = n * m, m * p, p * n
    n_params = rank * (nm + mp + pn)
    
    def loss(params):
        if time.time() > deadline:
            return float('inf')
        
        # Reshape parameters into U, V, W
        U = params[:rank*nm].reshape(nm, rank)
        V = params[rank*nm:rank*(nm+mp)].reshape(mp, rank)
        W = params[rank*(nm+mp):].reshape(pn, rank)
        
        # Compute tensor contraction
        T_calc = np.einsum('ar,br,dr->abd', U, V, W)
        
        # Compute squared error
        diff = T_calc - T_target
        return np.sum(diff * diff)
    
    def grad(params):
        if time.time() > deadline:
            return np.zeros(n_params)
        
        U = params[:rank*nm].reshape(nm, rank)
        V = params[rank*nm:rank*(nm+mp)].reshape(mp, rank)
        W = params[rank*(nm+mp):].reshape(pn, rank)
        
        T_calc = np.einsum('ar,br,dr->abd', U, V, W)
        diff = T_calc - T_target
        
        # Gradients
        dU = np.einsum('abd,br,dr->ar', diff, V, W)
        dV = np.einsum('abd,ar,dr->br', diff, U, W)
        dW = np.einsum('abd,ar,br->dr', diff, U, V)
        
        # Flatten gradients
        grad_flat = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return 2.0 * grad_flat
    
    # Try several random initializations with different strategies
    for trial in range(n_restarts):
        if time.time() > deadline - 2.0:
            return None
        
        # Different initialization strategies
        if trial == 0:
            # Small random normal
            x0 = np.random.randn(n_params) * 0.1
        elif trial == 1:
            # Sparse random initialization
            x0 = np.random.randn(n_params) * 0.05
            # Make it sparser
            mask = np.random.random(n_params) < 0.3
            x0[~mask] = 0.0
        elif trial == 2:
            # Structured initialization based on schoolbook pattern
            x0 = np.random.randn(n_params) * 0.2
        else:
            # Very small random
            x0 = np.random.randn(n_params) * 0.01
        
        # Optimize
        try:
            result = opt.minimize(loss, x0, jac=grad, method='L-BFGS-B', 
                                  options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30})
            
            if result.fun < 1e-10:
                # Extract U, V, W
                U = result.x[:rank*nm].reshape(nm, rank)
                V = result.x[rank*nm:rank*(nm+mp)].reshape(mp, rank)
                W = result.x[rank*(nm+mp):].reshape(pn, rank)
                
                # Final verification
                if _verify_decomposition(U, V, W, T_target, tol=1e-8):
                    return U, V, W
                    
        except Exception:
            pass
            
    return None

def _try_structured_decomposition(n, m, p, rank, deadline):
    """
    Try a structured decomposition approach exploiting the small dimension n=3.
    This uses a block structure where we split the computation.
    """
    nm, mp, pn = n * m, m * p, p * n
    T_target = _build_target_tensor(n, m, p)
    
    # Strategy: For each i (row of A, 0..n-1), the output row depends on
    # all rows of B. We can try to factor this in a structured way.
    
    # Try a rank-r decomposition where we use a structured pattern
    # based on the fact that n=3 is small
    
    # Approach: Use a "partial" Strassen-like decomposition
    # For 3x5x6, we can try to exploit the structure
    
    # Generate random factors with a structured sparsity pattern
    for trial in range(20):
        if time.time() > deadline - 5.0:
            return None
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # For each rank dimension, assign a structured pattern
        for c in range(rank):
            # Randomly choose which (i,j) pairs are active in U
            # and which (j,k) pairs are active in V
            # and which (k,i) pairs are active in W
            
            # U: pick 1-3 (i,j) pairs
            n_u = np.random.randint(1, min(4, nm) + 1)
            u_indices = np.random.choice(nm, n_u, replace=False)
            for idx in u_indices:
                U[idx, c] = np.random.choice([-1, 1, 0.5, -0.5, 2, -2])
            
            # V: pick 1-3 (j,k) pairs
            n_v = np.random.randint(1, min(4, mp) + 1)
            v_indices = np.random.choice(mp, n_v, replace=False)
            for idx in v_indices:
                V[idx, c] = np.random.choice([-1, 1, 0.5, -0.5, 2, -2])
            
            # W: pick 1-3 (k,i) pairs
            n_w = np.random.randint(1, min(4, pn) + 1)
            w_indices = np.random.choice(pn, n_w, replace=False)
            for idx in w_indices:
                W[idx, c] = np.random.choice([-1, 1, 0.5, -0.5, 2, -2])
        
        # Check if this works
        if _verify_decomposition(U, V, W, T_target, tol=1e-8):
            return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Get schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    # Build target tensor
    T_target = _build_target_tensor(n, m, p)
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = schoolbook_rank
    
    # Search for lower ranks
    # Start from a reasonable lower bound and go up, or from schoolbook-1 and go down
    # For 3x5x6, schoolbook is 90
    # Try to find something below 90
    
    # First, try structured approach for various ranks
    # Then try gradient-based optimization
    
    # Try ranks from 40 to 89 (below schoolbook)
    # Prioritize ranks that are likely to work based on known results
    
    # Known: 3x3x3 has rank 23, so 3x5x6 should be significantly less than 90
    # Let's try a range of ranks
    
    # Strategy: Try gradient-based optimization for several ranks
    # Start from lower ranks and work up
    
    ranks_to_try = list(range(45, 90))
    # Shuffle to explore different ranks
    np.random.shuffle(ranks_to_try)
    # But keep lower ranks slightly prioritized
    ranks_to_try.sort()
    
    for rank in ranks_to_try:
        if time.time() > deadline - 15.0:
            break
        
        if rank >= best_rank:
            continue
        
        # Try structured decomposition first (faster)
        result_struct = _try_structured_decomposition(n, m, p, rank, deadline)
        if result_struct is not None:
            U, V, W = result_struct
            best_U, best_V, best_W = U, V, W
            best_rank = rank
            # Continue searching for even lower ranks
            continue
        
        # Try gradient-based optimization
        result_opt = _optimize_rank(n, m, p, rank, T_target, deadline, n_restarts=3)
        if result_opt is not None:
            U, V, W = result_opt
            best_U, best_V, best_W = U, V, W
            best_rank = rank
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
