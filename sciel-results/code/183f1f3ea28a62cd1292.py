import numpy as np
import time
import sys

DIMENSIONS = [2, 2, 15]

def _build_target_tensor():
    """
    Construct the matrix multiplication tensor M_{2,2,15}.
    Indices:
    U: n*m = 4
    V: m*p = 30
    W: p*n = 30
    
    Convention:
    A is 2x2. Index (i, j) -> flat index i*m + j.
    B is 2x15. Index (j, k) -> flat index j*p + k.
    C is 2x15. Index (i, k) -> flat index k*n + i (as per task description W[k*n+i, r]).
    
    Tensor T[i*m+j, j*p+k, k*n+i] = 1 if valid, else 0?
    Actually, the tensor entries are 1 if the indices are consistent with the multiplication.
    T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k.
    Note that j must match in a and b. i must match in a and c. k must match in b and c.
    """
    n, m, p = DIMENSIONS
    size_u = n * m
    size_v = m * p
    size_w = p * n
    T = np.zeros((size_u, size_v, size_w), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    return T

def _verify_decomposition(U, V, W, T, tol=1e-6):
    """
    Verify if U, V, W decompose T exactly.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    if not (U.shape[1] == V.shape[1] == W.shape[1]):
        return False
    rank = U.shape[1]
    
    # Efficiently compute the tensor reconstruction
    # T_recon = sum_r U[:, r] outer V[:, r] outer W[:, r]
    # We can do this via einsum or loops for small ranks.
    # Given rank ~60 and sizes ~30, 30x30x30=27000 elements.
    # 60 iterations of outer products is fast.
    
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
    diff = np.max(np.abs(T_recon - T))
    return diff < tol

def _get_seed_decomposition():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
    return U, V, W

def _attempt_rank_reduction(T, target_rank, time_limit=100):
    """
    Attempt to find a decomposition of lower rank using random initialization and gradient descent.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = DIMENSIONS
    size_u, size_v, size_w = T.shape
    T_flat = T.flatten()
    
    start_time = time.time()
    
    # Initialize random U, V, W for target_rank
    # We will optimize the factors to match T.
    # Loss: || T - sum(U_r, V_r, W_r) ||^2
    
    # To make it faster, we can fix one factor to 1s? No, that restricts the space.
    # We can use a parameterization where we optimize U, V, W directly.
    
    # Random init
    U = np.random.randn(size_u, target_rank)
    V = np.random.randn(size_v, target_rank)
    W = np.random.randn(size_w, target_rank)
    
    # Normalize to avoid scaling issues
    for r in range(target_rank):
        norm = np.linalg.norm(U[:, r])
        if norm > 0: U[:, r] /= norm
        norm = np.linalg.norm(V[:, r])
        if norm > 0: V[:, r] /= norm
        norm = np.linalg.norm(W[:, r])
        if norm > 0: W[:, r] /= norm

    def loss_and_grad(params):
        # Unpack
        U = params[0:size_u * target_rank].reshape(size_u, target_rank)
        V = params[size_u * target_rank : size_u * target_rank + size_v * target_rank].reshape(size_v, target_rank)
        W = params[size_u * target_rank + size_v * target_rank : :].reshape(size_w, target_rank)
        
        # Reconstruct
        T_recon = np.zeros_like(T)
        for r in range(target_rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
        diff = T_recon - T
        loss = np.sum(diff**2)
        
        # Gradient calculation is complex for arbitrary tensors.
        # We can use a simple finite difference or just return loss and let scipy handle it?
        # scipy.optimize.minimize can do numerical gradients, but it's slow.
        # Given the time constraint and the likely failure, we might just do a few random restarts with local search?
        # Actually, let's just return the loss. We'll use a simpler heuristic:
        # If we can't easily get gradients, we might skip this complex optimization in favor of a simpler check?
        # But the directive wants a structural change.
        # Let's use `scipy.optimize.minimize` with 'L-BFGS-B' if we can provide gradients.
        # Providing analytical gradients for tensor decomposition:
        # dL/dU[i, r] = 2 * sum_{j,k} diff[i, j, k] * V[j, r] * W[k, r]
        # dL/dV[j, r] = 2 * sum_{i,k} diff[i, j, k] * U[i, r] * W[k, r]
        # dL/dW[k, r] = 2 * sum_{i,j} diff[i, j, k] * U[i, r] * V[j, r]
        
        # This is computable.
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # Use einsum for gradients
        # diff is shape (I, J, K)
        # grad_U[i, r] = sum_{j,k} diff[i, j, k] V[j, r] W[k, r]
        # Can be computed as:
        # tmp1 = np.einsum('ijk, jr, kr -> ir', diff, V, W)
        # tmp2 = np.einsum('ijk, ir, kr -> jr', diff, U, W)
        # tmp3 = np.einsum('ijk, ir, jr -> kr', diff, U, V)
        
        grad_U = np.einsum('ijk, jr, kr -> ir', diff, V, W)
        grad_V = np.einsum('ijk, ir, kr -> jr', diff, U, W)
        grad_W = np.einsum('ijk, ir, jr -> kr', diff, U, V)
        
        grad_flat = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad_flat

    # Use scipy
    from scipy.optimize import minimize
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # We do a few restarts if time permits
    best_loss = np.inf
    best_x = None
    
    num_restarts = 5
    time_per_restart = time_limit / num_restarts
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Re-initialize for restart
        U = np.random.randn(size_u, target_rank)
        V = np.random.randn(size_v, target_rank)
        W = np.random.randn(size_w, target_rank)
        # Scale randomly to avoid symmetry issues
        scales = np.random.rand(target_rank)
        for r in range(target_rank):
            U[:, r] *= scales[r]
            V[:, r] *= scales[r]
            W[:, r] /= (scales[r]*scales[r] + 1e-9) # Balance
            
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(loss_and_grad, x0, method='L-BFGS-B', jac=True, 
                           options={'maxiter': 200, 'maxfun': 1000, 'maxls': 100})
            if res.success and res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x
        except Exception:
            pass
            
    if best_loss is None or best_loss > 1e-4:
        return None
        
    # Parse best_x
    U = best_x[0:size_u * target_rank].reshape(size_u, target_rank)
    V = best_x[size_u * target_rank : size_u * target_rank + size_v * target_rank].reshape(size_v, target_rank)
    W = best_x[size_u * target_rank + size_v * target_rank : :].reshape(size_w, target_rank)
    
    # Verify
    if _verify_decomposition(U, V, W, T, tol=1e-4):
        # Convert to list of lists of floats
        return [U.tolist(), V.tolist(), W.tolist()]
    else:
        return None

def construct():
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # 1. Get seed decomposition (Rank 60)
    U_seed, V_seed, W_seed = _get_seed_decomposition()
    seed_rank = U_seed.shape[1]
    
    # 2. Attempt to find a lower rank decomposition
    # We try rank seed_rank - 1
    target_rank = seed_rank - 1
    
    # Time management
    start_time = time.time()
    time_limit_search = 110 # Leave 10s for overhead and return
    
    lower_rank_sol = None
    try:
        lower_rank_sol = _attempt_rank_reduction(T, target_rank, time_limit=time_limit_search)
    except Exception as e:
        print(f"Search failed: {e}", file=sys.stderr)
        
    if lower_rank_sol is not None:
        U, V, W = lower_rank_sol
        # Final verification before returning
        if _verify_decomposition(np.array(U), np.array(V), np.array(W), T, tol=1e-4):
            return {"U": U, "V": V, "W": W}
            
    # Fallback to seed
    return {
        "U": U_seed.tolist(),
        "V": V_seed.tolist(),
        "W": W_seed.tolist()
    }
