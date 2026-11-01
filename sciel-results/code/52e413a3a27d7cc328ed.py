"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0
    
    # Schoolbook rank as fallback
    schoolbook_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    deadline = time.time() + 130  # 130s budget, leave 20s margin for safety
    
    # Try ranks from schoolbook-1 down to 20
    for rank in range(schoolbook_rank - 1, 19, -1):
        if time.time() > deadline:
            break
        result = _try_rank_structured(T, nm, mp, pn, rank, deadline)
        if result is not None:
            best_result = result
            break  # Found lower rank, stop
            
    return best_result

def _schoolbook(n, m, p):
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_rank_structured(T, nm, mp, pn, rank, deadline):
    """Try to find a rank-`rank` decomposition using structured approaches."""
    results = []
    
    # Approach 1: Random search with diverse seeds
    for seed in range(50):
        if time.time() > deadline:
            break
        result = _random_search(T, nm, mp, pn, rank, seed, deadline)
        if result is not None:
            return result
    
    # Approach 2: Sparse structured search
    result = _sparse_search(T, nm, mp, pn, rank, deadline)
    if result is not None:
        return result
    
    # Approach 3: Try to find near-integer solutions via optimization
    result = _optimize_rank(T, nm, mp, pn, rank, deadline)
    if result is not None:
        return result
    
    return None

def _random_search(T, nm, mp, pn, rank, seed, deadline):
    """Random search for rank decomposition."""
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(nm, rank) * 0.5
    V = rng.randn(mp, rank) * 0.5
    W = rng.randn(pn, rank) * 0.5
    
    # Use L-BFGS-like approach with multiple restarts
    lr = 0.1
    best_residual = np.inf
    
    for it in range(200):
        if time.time() > deadline:
            return None
        
        tensor = np.einsum('ac,bc,dc->abd', U, V, W)
        residual = tensor - T
        res_norm = np.sum(residual**2)
        
        if res_norm < best_residual:
            best_residual = res_norm
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if res_norm < 1e-12:
            break
        
        # Compute gradients
        dU = np.einsum('abd,bc,dc->ac', residual, V, W)
        dV = np.einsum('abd,ac,dc->bc', residual, U, W)
        dW = np.einsum('abd,ac,bc->dc', residual, U, V)
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm > 0:
            step = lr / max(grad_norm, 1e-8)
            U -= step * dU
            V -= step * dV
            W -= step * dW
        
        # Decay learning rate
        lr *= 0.995
    
    # Check if we found a solution
    tensor = np.einsum('ac,bc,dc->abd', best_U, best_V, best_W)
    if np.sum((tensor - T)**2) < 1e-10:
        # Try to snap to half-integers
        for scale in [2, 4, 8, 16]:
            U_snap = np.round(best_U * scale) / scale
            V_snap = np.round(best_V * scale) / scale
            W_snap = np.round(best_W * scale) / scale
            if _verify(T, U_snap, V_snap, W_snap):
                return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
        # Return as-is if close enough
        if _verify(T, best_U, best_V, best_W):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None

def _sparse_search(T, nm, mp, pn, rank, deadline):
    """Search for sparse structured decompositions."""
    # For small ranks, try binary/sparse patterns
    if rank > 20:
        return None
    
    # Try all combinations of signs for a base pattern
    base_patterns = []
    
    # Generate some structured patterns
    for i in range(nm):
        for j in range(mp):
            for k in range(pn):
                if T[i, j, k] != 0:
                    base_patterns.append((i, j, k))
    
    # Try to find a subset of rank terms that covers all non-zero entries
    # This is a set cover-like problem, use greedy approach
    if not base_patterns:
        return None
    
    # Greedy: pick terms that cover most uncovered entries
    covered = set()
    selected = []
    
    all_terms = []
    for idx in range(rank):
        # Generate random sparse term
        U_col = np.zeros(nm)
        V_col = np.zeros(mp)
        W_col = np.zeros(pn)
        
        # Randomly assign ±1 to some positions
        for _ in range(3):
            pos = np.random.randint(nm)
            U_col[pos] = np.random.choice([-1, 1])
        for _ in range(3):
            pos = np.random.randint(mp)
            V_col[pos] = np.random.choice([-1, 1])
        for _ in range(3):
            pos = np.random.randint(pn)
            W_col[pos] = np.random.choice([-1, 1])
        
        all_terms.append((U_col, V_col, W_col))
    
    return None

def _optimize_rank(T, nm, mp, pn, rank, deadline):
    """Optimize for exact solution using more sophisticated methods."""
    # Use scipy's optimization if available
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    
    # Flatten parameters
    def _objective(params):
        U = params[:nm*rank].reshape(nm, rank)
        V = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = params[nm*rank+mp*rank:].reshape(pn, rank)
        tensor = np.einsum('ac,bc,dc->abd', U, V, W)
        residual = tensor - T
        return np.sum(residual**2)
    
    # Try a few random initializations
    for trial in range(5):
        if time.time() > deadline:
            return None
        
        x0 = np.random.randn(nm*rank + mp*rank + pn*rank) * 0.1
        try:
            res = minimize(_objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-15})
            if res.fun < 1e-10:
                U = res.x[:nm*rank].reshape(nm, rank)
                V = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                W = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                
                # Try to snap to half-integers
                for scale in [2, 4, 8, 16]:
                    U_snap = np.round(U * scale) / scale
                    V_snap = np.round(V * scale) / scale
                    W_snap = np.round(W * scale) / scale
                    if _verify(T, U_snap, V_snap, W_snap):
                        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                
                if _verify(T, U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            continue
    
    return None

def _verify(T, U, V, W):
    """Verify that sum_c U[:,c]*V[:,c]*W[:,c] == T exactly (within tolerance)."""
    tensor = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(tensor, T, atol=1e-10)
# EVOLVE-BLOCK-END
