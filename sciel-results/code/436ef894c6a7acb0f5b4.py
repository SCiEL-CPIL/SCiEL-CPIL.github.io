"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _make_schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _build_target(n, m, p):
    """Build the target matrix multiplication tensor."""
    target = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i * m + j, j * p + k, k * n + i] = 1.0
    return target

def _compute_tensor(U, V, W):
    """Compute the decomposed tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    rank = U.shape[1]
    T = np.zeros((U.shape[0], V.shape[0], W.shape[0]), dtype=float)
    # Use einsum in chunks
    chunk = max(1, min(rank, 200))
    for s in range(0, rank, chunk):
        e = min(s + chunk, rank)
        T += np.einsum('ar,br,cr->abc', U[:, s:e], V[:, s:e], W[:, s:e])
    return T

def _verify(U, V, W, target, tol=1e-6):
    """Verify decomposition matches target."""
    T = _compute_tensor(U, V, W)
    return np.max(np.abs(T - target)) < tol

def _optimize_rank(n, m, p, target, rank, max_time, rng):
    """Try to find a rank-R decomposition via scipy optimization."""
    nm, mp, pn = n * m, m * p, p * n
    # Flatten parameters for scipy
    init_U = rng.standard_normal((nm, rank)) * 0.5
    init_V = rng.standard_normal((mp, rank)) * 0.5
    init_W = rng.standard_normal((pn, rank)) * 0.5
    
    x0 = np.concatenate([init_U.flatten(), init_V.flatten(), init_W.flatten()])
    
    def objective(x):
        U = x[:nm*rank].reshape(nm, rank)
        V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = x[nm*rank+mp*rank:].reshape(pn, rank)
        
        T = _compute_tensor(U, V, W)
        diff = T - target
        return np.sum(diff**2)
        
    def gradient(x):
        U = x[:nm*rank].reshape(nm, rank)
        V = x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W = x[nm*rank+mp*rank:].reshape(pn, rank)
        
        T = _compute_tensor(U, V, W)
        diff = T - target
        
        dU = np.einsum('abc,br,cr->ar', diff, V, W)
        dV = np.einsum('abc,ar,cr->br', diff, U, W)
        dW = np.einsum('abc,ar,br->cr', diff, U, V)
        
        return np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])

    start_time = time.time()
    # L-BFGS-B is generally better for this
    options = {
        'maxiter': 1000,
        'ftol': 1e-12,
        'gtol': 1e-8,
        'disp': False
    }
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, options=options)
        if time.time() > start_time + max_time:
            return None, None, None, False
            
        x_opt = res.x
        U_opt = x_opt[:nm*rank].reshape(nm, rank)
        V_opt = x_opt[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
        W_opt = x_opt[nm*rank+mp*rank:].reshape(pn, rank)
        
        # Try to snap to half-integers
        U_snap = np.round(U_opt * 2) / 2
        V_snap = np.round(V_opt * 2) / 2
        W_snap = np.round(W_opt * 2) / 2
        
        if _verify(U_snap, V_snap, W_snap, target, tol=1e-5):
            return U_snap, V_snap, W_snap, True
            
        # If snap failed, maybe the solution is very close but not exactly half-integer?
        # The problem requires half-integers. If it's not close enough, it's a fail.
        
    except Exception:
        pass
        
    return None, None, None, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Build target
    target = _build_target(n, m, p)
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _make_schoolbook(n, m, p)
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    rng = np.random.default_rng(42)
    
    # Known lower bound for 4x5x6 is likely around 24-30.
    # Let's search from a reasonable lower bound up to schoolbook.
    # Schoolbook is 120.
    # We will try ranks in increasing order of difficulty (lower rank is harder).
    # But usually, we want the *minimum* rank. So we should try low ranks first.
    # However, low ranks are harder to find.
    # Let's try a few specific ranks that are likely candidates.
    # Literature suggests rank(4x5x6) might be 24 or slightly higher.
    
    ranks_to_try = list(range(20, 60)) + list(range(60, 120, 5))
    # Ensure we try the schoolbook rank last as a fallback
    if best_rank not in ranks_to_try:
        ranks_to_try.append(best_rank)
        
    for rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        # Skip if we already found a better rank (we are going low to high, so this shouldn't happen unless we jump)
        # Actually, we want the MINIMAL rank. So if we find rank R, we should stop?
        # No, we should check if there is a lower one.
        # Since we iterate low to high, the first success is the best.
        
        num_restarts = 2
        success = False
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            remaining = deadline - time.time() - 5
            if remaining < 5:
                break
                
            U, V, W, ok = _optimize_rank(n, m, p, target, rank, remaining, rng)
            if ok:
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                success = True
                break
        
        if success:
            # Found a valid decomposition at this rank.
            # Since we are iterating from low to high, this is the minimal rank found so far.
            # We can break if we are confident, or continue to see if we missed a lower one?
            # We are iterating in increasing order, so this is the lowest rank that worked.
            break
    
    # Convert to lists with half-integer rounding
    def to_list(mat):
        rounded = np.round(np.array(mat) * 2) / 2
        return rounded.tolist()
    
    U_out = to_list(best_U)
    V_out = to_list(best_V)
    W_out = to_list(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
