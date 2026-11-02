"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    
    Strategy:
    1. Try structured rank-21 construction based on known algebraic reductions
    2. Use efficient ALS optimization with multiple initializations
    3. Try rank 20 and 19 with more aggressive search
    4. Fall back to schoolbook rank 24 if nothing better is found
    """
    n, m, p = 2, 3, 4
    nm, mp, pn = n * m, m * p, p * n
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    deadline = time.time() + 140  # 140s budget, 10s margin
    
    # Try ranks from 19 up to 24 (lower is better)
    for rank in [19, 20, 21]:
        if time.time() > deadline:
            break
        
        result = _try_rank(T, n, m, p, nm, mp, pn, rank, deadline)
        if result is not None:
            return result
    
    # Fallback: schoolbook rank 24
    return _schoolbook(n, m, p)


def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _try_rank(T, n, m, p, nm, mp, pn, rank, deadline):
    """
    Try to find a rank-r decomposition using multiple strategies.
    """
    # Strategy 1: Structured initialization based on schoolbook with perturbations
    result = _try_structured_init(T, n, m, p, nm, mp, pn, rank, deadline)
    if result is not None:
        return result
    
    # Strategy 2: Random initialization with ALS
    result = _try_random_als(T, n, m, p, nm, mp, pn, rank, deadline)
    if result is not None:
        return result
    
    # Strategy 3: Use scipy optimize with Nelder-Mead on smaller parameter space
    result = _try_scipy_optimize(T, n, m, p, nm, mp, pn, rank, deadline)
    if result is not None:
        return result
    
    return None


def _try_structured_init(T, n, m, p, nm, mp, pn, rank, deadline):
    """
    Try structured initialization: start with schoolbook terms, 
    then use ALS to optimize with the constraint that we only use 'rank' terms.
    """
    # Get all 24 schoolbook terms
    nz = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nz.append((i * m + j, j * p + k, k * n + i))
    
    best_result = None
    best_loss = np.inf
    
    # Try multiple structured initializations
    for trial in range(15):
        if time.time() > deadline:
            break
        
        np.random.seed(42 + trial * 13)
        
        # Initialize: use first 'rank' schoolbook terms, then add structured extras
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Fill with schoolbook terms
        for col in range(min(rank, len(nz))):
            a, b, d = nz[col]
            U[a, col] = 1.0
            V[b, col] = 1.0
            W[d, col] = 1.0
        
        # For remaining columns, use structured patterns
        for col in range(min(rank, len(nz)), rank):
            # Try different structured patterns
            pattern = (col - min(rank, len(nz))) % 4
            if pattern == 0:
                # Spread across i dimension
                a = col % nm
                b = (col * 3) % mp
                d = (col * 5) % pn
            elif pattern == 1:
                a = (col * 7) % nm
                b = col % mp
                d = (col * 11) % pn
            elif pattern == 2:
                a = (col * 13) % nm
                b = (col * 17) % mp
                d = col % pn
            else:
                a = col % nm
                b = (col * 19) % mp
                d = (col * 23) % pn
            
            U[a, col] = 1.0
            V[b, col] = 1.0
            W[d, col] = 1.0
        
        # Add small perturbations
        U += np.random.randn(nm, rank) * 0.05
        V += np.random.randn(mp, rank) * 0.05
        W += np.random.randn(pn, rank) * 0.05
        
        # Optimize using efficient ALS
        U, V, W, loss = _optimize_als_efficient(T, U, V, W, nm, mp, pn, rank, deadline)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
    
    if best_loss < 1e-8:
        # Snap to half-integers and verify
        result = _snap_and_verify(T, best_U, best_V, best_W, nm, mp, pn)
        if result is not None:
            return result
        
        # Try without snapping
        if _verify(T, best_U, best_V, best_W, nm, mp, pn):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None


def _try_random_als(T, n, m, p, nm, mp, pn, rank, deadline):
    """
    Try random initialization with ALS optimization.
    """
    best_result = None
    best_loss = np.inf
    
    num_trials = 8
    for trial in range(num_trials):
        if time.time() > deadline:
            break
        
        np.random.seed(1000 + trial * 31)
        
        # Random initialization
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        # Optimize
        U, V, W, loss = _optimize_als_efficient(T, U, V, W, nm, mp, pn, rank, deadline)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
    
    if best_loss < 1e-8:
        result = _snap_and_verify(T, best_U, best_V, best_W, nm, mp, pn)
        if result is not None:
            return result
        
        if _verify(T, best_U, best_V, best_W, nm, mp, pn):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return None


def _try_scipy_optimize(T, n, m, p, nm, mp, pn, rank, deadline):
    """
    Try scipy.optimize with Nelder-Mead on a reduced parameter space.
    We exploit the structure: for each rank-1 term, the non-zero positions 
    are constrained to specific indices.
    """
    from scipy.optimize import minimize
    
    # Precompute index arrays for the 24 schoolbook positions
    i_vals, j_vals, k_vals = np.indices((n, m, p))
    u_idx = (i_vals * m + j_vals).flatten()
    v_idx = (j_vals * p + k_vals).flatten()
    w_idx = (k_vals * n + i_vals).flatten()
    num_terms = n * m * p  # 24
    
    def error_func(params, r):
        """Compute error for given parameters."""
        U_flat = params[:r * nm]
        V_flat = params[r * nm : r * nm + r * mp]
        W_flat = params[r * nm + r * mp : r * nm + r * mp + r * pn]
        
        U = U_flat.reshape(r, nm)
        V = V_flat.reshape(r, mp)
        W = W_flat.reshape(r, pn)
        
        # Vectorized error computation
        U_sel = U[:, u_idx].T  # (num_terms, r)
        V_sel = V[:, v_idx].T  # (num_terms, r)
        W_sel = W[:, w_idx].T  # (num_terms, r)
        
        prod = U_sel * V_sel * W_sel  # (num_terms, r)
        sums = prod.sum(axis=1)  # (num_terms,)
        
        err = np.sum((sums - 1.0) ** 2)
        return float(err)
    
    best_result = None
    best_loss = np.inf
    
    # Try multiple initializations
    for trial in range(4):
        if time.time() > deadline:
            break
        
        np.random.seed(2000 + trial * 47)
        
        # Initialize with schoolbook-like structure
        total_vars = rank * (nm + mp + pn)
        x0 = np.zeros(total_vars)
        
        for c in range(min(rank, num_terms)):
            i, jk = divmod(c, m * p)
            j, k = divmod(jk, p)
            x0[c * nm + i * m + j] = 1.0
            x0[rank * nm + c * mp + j * p + k] = 1.0
            x0[rank * nm + rank * mp + c * pn + k * n + i] = 1.0
        
        # Add noise for diversity
        if trial > 0:
            noise_scale = 0.1 * trial
            x0 += noise_scale * np.random.randn(total_vars)
        
        # Optimize with Nelder-Mead
        try:
            res = minimize(error_func, x0, args=(rank,), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_loss:
                best_loss = res.fun
                best_x = res.x.copy()
            
            if res.fun < 1e-10:
                break
        except Exception:
            continue
    
    if best_loss < 1e-8:
        U_flat = best_x[:rank * nm]
        V_flat = best_x[rank * nm : rank * nm + rank * mp]
        W_flat = best_x[rank * nm + rank * mp : rank * nm + rank * mp + rank * pn]
        
        U = U_flat.reshape(rank, nm).T
        V = V_flat.reshape(rank, mp).T
        W = W_flat.reshape(rank, pn).T
        
        result = _snap_and_verify(T, U, V, W, nm, mp, pn)
        if result is not None:
            return result
        
        if _verify(T, U, V, W, nm, mp, pn):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def _optimize_als_efficient(T, U, V, W, nm, mp, pn, rank, deadline):
    """
    Efficient Alternating Least Squares optimization.
    Uses vectorized operations and early stopping.
    """
    for iteration in range(1500):
        if time.time() > deadline:
            break
        
        # Update U
        VW = np.einsum('ic,jc->ijc', V, W)  # (mp, pn, rank)
        VW_flat = VW.reshape(mp * pn, rank)  # (mp*pn, rank)
        T_mat = T.reshape(nm, mp * pn)  # (nm, mp*pn)
        try:
            U_new = np.linalg.lstsq(VW_flat.T, T_mat.T, rcond=None)[0].T
        except:
            U_new = U
        
        # Update V
        UW = np.einsum('ic,jc->ijc', U, W)  # (nm, pn, rank)
        UW_flat = UW.reshape(nm * pn, rank)  # (nm*pn, rank)
        T_mat_v = T.reshape(nm * pn, mp)  # (nm*pn, mp)
        try:
            V_new = np.linalg.lstsq(UW_flat.T, T_mat_v.T, rcond=None)[0].T
        except:
            V_new = V
        
        # Update W
        UV = np.einsum('ic,jc->ijc', U, V)  # (nm, mp, rank)
        UV_flat = UV.reshape(nm * mp, rank)  # (nm*mp, rank)
        T_mat_w = T.reshape(nm * mp, pn)  # (nm*mp, pn)
        try:
            W_new = np.linalg.lstsq(UV_flat.T, T_mat_w.T, rcond=None)[0].T
        except:
            W_new = W
        
        U, V, W = U_new, V_new, W_new
        
        # Compute loss efficiently
        tensor = np.einsum('ac,bc,dc->abd', U, V, W)
        loss = np.sum((tensor - T) ** 2)
        
        if loss < 1e-12:
            break
    
    return U, V, W, loss


def _snap_and_verify(T, U, V, W, nm, mp, pn):
    """
    Snap to half-integers and verify exact decomposition.
    """
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    if _verify(T, U_snap, V_snap, W_snap, nm, mp, pn):
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Try integer snapping
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    
    if _verify(T, U_int, V_int, W_int, nm, mp, pn):
        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
    
    return None


def _verify(T, U, V, W, nm, mp, pn):
    """Verify exact decomposition."""
    tensor = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(tensor, T, atol=1e-9)
# EVOLVE-BLOCK-END
