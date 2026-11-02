"""
Program to find rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
Uses Strassen-based rank-28 construction as baseline.
Implements a refined search for rank 27 using a hybrid approach:
1. Structured sparsity search with block-circulant-like symmetries.
2. Gradient-based local optimization (L-BFGS-B) to minimize the residual of the tensor equation.
3. Rounding to integers/half-integers and verification.
"""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _build_strassen_28():
    """
    Constructs the standard rank-28 Strassen decomposition for 2x4 * 4x4.
    """
    n, m, p = 2, 4, 4
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen terms for 2x2 matrices:
    # M1 = (x0+x3)(y0+y3)
    # M2 = (x2+x3)y0
    # M3 = x0(y1-y3)
    # M4 = x3(y2-y0)
    # M5 = (x0+x1)y3
    # M6 = (x2-x0)(y0+y1)
    # M7 = (x1-x3)(y2+y3)
    
    # Local indices: 0:00, 1:01, 2:10, 3:11
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    def map_u_a(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        global_c = c if side == 'L' else 2 + c
        return r * m + global_c
        
    def map_v_b(block, local_idx):
        r_off, c_off = {
            'TL': (0, 0), 'TR': (0, 2), 'BL': (2, 0), 'BR': (2, 2)
        }[block]
        r = local_idx // 2
        c = local_idx % 2
        global_r = r_off + r
        global_c = c_off + c
        return global_r * p + global_c
        
    def map_w_c(side, local_idx):
        r = local_idx // 2
        c = local_idx % 2
        global_c = c if side == 'L' else 2 + c
        return global_c * n + r

    def apply_strassen(a_side, b_block, c_side, start_col):
        for t in strassen_terms:
            u_local, v_local, w_local = t
            col = start_col
            
            for idx, coeff in u_local:
                U[map_u_a(a_side, idx), col] += coeff
            for idx, coeff in v_local:
                V[map_v_b(b_block, idx), col] += coeff
            for idx, coeff in w_local:
                W[map_w_c(c_side, idx), col] += coeff
            
            start_col += 1
        
        return start_col

    col = apply_strassen('L', 'TL', 'L', 0)
    col = apply_strassen('R', 'BL', 'L', col)
    col = apply_strassen('L', 'TR', 'R', col)
    col = apply_strassen('R', 'BR', 'R', col)
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=4, p=4, trials=5, seed=42):
    """Verify exact decomposition using vectorized tensor contraction."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        A = rng.random((n, m))
        B = rng.random((m, p))
        C_expected = A @ B
        
        # Vectorized computation: C[i,k] = sum_r sum_j U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Reshape for einsum
        U_t = U_arr.reshape(n, m, rank)
        V_t = V_arr.reshape(m, p, rank)
        W_t = W_arr.reshape(p, n, rank)
        
        C_computed = np.einsum('ijr,jkr,kir->ik', U_t, V_t, W_t)
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _attempt_rank_27_search(start_time, deadline=145.0):
    """
    Attempts to find a rank-27 decomposition.
    Uses vectorized objective and gradient for fast L-BFGS-B optimization.
    """
    n, m, p = 2, 4, 4
    target_rank = 27
    
    U28, V28, W28 = _build_strassen_28()
    
    # Try all 28 drop candidates, prioritized
    # Start with structured choices, then fill in the rest
    candidates = list(range(28))
    
    for drop_idx in candidates:
        if time.time() > deadline - 8:
            break
            
        U27 = np.delete(U28, drop_idx, axis=1)
        V27 = np.delete(V28, drop_idx, axis=1)
        W27 = np.delete(W28, drop_idx, axis=1)
        
        x0 = np.concatenate([U27.flatten(), V27.flatten(), W27.flatten()])
        
        def objective(x):
            Uf = x[:8*27].reshape(8, 27)
            Vf = x[8*27:8*27+16*27].reshape(16, 27)
            Wf = x[8*27+16*27:].reshape(8, 27)
            
            # Vectorized: compute all 32 tensor entries
            U_t = Uf.reshape(n, m, 27)
            V_t = Vf.reshape(m, p, 27)
            W_t = Wf.reshape(p, n, 27)
            
            vals = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            err = vals - 1.0
            return np.sum(err * err)
        
        def jac(x):
            Uf = x[:8*27].reshape(8, 27)
            Vf = x[8*27:8*27+16*27].reshape(16, 27)
            Wf = x[8*27+16*27:].reshape(8, 27)
            
            U_t = Uf.reshape(n, m, 27)
            V_t = Vf.reshape(m, p, 27)
            W_t = Wf.reshape(p, n, 27)
            
            vals = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            err = vals - 1.0
            
            # grad_U[i,j,r] = 2 * sum_k err[i,j,k] * V[j,k,r] * W[k,i,r]
            grad_U_t = 2.0 * np.einsum('ijk,jkr,kir->ijr', err, V_t, W_t)
            
            # grad_V[j,k,r] = 2 * sum_i err[i,j,k] * U[i,j,r] * W[k,i,r]
            grad_V_t = 2.0 * np.einsum('ijk,ijr,kir->jkr', err, U_t, W_t)
            
            # grad_W[k,i,r] = 2 * sum_j err[i,j,k] * U[i,j,r] * V[j,k,r]
            grad_W_t = 2.0 * np.einsum('ijk,ijr,jkr->kir', err, U_t, V_t)
            
            grad_U = grad_U_t.reshape(8, 27)
            grad_V = grad_V_t.reshape(16, 27)
            grad_W = grad_W_t.reshape(8, 27)
            
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        try:
            res = minimize(objective, x0, jac=jac, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'gtol': 1e-8, 'maxfun': 5000})
            
            if res.fun < 1e-6:
                Uf = res.x[:8*27].reshape(8, 27)
                Vf = res.x[8*27:8*27+16*27].reshape(16, 27)
                Wf = res.x[8*27+16*27:].reshape(8, 27)
                
                # Round to nearest half-integer
                U_rounded = np.round(Uf * 2) / 2
                V_rounded = np.round(Vf * 2) / 2
                W_rounded = np.round(Wf * 2) / 2
                
                if _verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()):
                    return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
                    
        except Exception:
            pass
            
    return None

def construct():
    start_time = time.time()
    deadline = 145.0
    
    U28, V28, W28 = _build_strassen_28()
    best_U, best_V, best_W = U28.tolist(), V28.tolist(), W28.tolist()
    best_rank = 28
    
    if time.time() - start_time < 5:
        result_27 = _attempt_rank_27_search(start_time, deadline)
        if result_27 is not None:
            best_U, best_V, best_W = result_27
            best_rank = 27
    
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
