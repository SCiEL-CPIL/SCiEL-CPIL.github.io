"""
Program to find rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
Uses a structural analysis to attempt rank-27 construction, falling back to rank-28 Strassen.
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
    
    # Strassen terms for 2x2 matrices
    strassen_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]), # M1
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),        # M2
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),        # M3
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),        # M4
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),        # M5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),        # M6
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),        # M7
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

def _verify_decomposition(U, V, W, n=2, m=4, p=4):
    """Verify exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Use multiple random trials to increase confidence
    for _ in range(5):
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                acc = 0.0
                for r in range(rank):
                    for j in range(m):
                        acc += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                C_computed[i, k] = acc
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _attempt_rank_27_structural(start_time):
    """
    Attempts to find a rank-27 decomposition by exploiting linear dependencies.
    Strategy: Start from Strassen-28, drop one term, and optimize the rest.
    """
    n, m, p = 2, 4, 4
    target_rank = 27
    
    U28, V28, W28 = _build_strassen_28()
    
    # We will try dropping each of the 28 terms and see if we can recover a valid solution
    # with the remaining 27 terms.
    # To save time, we will only try a few promising candidates or use a heuristic.
    # However, given the time limit, a full search over all 28 drop candidates with optimization
    # is likely too slow. 
    # Instead, we will perform a single local optimization from a perturbed state.
    
    # Let's try dropping the last term (index 27)
    U27 = U28[:, :27].copy()
    V27 = V28[:, :27].copy()
    W27 = W28[:, :27].copy()
    
    U_flat = U27.flatten()
    V_flat = V27.flatten()
    W_flat = W27.flatten()
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    def objective(x):
        if time.time() - start_time > 145:
            return float('inf')
        
        Uf = x[:8*27].reshape(8, 27)
        Vf = x[8*27:8*27+16*27].reshape(16, 27)
        Wf = x[8*27+16*27:].reshape(8, 27)
        
        residual = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(27):
                        val += Uf[i*m+j, r] * Vf[j*p+k, r] * Wf[k*n+i, r]
                    residual += (val - 1.0)**2
        return residual
    
    try:
        # Use L-BFGS-B for better convergence on smooth objectives
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 50, 'ftol': 1e-10})
        
        if res.fun < 1e-8:
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
    
    # 1. Get the guaranteed rank-28 solution
    U28, V28, W28 = _build_strassen_28()
    best_U, best_V, best_W = U28.tolist(), V28.tolist(), W28.tolist()
    best_rank = 28
    
    # 2. Attempt to find rank 27
    if time.time() - start_time < 10:
        result_27 = _attempt_rank_27_structural(start_time)
        if result_27 is not None:
            best_U, best_V, best_W = result_27
            best_rank = 27
    
    # Deadline guard
    if time.time() - start_time > 145:
        pass
        
    return {"U": best_U, "V": best_V, "W": best_W}

# EVOLVE-BLOCK-END
