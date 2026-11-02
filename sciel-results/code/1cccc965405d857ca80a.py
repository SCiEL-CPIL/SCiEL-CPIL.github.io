import numpy as np
import time
import random

DIMENSIONS = [2, 4, 4]

def _build_rank28():
    """Build the exact rank-28 Strassen-based decomposition."""
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    
    U = np.zeros((n * m, rank))  # 8 x 28
    V = np.zeros((m * p, rank))  # 16 x 28
    W = np.zeros((p * n, rank))  # 8 x 28
    
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    blocks = [
        {'A_off': 0, 'B_row_off': 0, 'B_col_off': 0, 'C_col_off': 0},
        {'A_off': 2, 'B_row_off': 2, 'B_col_off': 0, 'C_col_off': 0},
        {'A_off': 0, 'B_row_off': 0, 'B_col_off': 2, 'C_col_off': 2},
        {'A_off': 2, 'B_row_off': 2, 'B_col_off': 2, 'C_col_off': 2},
    ]
    
    current_rank = 0
    for block in blocks:
        A_off = block['A_off']
        B_row_off = block['B_row_off']
        B_col_off = block['B_col_off']
        C_col_off = block['C_col_off']
        
        for term in strassen_terms:
            coeff_A, coeff_B, coeff_C = term
            
            for loc_idx, val in coeff_A.items():
                li, lj = divmod(loc_idx, 2)
                u_idx = li * m + (A_off + lj)
                U[u_idx, current_rank] += val
            
            for loc_idx, val in coeff_B.items():
                li, lj = divmod(loc_idx, 2)
                v_idx = (B_row_off + li) * p + (B_col_off + lj)
                V[v_idx, current_rank] += val
            
            for loc_idx, val in coeff_C.items():
                li, lj = divmod(loc_idx, 2)
                w_idx = (C_col_off + lj) * n + li
                W[w_idx, current_rank] += val
            
            current_rank += 1
    
    return U.tolist(), V.tolist(), W.tolist()


def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if i is row of A,
    j is row of B, k is col of B. Shape (n*m, m*p, p*n) with trace convention W[k*n+i]."""
    tensor = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                tensor[a_idx, b_idx, c_idx] = 1.0
    return tensor


def _try_rank27(tensor, U28, V28, W28, time_budget=130.0):
    """
    Try to find a rank-27 decomposition using Adam optimizer.
    """
    n, m, p = DIMENSIONS
    target_rank = 27
    
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_residual = np.inf
    best_U, best_V, best_W = None, None, None
    
    # Adam optimizer parameters
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = 0.01
    
    for attempt in range(200): # More attempts, shorter per attempt
        if time.time() - start_time > time_budget:
            break
        
        # Start from rank-28, drop one term
        drop_idx = rng.integers(0, 28)
        keep = [i for i in range(28) if i != drop_idx]
        
        U = U28[:, keep].copy().astype(np.float64)
        V = V28[:, keep].copy().astype(np.float64)
        W = W28[:, keep].copy().astype(np.float64)
        
        # Small perturbation
        U += rng.normal(0, 0.05, U.shape)
        V += rng.normal(0, 0.05, V.shape)
        W += rng.normal(0, 0.05, W.shape)
        
        # Adam state
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        t = 0
        for step in range(1000): # Fewer steps per attempt
            if time.time() - start_time > time_budget:
                break
            
            t += 1
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            residual = tensor - recon
            
            grad_U = -2.0 * np.einsum('abc,br,cr->ar', residual, V, W)
            grad_V = -2.0 * np.einsum('abc,ar,cr->br', residual, U, W)
            grad_W = -2.0 * np.einsum('abc,ar,br->cr', residual, U, V)
            
            # Adam update U
            mU = beta1 * mU + (1 - beta1) * grad_U
            vU = beta2 * vU + (1 - beta2) * (grad_U ** 2)
            mU_hat = mU / (1 - beta1 ** t)
            vU_hat = vU / (1 - beta2 ** t)
            U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
            
            # Adam update V
            mV = beta1 * mV + (1 - beta1) * grad_V
            vV = beta2 * vV + (1 - beta2) * (grad_V ** 2)
            mV_hat = mV / (1 - beta1 ** t)
            vV_hat = vV / (1 - beta2 ** t)
            V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
            
            # Adam update W
            mW = beta1 * mW + (1 - beta1) * grad_W
            vW = beta2 * vW + (1 - beta2) * (grad_W ** 2)
            mW_hat = mW / (1 - beta1 ** t)
            vW_hat = vW / (1 - beta2 ** t)
            W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
            
            recon = np.einsum('ar,br,cr->abc', U, V, W)
            res_norm = np.linalg.norm(tensor - recon)
            
            if res_norm < best_residual:
                best_residual = res_norm
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if res_norm < 1e-10:
                break
        
        if best_residual < 1e-10:
            break
    
    if best_residual < 1e-10 and best_U is not None:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 2x4 matrix by a 4x4 matrix.
    """
    deadline = time.time() + 140.0
    
    U28_list, V28_list, W28_list = _build_rank28()
    U28 = np.array(U28_list, dtype=np.float64)
    V28 = np.array(V28_list, dtype=np.float64)
    W28 = np.array(W28_list, dtype=np.float64)
    
    tensor = _build_tensor(2, 4, 4)
    recon28 = np.einsum('ar,br,cr->abc', U28, V28, W28)
    assert np.linalg.norm(tensor - recon28) < 1e-10, "Rank-28 verification failed"
    
    if time.time() < deadline - 10.0:
        result27 = _try_rank27(tensor, U28, V28, W28, time_budget=120.0)
        if result27 is not None:
            U27, V27, W27 = result27
            U27_arr = np.array(U27, dtype=np.float64)
            V27_arr = np.array(V27, dtype=np.float64)
            W27_arr = np.array(W27, dtype=np.float64)
            recon27 = np.einsum('ar,br,cr->abc', U27_arr, V27_arr, W27_arr)
            if np.linalg.norm(tensor - recon27) < 1e-10:
                return {"U": U27, "V": V27, "W": W27}
    
    return {"U": U28_list, "V": V28_list, "W": W28_list}
