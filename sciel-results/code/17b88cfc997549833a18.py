import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb

    # Target ranks to try
    # Theoretical lower bound is 9, but likely higher.
    # Strassen 2x2x2 is 7.
    # 3x3x3 is 23.
    # 3x3x5 might be around 30-40.
    # Let's try a range.
    target_ranks = list(range(30, 45))
    
    # Prioritize lower ranks
    for target_rank in target_ranks:
        if target_rank >= best_rank:
            continue
        if time.time() > deadline - 10:
            break

        # Number of restarts decreases as rank increases (easier to find)
        max_restarts = 20 if target_rank <= 35 else 10
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            try:
                result = search_rank_als(n, m, p, target_rank, deadline)
                if result is not None:
                    return result
            except Exception:
                pass

    return best_result

def search_rank_als(n, m, p, rank, deadline):
    """
    Uses Alternating Least Squares and Gradient Descent hybrid with rational reconstruction.
    """
    # Initialization strategies
    inits = [
        'random_small',
        'strassen_like',
        'sparse_random'
    ]
    
    for init_strategy in inits:
        if time.time() > deadline - 5:
            break
            
        if init_strategy == 'random_small':
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
        elif init_strategy == 'strassen_like':
            # Try to mimic Strassen structure
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            # Fill with random sparse structure
            for r in range(rank):
                # Pick random indices
                u_idx = np.random.randint(0, n * m, size=2)
                v_idx = np.random.randint(0, m * p, size=2)
                w_idx = np.random.randint(0, p * n, size=2)
                U[u_idx, r] = np.random.choice([-1, 1])
                V[v_idx, r] = np.random.choice([-1, 1])
                W[w_idx, r] = np.random.choice([-1, 1])
        else:
            U = np.random.randn(n * m, rank) * 0.1
            V = np.random.randn(m * p, rank) * 0.1
            W = np.random.randn(p * n, rank) * 0.1

        # Phase 1: Gradient Descent to get close
        U, V, W = _gradient_descent_adam(U, V, W, n, m, p, rank, deadline)
        
        # Phase 2: Snap to half-integers and verify
        result = _try_snap_and_correct(U, V, W, n, m, p)
        if result is not None:
            return result
            
        # Phase 3: If close but not exact, try local correction
        if _is_close(U, V, W, n, m, p, 1e-4):
             result = _local_correction(U, V, W, n, m, p, deadline)
             if result is not None:
                 return result

    return None

def _gradient_descent_adam(U, V, W, n, m, p, rank, deadline):
    """
    Adam optimizer for tensor decomposition.
    """
    U3 = U.copy().reshape(n, m, rank)
    V3 = V.copy().reshape(m, p, rank)
    W3 = W.copy().reshape(p, n, rank)
    
    # Adam parameters
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    lr = 0.01
    
    mU = np.zeros_like(U3)
    vU = np.zeros_like(U3)
    mV = np.zeros_like(V3)
    vV = np.zeros_like(V3)
    mW = np.zeros_like(W3)
    vW = np.zeros_like(W3)
    
    t = 0
    max_iters = 1000
    best_error = np.inf
    
    for iter in range(max_iters):
        if time.time() > deadline - 2:
            break
            
        t += 1
        # Compute T
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        error = np.sum((T - 1.0)**2)
        
        if error < 1e-10:
            break
            
        if error < best_error:
            best_error = error
            best_U3, best_V3, best_W3 = U3.copy(), V3.copy(), W3.copy()
        
        # Compute gradients
        dU3 = 2.0 * np.einsum('ijk,jkr,kir->ijr', T - 1.0, V3, W3)
        dV3 = 2.0 * np.einsum('ijk,ijr,kir->jkr', T - 1.0, U3, W3)
        dW3 = 2.0 * np.einsum('ijk,ijr,jkr->kir', T - 1.0, U3, V3)
        
        # Adam update
        mU = beta1 * mU + (1 - beta1) * dU3
        vU = beta2 * vU + (1 - beta2) * (dU3 ** 2)
        mU_hat = mU / (1 - beta1 ** t)
        vU_hat = vU / (1 - beta2 ** t)
        U3 -= lr * mU_hat / (np.sqrt(vU_hat) + epsilon)
        
        mV = beta1 * mV + (1 - beta1) * dV3
        vV = beta2 * vV + (1 - beta2) * (dV3 ** 2)
        mV_hat = mV / (1 - beta1 ** t)
        vV_hat = vV / (1 - beta2 ** t)
        V3 -= lr * mV_hat / (np.sqrt(vV_hat) + epsilon)
        
        mW = beta1 * mW + (1 - beta1) * dW3
        vW = beta2 * vW + (1 - beta2) * (dW3 ** 2)
        mW_hat = mW / (1 - beta1 ** t)
        vW_hat = vW / (1 - beta2 ** t)
        W3 -= lr * mW_hat / (np.sqrt(vW_hat) + epsilon)
        
        # Decay learning rate
        if iter % 100 == 0 and iter > 0:
            lr *= 0.95

    return best_U3.reshape(-1), best_V3.reshape(-1), best_W3.reshape(-1)

def _try_snap_and_correct(U, V, W, n, m, p):
    """
    Try to snap to half-integers and verify.
    """
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    error = compute_error(U_snap, V_snap, W_snap, n, m, p)
    if error < 1e-8:
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
    
    # Try integer snap
    U_snap_int = np.round(U)
    V_snap_int = np.round(V)
    W_snap_int = np.round(W)
    error_int = compute_error(U_snap_int, V_snap_int, W_snap_int, n, m, p)
    if error_int < 1e-8:
        return {"U": U_snap_int.tolist(), "V": V_snap_int.tolist(), "W": W_snap_int.tolist()}
    
    return None

def _is_close(U, V, W, n, m, p, threshold):
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    error = compute_error(U_snap, V_snap, W_snap, n, m, p)
    return error < threshold

def _local_correction(U, V, W, n, m, p, deadline):
    """
    If the solution is close to half-integers, try to correct it by solving a linear system.
    This is a heuristic: fix U and V, solve for W, then check if W is close to half-integers.
    """
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    rank = U3.shape[2]
    
    # Construct linear system for W
    # T_ijk = sum_r U_ijr V_jkr W_kir
    # We want T_ijk = 1
    # This is a linear system in W_kir.
    
    # Matrix A of size (n*m*p) x (p*n*rank)
    # Indexing:
    # Row index: i, j, k -> i * m * p + j * p + k
    # Col index: k, i, r -> k * n * rank + i * rank + r
    
    num_eqs = n * m * p
    num_unk = p * n * rank
    
    A = np.zeros((num_eqs, num_unk))
    b = np.ones(num_eqs)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row = i * m * p + j * p + k
                for r in range(rank):
                    col = k * n * rank + i * rank + r
                    A[row, col] = U3[i, j, r] * V3[j, k, r]
    
    # Solve least squares
    try:
        W_sol, residuals, rank_sol, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
        
    W_sol = W_sol.reshape(p, n, rank)
    W_flat = W_sol.reshape(-1)
    
    # Snap W
    W_snap = np.round(W_flat * 2) / 2
    W_snap_int = np.round(W_flat)
    
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    
    # Check if snapped W works with snapped U, V
    error1 = compute_error(U_snap, V_snap, W_snap, n, m, p)
    if error1 < 1e-8:
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
        
    error2 = compute_error(U_snap, V_snap, W_snap_int, n, m, p)
    if error2 < 1e-8:
        return {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap_int.tolist()}
        
    # Try snapping U and V to integers if they are close
    U_snap_int = np.round(U)
    V_snap_int = np.round(V)
    
    error3 = compute_error(U_snap_int, V_snap_int, W_snap, n, m, p)
    if error3 < 1e-8:
        return {"U": U_snap_int.tolist(), "V": V_snap_int.tolist(), "W": W_snap.tolist()}
        
    return None

def compute_error(U, V, W, n, m, p):
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    error = np.sum((T - 1.0)**2)
    return error
