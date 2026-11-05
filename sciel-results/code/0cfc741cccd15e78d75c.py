import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety

    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1

    # First, try to find a rank-16 decomposition using a structured ansatz
    # based on Strassen's 3x3x3 decomposition
    result_16 = _try_rank16_structured(n, m, p, deadline)
    if result_16 is not None:
        return result_16

    # Fallback: rank-17 decomposition
    result_17 = _rank17_decomposition(n, m, p)
    return result_17

def _rank17_decomposition(n, m, p):
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)

    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)

    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)

    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)

    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)

    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)

    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)

    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)

    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)

    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)

    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)

    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)

    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)

    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)

    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)

    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)

    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)

    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }

def _check_exact(U, V, W, n, m, p):
    """Check if the decomposition is exact (half-integer entries)."""
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0

    Ui = np.rint(U_s * 2).astype(int)
    Vi = np.rint(V_s * 2).astype(int)
    Wi = np.rint(W_s * 2).astype(int)

    U_t = Ui.reshape(n, m, -1)
    V_t = Vi.reshape(m, p, -1)
    W_t = Wi.reshape(p, n, -1)

    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)

    if np.all(prod == 8):
        return True, U_s, V_s, W_s
    return False, None, None, None

def _try_rank16_structured(n, m, p, deadline):
    """
    Try to find a rank-16 decomposition using Adam optimizer and multiple initializations.
    """
    rank = 16
    
    def _polish(U, V, W, n, m, p, deadline, max_iters=50):
        """High-precision Newton-style polish with tight tolerance."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            res = prod - 1.0
            err = np.sum(res**2)
            
            if err < 1e-14:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return Us, Vs, Ws
            
            # Newton step: solve linear system for updates
            dU = np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
            dV = np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
            dW = np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
            
            # Use small step for stability
            lr = 0.5
            U -= lr * dU.reshape(n * m, rank)
            V -= lr * dV.reshape(m * p, rank)
            W -= lr * dW.reshape(p * n, rank)
            
            if err < 1e-10:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return Us, Vs, Ws
        
        # Final check
        if err < 1e-8:
            ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
            if ok:
                return Us, Vs, Ws
        return None, None, None

    def _adam_step(params, grads, m_params, v_params, t, lr, beta1, beta2, eps):
        for i in range(len(params)):
            m_params[i] = beta1 * m_params[i] + (1 - beta1) * grads[i]
            v_params[i] = beta2 * v_params[i] + (1 - beta2) * (grads[i] ** 2)
            m_hat = m_params[i] / (1 - beta1 ** t)
            v_hat = v_params[i] / (1 - beta2 ** t)
            params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def _structured_init(seed):
        np.random.seed(seed)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Structured part
        U[0*3+0, 0] = 1; U[1*3+0, 0] = 0
        V[0*3+0, 0] = 1; V[1*3+1, 0] = 1
        W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
        
        U[0*3+1, 1] = 1
        V[1*3+0, 1] = 1
        W[0*2+1, 1] = 1
        
        U[1*3+0, 2] = 1
        V[0*3+1, 2] = 1
        W[1*2+0, 2] = 1
        
        U[1*3+1, 3] = 1
        V[1*3+1, 3] = 1
        W[1*2+1, 3] = 1
        
        U[0*3+0, 4] = 1; U[1*3+1, 4] = 1
        V[0*3+0, 4] = 1; V[1*3+1, 4] = 1
        W[0*2+0, 4] = 1; W[1*2+1, 4] = 1
        
        U[0*3+0, 5] = 1; U[1*3+0, 5] = 1
        V[1*3+0, 5] = 1; V[1*3+1, 5] = 1
        W[0*2+0, 5] = -1; W[1*2+0, 5] = 1; W[1*2+1, 5] = 1
        
        U[1*3+0, 6] = 1; U[1*3+1, 6] = 1
        V[0*3+0, 6] = 1; V[0*3+1, 6] = 1
        W[0*2+1, 6] = -1; W[1*2+0, 6] = 1; W[1*2+1, 6] = 1
        
        U[0*3+2, 7] = 1
        V[0*3+2, 7] = 1
        W[2*2+0, 7] = 1
        
        U[1*3+2, 8] = 1
        V[1*3+2, 8] = 1
        W[2*2+1, 8] = 1
        
        U[0*3+2, 9] = 1; U[1*3+2, 9] = 1
        V[0*3+2, 9] = 1; V[1*3+2, 9] = 1
        W[2*2+0, 9] = 1; W[2*2+1, 9] = 1
        
        for r in range(10, rank):
            U[:, r] = np.random.randn(n * m) * 0.1
            V[:, r] = np.random.randn(m * p) * 0.1
            W[:, r] = np.random.randn(p * n) * 0.1
        
        U += np.random.randn(n * m, rank) * 0.01
        V += np.random.randn(m * p, rank) * 0.01
        W += np.random.randn(p * n, rank) * 0.01
        
        return U, V, W

    # Try multiple initializations
    num_seeds = 50
    for seed in range(num_seeds):
        if time.time() > deadline:
            break
        
        U, V, W = _structured_init(seed)
        
        # Adam optimizer
        lr = 0.01
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = 0
        
        m_U = np.zeros_like(U)
        v_U = np.zeros_like(U)
        m_V = np.zeros_like(V)
        v_V = np.zeros_like(V)
        m_W = np.zeros_like(W)
        v_W = np.zeros_like(W)
        
        best_err = float('inf')
        best_state = (U.copy(), V.copy(), W.copy())
        
        for step in range(200):
            if time.time() > deadline:
                break
            
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            res = prod - 1.0
            err = np.sum(res**2)
            
            if err < best_err:
                best_err = err
                best_state = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
            
            # Compute gradients
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t).reshape(n * m, rank)
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t).reshape(m * p, rank)
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t).reshape(p * n, rank)
            
            t += 1
            _adam_step([U], [dU], [m_U], [v_U], t, lr, beta1, beta2, eps)
            _adam_step([V], [dV], [m_V], [v_V], t, lr, beta1, beta2, eps)
            _adam_step([W], [dW], [m_W], [v_W], t, lr, beta1, beta2, eps)
            
            if step % 10 == 0:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
        
        if best_err < 1e-4:
            U_b, V_b, W_b = best_state
            Us, Vs, Ws = _polish(U_b, V_b, W_b, n, m, p, deadline, max_iters=50)
            if Us is not None:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
    
    return None
