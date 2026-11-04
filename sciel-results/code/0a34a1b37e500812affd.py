import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety

    # First, try to find a rank-16 decomposition
    result_16 = _try_rank16(n, m, p, deadline)
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

    # Block 1: Strassen 2x2x2 (j in {0,1}, k in {0,1})
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

    # Block 2: Direct terms for k=2
    set_U17(0, 0, 7, 1); set_V17(0, 2, 7, 1); set_W17(2, 0, 7, 1)
    set_U17(0, 1, 8, 1); set_V17(1, 2, 8, 1); set_W17(2, 0, 8, 1)
    set_U17(1, 0, 9, 1); set_V17(0, 2, 9, 1); set_W17(2, 1, 9, 1)
    set_U17(1, 1, 10, 1); set_V17(1, 2, 10, 1); set_W17(2, 1, 10, 1)

    # Block 3: Direct terms for j=2
    set_U17(0, 2, 11, 1); set_V17(2, 0, 11, 1); set_W17(0, 0, 11, 1)
    set_U17(0, 2, 12, 1); set_V17(2, 1, 12, 1); set_W17(1, 0, 12, 1)
    set_U17(0, 2, 13, 1); set_V17(2, 2, 13, 1); set_W17(2, 0, 13, 1)
    set_U17(1, 2, 14, 1); set_V17(2, 0, 14, 1); set_W17(0, 1, 14, 1)
    set_U17(1, 2, 15, 1); set_V17(2, 1, 15, 1); set_W17(1, 1, 15, 1)
    set_U17(1, 2, 16, 1); set_V17(2, 2, 16, 1); set_W17(2, 1, 16, 1)

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

def _try_rank16(n, m, p, deadline):
    """
    Try to find a rank-16 decomposition.
    Uses multiple strategies:
    1. Structured initialization based on sharing Strassen terms
    2. Random restarts with gradient descent
    3. Newton polish for high precision
    """
    rank = 16
    
    def _compute_error(U, V, W, rank):
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        res = prod - 1.0
        return np.sum(res**2)
    
    def _polish(U, V, W, rank, deadline, max_iters=100):
        """High-precision Newton-style polish."""
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
            
            # Newton step
            dU = np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
            dV = np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
            dW = np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
            
            lr = 0.5
            U -= lr * dU.reshape(n * m, rank)
            V -= lr * dV.reshape(m * p, rank)
            W -= lr * dW.reshape(p * n, rank)
            
            if err < 1e-10:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return Us, Vs, Ws
        
        if err < 1e-8:
            ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
            if ok:
                return Us, Vs, Ws
        return None, None, None
    
    # Strategy 1: Structured initialization with shared Strassen terms
    def _structured_init(seed):
        np.random.seed(seed)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Use Strassen-like structure for the first 7 terms (2x2x2 block)
        # Term 0: (A00+A11)(B00+B11)
        U[0, 0] = 1; U[4, 0] = 1
        V[0, 0] = 1; V[4, 0] = 1
        W[0, 0] = 1; W[5, 0] = 1
        
        # Term 1: (A10+A11)B00
        U[3, 1] = 1; U[4, 1] = 1
        V[0, 1] = 1
        W[3, 1] = 1; W[5, 1] = -1
        
        # Term 2: A00(B01-B11)
        U[0, 2] = 1
        V[1, 2] = 1; V[4, 2] = -1
        W[3, 2] = 1; W[5, 2] = 1
        
        # Term 3: A11(B10-B00)
        U[4, 3] = 1
        V[3, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[3, 3] = 1
        
        # Term 4: (A00+A01)B11
        U[0, 4] = 1; U[1, 4] = 1
        V[4, 4] = 1
        W[0, 4] = -1; W[3, 4] = 1
        
        # Term 5: (A10-A00)(B00+B01)
        U[3, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[5, 5] = 1
        
        # Term 6: (A01-A11)(B10+B11)
        U[1, 6] = 1; U[4, 6] = -1
        V[3, 6] = 1; V[4, 6] = 1
        W[0, 6] = 1
        
        # Terms 7-15: Direct terms for k=2 and j=2
        # k=2 terms (4 terms)
        U[0, 7] = 1; V[6, 7] = 1; W[6, 7] = 1
        U[1, 8] = 1; V[7, 8] = 1; W[6, 8] = 1
        U[3, 9] = 1; V[6, 9] = 1; W[7, 9] = 1
        U[4, 10] = 1; V[7, 10] = 1; W[7, 10] = 1
        
        # j=2 terms (6 terms)
        U[2, 11] = 1; V[9, 11] = 1; W[0, 11] = 1
        U[2, 12] = 1; V[10, 12] = 1; W[1, 12] = 1
        U[2, 13] = 1; V[11, 13] = 1; W[6, 13] = 1
        U[5, 14] = 1; V[9, 14] = 1; W[3, 14] = 1
        U[5, 15] = 1; V[10, 15] = 1; W[4, 15] = 1
        # We only have 16 terms, so we need to merge or drop one
        # Let's drop the last one and see if it works, or merge
        
        # Add small random perturbation
        U += np.random.randn(n * m, rank) * 0.01
        V += np.random.randn(m * p, rank) * 0.01
        W += np.random.randn(p * n, rank) * 0.01
        
        return U, V, W
    
    # Strategy 2: Random initialization
    def _random_init(seed):
        np.random.seed(seed)
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        return U, V, W
    
    # Try structured initializations
    for seed in range(50):
        if time.time() > deadline:
            break
        
        U, V, W = _structured_init(seed)
        
        # Gradient descent
        lr = 0.03
        best_err = float('inf')
        best_state = (U.copy(), V.copy(), W.copy())
        
        for step in range(200):
            if time.time() > deadline:
                break
            
            err = _compute_error(U, V, W, rank)
            
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
            
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            res = prod - 1.0
            
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
            
            if step % 50 == 0 and step > 0:
                lr *= 0.92
            
            U -= lr * dU.reshape(n * m, rank)
            V -= lr * dV.reshape(m * p, rank)
            W -= lr * dW.reshape(p * n, rank)
            
            if step % 20 == 0:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
        
        # Polish best state
        if best_err < 1e-4:
            U_b, V_b, W_b = best_state
            Us, Vs, Ws = _polish(U_b, V_b, W_b, rank, deadline, max_iters=80)
            if Us is not None:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
    
    # Try random initializations
    for seed in range(50):
        if time.time() > deadline:
            break
        
        U, V, W = _random_init(seed)
        
        # Gradient descent
        lr = 0.02
        best_err = float('inf')
        best_state = (U.copy(), V.copy(), W.copy())
        
        for step in range(300):
            if time.time() > deadline:
                break
            
            err = _compute_error(U, V, W, rank)
            
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
            
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            res = prod - 1.0
            
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
            
            if step % 50 == 0 and step > 0:
                lr *= 0.93
            
            U -= lr * dU.reshape(n * m, rank)
            V -= lr * dV.reshape(m * p, rank)
            W -= lr * dW.reshape(p * n, rank)
            
            if step % 20 == 0:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
        
        # Polish best state
        if best_err < 1e-4:
            U_b, V_b, W_b = best_state
            Us, Vs, Ws = _polish(U_b, V_b, W_b, rank, deadline, max_iters=80)
            if Us is not None:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
    
    return None
