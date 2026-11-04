import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety

    # Try to find a rank-16 decomposition
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

    # Row 0 of the 2x3 matrix
    # Term 0: U[0,0]=1, V[0,0]=1, W[0,0]=1
    set_U17(0, 0, 0, 1); set_V17(0, 0, 0, 1); set_W17(0, 0, 0, 1)
    # Term 1: U[0,1]=1, V[0,1]=1, W[1,0]=1
    set_U17(0, 1, 1, 1); set_V17(0, 1, 1, 1); set_W17(1, 0, 1, 1)
    # Term 2: U[0,2]=1, V[0,2]=1, W[2,0]=1
    set_U17(0, 2, 2, 1); set_V17(0, 2, 2, 1); set_W17(2, 0, 2, 1)
    
    # Term 3: U[0,0]=1, V[1,0]=1, W[0,0]=1
    set_U17(0, 0, 3, 1); set_V17(1, 0, 3, 1); set_W17(0, 0, 3, 1)
    # Term 4: U[0,1]=1, V[1,1]=1, W[1,0]=1
    set_U17(0, 1, 4, 1); set_V17(1, 1, 4, 1); set_W17(1, 0, 4, 1)
    # Term 5: U[0,2]=1, V[1,2]=1, W[2,0]=1
    set_U17(0, 2, 5, 1); set_V17(1, 2, 5, 1); set_W17(2, 0, 5, 1)
    
    # Term 6: U[0,0]=1, V[2,0]=1, W[0,0]=1
    set_U17(0, 0, 6, 1); set_V17(2, 0, 6, 1); set_W17(0, 0, 6, 1)
    # Term 7: U[0,1]=1, V[2,1]=1, W[1,0]=1
    set_U17(0, 1, 7, 1); set_V17(2, 1, 7, 1); set_W17(1, 0, 7, 1)
    # Term 8: U[0,2]=1, V[2,2]=1, W[2,0]=1
    set_U17(0, 2, 8, 1); set_V17(2, 2, 8, 1); set_W17(2, 0, 8, 1)

    # Row 1 of the 2x3 matrix
    # Term 9: U[1,0]=1, V[0,0]=1, W[0,1]=1
    set_U17(1, 0, 9, 1); set_V17(0, 0, 9, 1); set_W17(0, 1, 9, 1)
    # Term 10: U[1,1]=1, V[0,1]=1, W[1,1]=1
    set_U17(1, 1, 10, 1); set_V17(0, 1, 10, 1); set_W17(1, 1, 10, 1)
    # Term 11: U[1,2]=1, V[0,2]=1, W[2,1]=1
    set_U17(1, 2, 11, 1); set_V17(0, 2, 11, 1); set_W17(2, 1, 11, 1)
    
    # Term 12: U[1,0]=1, V[1,0]=1, W[0,1]=1
    set_U17(1, 0, 12, 1); set_V17(1, 0, 12, 1); set_W17(0, 1, 12, 1)
    # Term 13: U[1,1]=1, V[1,1]=1, W[1,1]=1
    set_U17(1, 1, 13, 1); set_V17(1, 1, 13, 1); set_W17(1, 1, 13, 1)
    # Term 14: U[1,2]=1, V[1,2]=1, W[2,1]=1
    set_U17(1, 2, 14, 1); set_V17(1, 2, 14, 1); set_W17(2, 1, 14, 1)
    
    # Term 15: U[1,0]=1, V[2,0]=1, W[0,1]=1
    set_U17(1, 0, 15, 1); set_V17(2, 0, 15, 1); set_W17(0, 1, 15, 1)
    # Term 16: U[1,1]=1, V[2,1]=1, W[1,1]=1
    set_U17(1, 1, 16, 1); set_V17(2, 1, 16, 1); set_W17(1, 1, 16, 1)
    # Term 17: U[1,2]=1, V[2,2]=1, W[2,1]=1
    set_U17(1, 2, 17, 1); set_V17(2, 2, 17, 1); set_W17(2, 1, 17, 1)

    # Wait, that's rank 18. Let me use the rank-17 from the original program.
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

def _try_rank16(n, m, p, deadline):
    """
    Try to find a rank-16 decomposition using gradient descent with structured initialization.
    """
    rank = 16
    
    for seed in range(300):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Initialize with structured pattern
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Try adaptive gradient descent with momentum
        lr = 0.02
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        best_err = float('inf')
        best_state = (U.copy(), V.copy(), W.copy())
        
        for step in range(800):
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
            
            if err < 1e-8:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
            
            # Compute gradients
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
            
            # Momentum update
            vU = momentum * vU - lr * dU.reshape(n * m, rank)
            vV = momentum * vV - lr * dV.reshape(m * p, rank)
            vW = momentum * vW - lr * dW.reshape(p * n, rank)
            
            U += vU
            V += vV
            W += vW
            
            # Adaptive learning rate
            if step % 100 == 0 and step > 0:
                lr *= 0.95
            
            # Periodic exact check
            if step % 20 == 0:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
        
        # Check best state from this seed
        if best_err < 1e-6:
            U_b, V_b, W_b = best_state
            ok, Us, Vs, Ws = _check_exact(U_b, V_b, W_b, n, m, p)
            if ok:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
    
    return None
