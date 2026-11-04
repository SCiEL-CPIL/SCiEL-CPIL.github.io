import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety

    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1

    # First, try to find a rank-16 decomposition
    # Strategy: Use Strassen for the 2x2x2 core (rank 7)
    # Then find a rank-9 decomposition for the residual.
    
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
    Try to find a rank-16 decomposition.
    Uses Strassen for the 2x2x2 core (rank 7).
    Then searches for a rank-9 decomposition of the residual.
    """
    rank = 16
    res_rank = rank - 7
    
    # Initialize Strassen part (indices 0-6)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    def set_U(i, j, r, val): U[i * m + j, r] = val
    def set_V(j, k, r, val): V[j * p + k, r] = val
    def set_W(k, i, r, val): W[k * n + i, r] = val

    # Strassen 2x2x2
    set_U(0, 0, 0, 1); set_U(1, 1, 0, 1)
    set_V(0, 0, 0, 1); set_V(1, 1, 0, 1)
    set_W(0, 0, 0, 1); set_W(1, 1, 0, 1)
    
    set_U(1, 0, 1, 1); set_U(1, 1, 1, 1)
    set_V(0, 0, 1, 1)
    set_W(0, 1, 1, 1); set_W(1, 1, 1, -1)
    
    set_U(0, 0, 2, 1)
    set_V(0, 1, 2, 1); set_V(1, 1, 2, -1)
    set_W(1, 0, 2, 1); set_W(1, 1, 2, 1)
    
    set_U(1, 1, 3, 1)
    set_V(1, 0, 3, 1); set_V(0, 0, 3, -1)
    set_W(0, 0, 3, 1); set_W(0, 1, 3, 1)
    
    set_U(0, 0, 4, 1); set_U(0, 1, 4, 1)
    set_V(1, 1, 4, 1)
    set_W(0, 0, 4, -1); set_W(1, 0, 4, 1)
    
    set_U(1, 0, 5, 1); set_U(0, 0, 5, -1)
    set_V(0, 0, 5, 1); set_V(0, 1, 5, 1)
    set_W(1, 1, 5, 1)
    
    set_U(0, 1, 6, 1); set_U(1, 1, 6, -1)
    set_V(1, 0, 6, 1); set_V(1, 1, 6, 1)
    set_W(0, 0, 6, 1)

    # Compute residual
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    residual = 1.0 - prod
    
    # We need to fill indices 7 to 15 (9 terms) to match the residual.
    # The residual is non-zero only where j=2 or k=2.
    # Let's use a discrete search for the remaining 9 terms.
    # Values are likely in {-1, 0, 1}.
    
    # Flatten the remaining part for easier manipulation
    # U_rem: (6, 9), V_rem: (9, 9), W_rem: (6, 9)
    
    # We will try random integer initializations and local search.
    for seed in range(100):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Initialize remaining terms with random integers -1, 0, 1
        U_rem = np.random.randint(-1, 2, size=(n * m, res_rank))
        V_rem = np.random.randint(-1, 2, size=(m * p, res_rank))
        W_rem = np.random.randint(-1, 2, size=(p * n, res_rank))
        
        U[:, 7:] = U_rem
        V[:, 7:] = V_rem
        W[:, 7:] = W_rem
        
        # Check if it matches
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
        
        if np.allclose(prod, 1.0, atol=1e-9):
            ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
            if ok:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
        
        # Local search: try to reduce error
        best_err = np.sum((prod - 1.0)**2)
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for step in range(1000):
            if time.time() > deadline:
                break
            
            # Pick a random variable to change
            var_type = np.random.randint(3) # 0: U, 1: V, 2: W
            if var_type == 0:
                idx = np.random.randint(n * m)
                col = np.random.randint(res_rank)
                old_val = U[idx, 7 + col]
                new_val = np.random.randint(-1, 2)
                if new_val == old_val: continue
                U[idx, 7 + col] = new_val
            elif var_type == 1:
                idx = np.random.randint(m * p)
                col = np.random.randint(res_rank)
                old_val = V[idx, 7 + col]
                new_val = np.random.randint(-1, 2)
                if new_val == old_val: continue
                V[idx, 7 + col] = new_val
            else:
                idx = np.random.randint(p * n)
                col = np.random.randint(res_rank)
                old_val = W[idx, 7 + col]
                new_val = np.random.randint(-1, 2)
                if new_val == old_val: continue
                W[idx, 7 + col] = new_val
            
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            err = np.sum((prod - 1.0)**2)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
                if err < 1e-9:
                    ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                    if ok:
                        return {
                            "U": Us.tolist(),
                            "V": Vs.tolist(),
                            "W": Ws.tolist()
                        }
            else:
                # Revert
                if var_type == 0:
                    U[idx, 7 + col] = old_val
                elif var_type == 1:
                    V[idx, 7 + col] = old_val
                else:
                    W[idx, 7 + col] = old_val
    
    return None

