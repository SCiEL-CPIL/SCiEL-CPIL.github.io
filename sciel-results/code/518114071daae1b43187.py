import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin before 120s limit
    
    # Fallback: Rank 17 solution
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

    # Block 2: Direct 2x2x1 (j in {0,1}, k=2)
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

    # Block 3: Direct 2x1x3 (j=2, k in {0,1,2})
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

    best_rank = 17
    best_U, best_V, best_W = U17, V17, W17

    # Build the target tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1.0

    def compute_error(U, V, W):
        S = np.einsum('ar,br,cr->abc', U, V, W)
        D = S - T
        return np.sum(D**2)

    def try_find_rank_16():
        if time.time() > deadline:
            return None
            
        target_rank = 16
        
        # Initialize from rank 17 solution by dropping the last column
        # This is a good starting point because it's close to a solution.
        U = U17[:, :target_rank].copy()
        V = V17[:, :target_rank].copy()
        W = W17[:, :target_rank].copy()
        
        # Also try random initializations
        initializations = [(U, V, W)]
        np.random.seed(42)
        for _ in range(10):
            U_r = np.random.randn(n*m, target_rank) * 0.1
            V_r = np.random.randn(m*p, target_rank) * 0.1
            W_r = np.random.randn(p*n, target_rank) * 0.1
            initializations.append((U_r, V_r, W_r))

        best_err = float('inf')
        best_sol = None

        for U_init, V_init, W_init in initializations:
            if time.time() > deadline:
                break
                
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            err = compute_error(U_cur, V_cur, W_cur)
            if err < 1e-10:
                return U_cur, V_cur, W_cur
                
            # Gradient Descent with Adam-like update
            # Simple implementation: accumulate gradients, update with momentum
            lr = 0.01
            momentum = 0.9
            v_U = np.zeros_like(U_cur)
            v_V = np.zeros_like(V_cur)
            v_W = np.zeros_like(W_cur)
            
            for step in range(2000):
                if time.time() > deadline:
                    break
                    
                S = np.einsum('ar,br,cr->abc', U_cur, V_cur, W_cur)
                D = S - T
                
                g_U = np.einsum('abc,br,cr->ar', D, V_cur, W_cur)
                g_V = np.einsum('abc,ar,cr->br', D, U_cur, W_cur)
                g_W = np.einsum('abc,ar,br->cr', D, U_cur, V_cur)
                
                # Update
                v_U = momentum * v_U + (1 - momentum) * g_U
                v_V = momentum * v_V + (1 - momentum) * g_V
                v_W = momentum * v_W + (1 - momentum) * g_W
                
                U_cur -= lr * v_U
                V_cur -= lr * v_V
                W_cur -= lr * v_W
                
                err = compute_error(U_cur, V_cur, W_cur)
                
                if err < best_err:
                    best_err = err
                    best_sol = (U_cur.copy(), V_cur.copy(), W_cur.copy())
                    
                if err < 1e-10:
                    return U_cur, V_cur, W_cur
                    
        if best_sol and best_err < 1e-8:
            return best_sol
            
        return None

    # Attempt to find rank 16
    result_16 = try_find_rank_16()
    if result_16:
        U, V, W = result_16
        # Verify and potentially round to half-integers
        if compute_error(U, V, W) < 1e-8:
            # Try to round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if compute_error(U_rounded, V_rounded, W_rounded) < 1e-8:
                best_rank = 16
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
            else:
                best_rank = 16
                best_U, best_V, best_W = U, V, W

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
