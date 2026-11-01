import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1.0

    def error(U, V, W):
        T_recon = np.einsum('ac,bc,dc->abd', U, V, W)
        return np.sum((T_recon - T) ** 2)

    # Try rank 15
    r = 15
    best_error = np.inf
    best_U, best_V, best_W = None, None, None
    
    start_time = time.time()
    while time.time() - start_time < 100:
        U = np.random.choice([-1, 0, 1], size=(n*m, r))
        V = np.random.choice([-1, 0, 1], size=(m*p, r))
        W = np.random.choice([-1, 0, 1], size=(p*n, r))
        
        # Local search
        for _ in range(1000):
            if time.time() - start_time > 110:
                break
            
            # Pick a random element to perturb
            idx = np.random.randint(n*m * r)
            i, j = divmod(idx, r)
            old_val = U[i, j]
            U[i, j] = np.random.choice([-1, 0, 1])
            
            new_err = error(U, V, W)
            if new_err < best_error:
                best_error = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_error == 0:
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            else:
                U[i, j] = old_val
                
            # Same for V and W
            idx = np.random.randint(m*p * r)
            i, j = divmod(idx, r)
            old_val = V[i, j]
            V[i, j] = np.random.choice([-1, 0, 1])
            
            new_err = error(U, V, W)
            if new_err < best_error:
                best_error = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_error == 0:
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            else:
                V[i, j] = old_val
                
            idx = np.random.randint(p*n * r)
            i, j = divmod(idx, r)
            old_val = W[i, j]
            W[i, j] = np.random.choice([-1, 0, 1])
            
            new_err = error(U, V, W)
            if new_err < best_error:
                best_error = new_err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                if best_error == 0:
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
            else:
                W[i, j] = old_val

    if best_error == 0:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
    # Fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
