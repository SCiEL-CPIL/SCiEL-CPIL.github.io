import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # Target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0

    def verify(U, V, W):
        res = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(res, T, atol=1e-9)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    start_time = time.time()
    deadline = start_time + 110.0

    # Try ranks from 35 down to 30
    for R in range(35, 29, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        attempts = 50 if R == 35 else 20
        
        for attempt in range(attempts):
            if time.time() > deadline:
                break
                
            # Initialize with random floats
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Run ALS
            for _ in range(10):
                if time.time() > deadline:
                    break
                
                # Update U
                num_U = np.einsum('ijk, jr, kr -> ir', T, V, W)
                den_U = (V.T @ V) * (W.T @ W)
                try:
                    U = np.linalg.solve(den_U.T, num_U.T).T
                except np.linalg.LinAlgError:
                    pass
                
                # Update V
                num_V = np.einsum('ijk, ir, kr -> jr', T, U, W)
                den_V = (U.T @ U) * (W.T @ W)
                try:
                    V = np.linalg.solve(den_V.T, num_V.T).T
                except np.linalg.LinAlgError:
                    pass
                
                # Update W
                num_W = np.einsum('ijk, ir, jr -> kr', T, U, V)
                den_W = (U.T @ U) * (V.T @ V)
                try:
                    W = np.linalg.solve(den_W.T, num_W.T).T
                except np.linalg.LinAlgError:
                    pass
                
                # Round to half-integers and verify
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                if verify(U_rounded, V_rounded, W_rounded):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }

    return get_schoolbook()
