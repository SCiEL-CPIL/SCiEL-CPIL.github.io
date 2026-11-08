import time
import random
import numpy as np

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    T_flat = T.reshape(n*m, -1)
    T_reshaped_V = T.transpose(1, 0, 2).reshape(m*p, -1)
    T_reshaped_W = T.transpose(2, 0, 1).reshape(p*n, -1)

    def verify(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        R = U_np.shape[1]
        
        # Check non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                    if abs(val - 1.0) > 1e-6:
                        return False
        
        # Check zero entries
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i1, j1 = divmod(a, m)
                    j2, k1 = divmod(b, p)
                    k2, i2 = divmod(c, n)
                    if j1 == j2 and k1 == k2 and i1 == i2:
                        continue
                    val = np.sum(U_np[a, :] * V_np[b, :] * W_np[c, :])
                    if abs(val) > 1e-6:
                        return False
        return True

    def try_rank(R):
        # Initialize
        best_sol = None
        for attempt in range(20):
            if time.time() > deadline:
                break
            U = np.random.uniform(-1, 1, (n*m, R))
            V = np.random.uniform(-1, 1, (m*p, R))
            W = np.random.uniform(-1, 1, (p*n, R))
            
            for it in range(100):
                if time.time() > deadline:
                    break
                
                # Solve U
                Z = (V * W).T.reshape(R, -1)
                U, _, _, _ = np.linalg.lstsq(Z.T, T_flat.T, rcond=None)
                U = U.T
                
                # Solve V
                Phi = np.array([np.outer(U[:, r], W[:, r]).flatten() for r in range(R)])
                V, _, _, _ = np.linalg.lstsq(Phi.T, T_reshaped_V.T, rcond=None)
                V = V.T
                
                # Solve W
                Phi = np.array([np.outer(U[:, r], V[:, r]).flatten() for r in range(R)])
                W, _, _, _ = np.linalg.lstsq(Phi.T, T_reshaped_W.T, rcond=None)
                W = W.T
                
                # Snap
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                if it % 10 == 0:
                    if verify(U.tolist(), V.tolist(), W.tolist()):
                        return U.tolist(), V.tolist(), W.tolist()
                        
        return None

    # Try ranks
    for R in range(50, 61):
        if time.time() > deadline:
            break
        sol = try_rank(R)
        if sol:
            U, V, W = sol
            return {"U": U, "V": V, "W": W}
            
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
