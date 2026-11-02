import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 2, 3, 3
    nm, mp, pn = n*m, m*p, p*n
    total = n*m*p
    
    def schoolbook():
        rank = total
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def compute_T(U, V, W):
        Ur = U.reshape(n, m, -1)
        Vr = V.reshape(m, p, -1)
        Wr = W.reshape(p, n, -1)
        return np.einsum('ijr,jkr,kir->ijk', Ur, Vr, Wr)
    
    def tensor_err(U, V, W):
        T = compute_T(U, V, W)
        return np.max(np.abs(T - 1.0))
    
    def gradient(U, V, W):
        T = compute_T(U, V, W)
        E = T - 1.0
        Ur = U.reshape(n, m, -1)
        Vr = V.reshape(m, p, -1)
        Wr = W.reshape(p, n, -1)
        R = Ur.shape[2]
        dUr = np.einsum('ijk,jkr,kir->ijr', E, Vr, Wr)
        dVr = np.einsum('ijk,ijr,kir->jkr', E, Ur, Wr)
        dWr = np.einsum('ijk,ijr,jkr->kir', E, Ur, Vr)
        return dUr.reshape(-1,R), dVr.reshape(-1,R), dWr.reshape(-1,R), np.max(np.abs(E))
    
    def try_round(U, V, W):
        for scale in [2, 4, 8]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if tensor_err(U_r, V_r, W_r) < 1e-6:
                return U_r, V_r, W_r
        return None
    
    best_result = schoolbook()
    
    for R in range(9, total):
        if time.time() > deadline:
            break
        found = False
        for trial in range(300):
            if time.time() > deadline:
                break
            np.random.seed(R * 100000 + trial)
            U = np.random.uniform(-5, 5, (nm, R))
            V = np.random.uniform(-5, 5, (mp, R))
            W = np.random.uniform(-5, 5, (pn, R))
            
            lr = 0.005
            for it in range(4000):
                if time.time() > deadline:
                    break
                dU, dV, dW, err = gradient(U, V, W)
                if err < 1e-10:
                    break
                U_n = U - lr*dU
                V_n = V - lr*dV
                W_n = W - lr*dW
                ne = tensor_err(U_n, V_n, W_n)
                if ne < err:
                    U, V, W, err = U_n, V_n, W_n, ne
                    if err < 1e-4:
                        lr = min(lr*1.1, 0.015)
                else:
                    lr *= 0.5
                    if lr < 1e-9:
                        break
            
            if tensor_err(U, V, W) < 1e-6:
                r = try_round(U, V, W)
                if r is not None:
                    best_result = {"U": r[0].tolist(), "V": r[1].tolist(), "W": r[2].tolist()}
                    found = True
                    break
        if found:
            break
    
    return best_result
