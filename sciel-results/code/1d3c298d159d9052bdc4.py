import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, np_ = n * m, m * p, p * n
    
    # Construct the sparse tensor T
    # T is a 3D array of shape (nm, mp, np_)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(i,k)
    T = np.zeros((nm, mp, np_))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify(U, V, W, tol=1e-9):
        # Check if U, V, W exactly decompose T
        np.random.seed(42)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_true = A @ B
        
        U_mat = np.array(U).reshape(nm, -1)
        V_mat = np.array(V).reshape(mp, -1)
        W_mat = np.array(W).reshape(np_, -1)
        
        A_flat = A.flatten() # (nm,)
        B_flat = B.flatten() # (mp,)
        
        M1 = U_mat.T @ A_flat # (R,)
        M2 = V_mat.T @ B_flat # (R,)
        
        prod = M1 * M2
        
        C_flat = W_mat.T @ prod # (np_,)
        
        C_calc = C_flat.reshape(p, n).T
        
        if np.allclose(C_true, C_calc, atol=tol):
            return True
        return False

    def round_to_half(x):
        return np.round(2 * x) / 2.0

    best_result = None
    start_time = time.time()
    deadline = 110.0

    # Try ranks from 35 to 105
    for R in range(35, 106):
        if time.time() > deadline:
            break
            
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(attempt * 100 + R)
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(np_, R)
            
            for it in range(50):
                if time.time() > deadline:
                    break
                
                # Update U
                V_kron_W = np.einsum('ir,jr->ijr', V, W).reshape(mp * np_, R)
                T_1 = T.reshape(nm, mp * np_)
                try:
                    U_T, _, _, _ = np.linalg.lstsq(V_kron_W.T, T_1.T, rcond=None)
                    U = U_T.T
                except:
                    pass
                    
                # Update V
                W_kron_U = np.einsum('ir,jr->ijr', W, U).reshape(np_ * nm, R)
                T_2 = T.transpose(1, 0, 2).reshape(mp, np_ * nm)
                try:
                    V_T, _, _, _ = np.linalg.lstsq(W_kron_U.T, T_2.T, rcond=None)
                    V = V_T.T
                except:
                    pass
                    
                # Update W
                U_kron_V = np.einsum('ir,jr->ijr', U, V).reshape(nm * mp, R)
                T_3 = T.transpose(2, 0, 1).reshape(np_, nm * mp)
                try:
                    W_T, _, _, _ = np.linalg.lstsq(U_kron_V.T, T_3.T, rcond=None)
                    W = W_T.T
                except:
                    pass

            if verify(U, V, W, tol=1e-6):
                U_r = round_to_half(U)
                V_r = round_to_half(V)
                W_r = round_to_half(W)
                
                if verify(U_r, V_r, W_r, tol=1e-9):
                    best_result = {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    return best_result

    # Fallback to schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
