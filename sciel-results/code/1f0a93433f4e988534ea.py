import time
import sys
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the full tensor T as a 3D array
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best = schoolbook()
    best_rank = n * m * p
    
    def verify_decomposition(U_arr, V_arr, W_arr, tol=1e-10):
        recon = np.zeros((nm, mp, pn), dtype=float)
        for t in range(U_arr.shape[1]):
            recon += np.einsum('a,b,c->abc', U_arr[:, t], V_arr[:, t], W_arr[:, t])
        max_err = np.max(np.abs(recon - T))
        return max_err < tol, max_err
    
    def try_als(r, n_trials=100, max_iter=300):
        np.random.seed(12345 + r * 1000)
        
        for trial in range(n_trials):
            if time.time() > deadline - 15:
                return None
            
            # Use small integer/half-integer initializations for better exact solutions
            scale = np.random.choice([1, 2, 3])
            U = np.random.randint(-scale, scale + 1, size=(nm, r)).astype(float)
            V = np.random.randint(-scale, scale + 1, size=(mp, r)).astype(float)
            W = np.random.randint(-scale, scale + 1, size=(pn, r)).astype(float)
            # Add small random perturbation to break symmetry
            U += 0.1 * np.random.randn(nm, r)
            V += 0.1 * np.random.randn(mp, r)
            W += 0.1 * np.random.randn(pn, r)
            
            VW = np.einsum('bt,ct->bct', V, W)
            
            for it in range(max_iter):
                if time.time() > deadline - 15:
                    return None
                
                # Update U
                VW_flat = VW.reshape(mp * pn, r)
                T_flat = T.reshape(nm, mp * pn)
                
                U_new = np.zeros((nm, r))
                for a in range(nm):
                    U_new[a, :] = np.linalg.lstsq(VW_flat, T_flat[a, :], rcond=None)[0]
                U = U_new
                
                # Update V
                UW = np.einsum('at,ct->act', U, W)
                UW_flat = UW.reshape(nm * pn, r)
                T_for_V = np.zeros((mp, nm * pn))
                for b in range(mp):
                    T_for_V[b, :] = T[:, b, :].flatten()
                
                V_new = np.zeros((mp, r))
                for b in range(mp):
                    V_new[b, :] = np.linalg.lstsq(UW_flat, T_for_V[b, :], rcond=None)[0]
                V = V_new
                
                # Update W
                UV = np.einsum('at,bt->abt', U, V)
                UV_flat = UV.reshape(nm * mp, r)
                T_for_W = np.zeros((pn, nm * mp))
                for c in range(pn):
                    T_for_W[c, :] = T[:, :, c].flatten()
                
                W_new = np.zeros((pn, r))
                for c in range(pn):
                    W_new[c, :] = np.linalg.lstsq(UV_flat, T_for_W[c, :], rcond=None)[0]
                W = W_new
                
                VW = np.einsum('bt,ct->bct', V, W)
                
                if it % 20 == 19:
                    ok, err = verify_decomposition(U, V, W)
                    if ok:
                        return U, V, W
            
            ok, err = verify_decomposition(U, V, W)
            if ok:
                return U, V, W
        
        return None
    
    # Known rank for 3x4x5 is 50. Try from 50 upward to 59.
    for r in range(50, min(best_rank, 60)):
        if time.time() > deadline - 20:
            break
        
        result = try_als(r, n_trials=50, max_iter=150)
        if result is not None:
            U_arr, V_arr, W_arr = result
            U_list = [[round(val, 12) for val in row] for row in U_arr]
            V_list = [[round(val, 12) for val in row] for row in V_arr]
            W_list = [[round(val, 12) for val in row] for row in W_arr]
            
            ok, err = verify_decomposition(np.array(U_list), np.array(V_list), np.array(W_list))
            if ok:
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                break
    
    return best
# EVOLVE-BLOCK-END
