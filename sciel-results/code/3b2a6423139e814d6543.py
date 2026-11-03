import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 16
    mp = m * p  # 28
    pn = p * n  # 28
    
    # Build target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    deadline = time.time() + 140
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = nm * mp * pn
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    best_U, best_V, best_W, best_rank = schoolbook()
    
    # Try to find a lower rank decomposition using ALS with half-integer rounding
    for target_rank in range(28, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(30):
            if time.time() > deadline:
                break
            
            np.random.seed(target_rank * 1000 + trial)
            U_t = np.random.randn(nm, target_rank)
            V_t = np.random.randn(mp, target_rank)
            W_t = np.random.randn(pn, target_rank)
            
            # ALS iterations
            for it in range(150):
                if time.time() > deadline:
                    break
                
                # Update U
                VtV = V_t.T @ V_t
                WtW = W_t.T @ W_t
                M = VtV * WtW
                TW = np.einsum('abd,dc->abc', T, W_t)
                rhs = np.einsum('abc,bc->ac', TW, V_t)
                try:
                    U_t = np.linalg.solve(M, rhs.T).T
                except:
                    U_t = np.linalg.lstsq(M, rhs.T, rcond=None)[0].T
                
                # Update V
                UtU = U_t.T @ U_t
                WtW = W_t.T @ W_t
                M = UtU * WtW
                TU = np.einsum('abd,ac->bcd', T, U_t)
                rhs = np.einsum('bcd,dc->bc', TU, W_t)
                try:
                    V_t = np.linalg.solve(M, rhs.T).T
                except:
                    V_t = np.linalg.lstsq(M, rhs.T, rcond=None)[0].T
                
                # Update W
                UtU = U_t.T @ U_t
                VtV = V_t.T @ V_t
                M = UtU * VtV
                TV = np.einsum('abd,bc->adc', T, V_t)
                rhs = np.einsum('adc,ac->dc', TV, U_t)
                try:
                    W_t = np.linalg.solve(M, rhs.T).T
                except:
                    W_t = np.linalg.lstsq(M, rhs.T, rcond=None)[0].T
            
            # Round to half-integers and check
            U_r = np.round(U_t * 2) / 2
            V_r = np.round(V_t * 2) / 2
            W_r = np.round(W_t * 2) / 2
            
            T_approx = np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
            if np.allclose(T_approx, T, atol=1e-9):
                best_U, best_V, best_W, best_rank = U_r, V_r, W_r, target_rank
                found = True
                break
        
        if found:
            break
    
    # Verify the result
    T_check = np.einsum('ac,bc,dc->abd', best_U, best_V, best_W)
    if not np.allclose(T_check, T, atol=1e-9):
        # Fall back to schoolbook
        best_U, best_V, best_W, best_rank = schoolbook()
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
