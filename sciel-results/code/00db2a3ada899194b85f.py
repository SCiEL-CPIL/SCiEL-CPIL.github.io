import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]

def _build_rank33():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for k in range(p):
        U[0][col] = 1; V[k][col] = 1; W[2*k][col] = 1; col += 1
        U[3][col] = 1; V[k][col] = 1; W[2*k+1][col] = 1; col += 1
    for x in range(3):
        cs = 2 * x
        U[1][col]=1;U[5][col]=1;V[6+cs][col]=1;V[12+cs+1][col]=1;W[2*cs][col]=1;W[2*cs+3][col]=1;col+=1
        U[4][col]=1;U[5][col]=1;V[6+cs][col]=1;W[2*cs+1][col]=1;W[2*cs+3][col]=-1;col+=1
        U[1][col]=1;V[6+cs+1][col]=1;V[12+cs+1][col]=-1;W[2*cs+2][col]=1;W[2*cs+3][col]=1;col+=1
        U[5][col]=1;V[12+cs][col]=1;V[6+cs][col]=-1;W[2*cs][col]=1;W[2*cs+1][col]=1;col+=1
        U[1][col]=1;U[2][col]=1;V[12+cs+1][col]=1;W[2*cs][col]=-1;W[2*cs+2][col]=1;col+=1
        U[4][col]=1;U[1][col]=-1;V[6+cs][col]=1;V[6+cs+1][col]=1;W[2*cs+3][col]=1;col+=1
        U[2][col]=1;U[5][col]=-1;V[12+cs][col]=1;V[12+cs+1][col]=1;W[2*cs][col]=1;col+=1
    return U, V, W

def _verify(U, V, W, n, m, p):
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    T_est = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr)
    T_target = np.zeros((n*m, m*p, p*n))
    for r in range(n):
        for s in range(m):
            for c in range(p):
                T_target[r*m+s, s*p+c, c*n+r] = 1
    return np.max(np.abs(T_est - T_target)) < 1e-9

def _try_rank32_search(deadline):
    """Try to find a rank-32 decomposition by targeted search."""
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    rank = 32
    
    U33, V33, W33 = _build_rank33()
    U33_arr = np.array(U33, dtype=float)
    V33_arr = np.array(V33, dtype=float)
    W33_arr = np.array(W33, dtype=float)
    
    random.seed(42)
    
    # Try dropping each column and doing a few ALS iterations
    for drop_col in range(33):
        if time.time() > deadline - 30:
            break
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        idx = [i for i in range(33) if i != drop_col]
        U[:, :] = U33_arr[:, idx]
        V[:, :] = V33_arr[:, idx]
        W[:, :] = W33_arr[:, idx]
        
        # Quick ALS: 3 iterations
        for it in range(3):
            if time.time() > deadline - 30:
                break
            
            # Update U rows
            for i in range(nm):
                VW = np.outer(V, W).reshape(mp, pn, rank).transpose(2, 0, 1).reshape(rank, mp*pn)
                Ti = np.zeros((mp, pn))
                r_idx = i // m
                s_idx = i % m
                for c in range(p):
                    Ti[s_idx*p+c, c*n+r_idx] = 1
                Ti_flat = Ti.flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(VW, Ti_flat, rcond=None)
                    U[i, :] = sol
                except:
                    pass
            
            # Update V rows
            for j in range(mp):
                UW = np.outer(U, W).reshape(nm, pn, rank).transpose(2, 0, 1).reshape(rank, nm*pn)
                Tj = np.zeros((nm, pn))
                s_idx = j // p
                c_idx = j % p
                for r in range(n):
                    Tj[r*m+s_idx, c_idx*n+r] = 1
                Tj_flat = Tj.flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(UW, Tj_flat, rcond=None)
                    V[j, :] = sol
                except:
                    pass
            
            # Update W rows
            for k in range(pn):
                UV = np.outer(U, V).reshape(nm, mp, rank).transpose(2, 0, 1).reshape(rank, nm*mp)
                Tk = np.zeros((nm, mp))
                c_idx = k // n
                r_idx = k % n
                for s in range(m):
                    Tk[r*m+s, s*p+c_idx] = 1
                Tk_flat = Tk.flatten()
                try:
                    sol, _, _, _ = np.linalg.lstsq(UV, Tk_flat, rcond=None)
                    W[k, :] = sol
                except:
                    pass
        
        # Check error
        T_est = np.einsum('ir,jr,kr->ijk', U, V, W)
        T_target = np.zeros((nm, mp, pn))
        for r in range(n):
            for s in range(m):
                for c in range(p):
                    T_target[r*m+s, s*p+c, c*n+r] = 1
        err = np.max(np.abs(T_est - T_target))
        
        if err < 1e-9:
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if _verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), n, m, p):
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try to find a rank-32 solution
    result = _try_rank32_search(deadline)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}
    
    # Fall back to rank-33
    U, V, W = _build_rank33()
    return {"U": U, "V": V, "W": W}
