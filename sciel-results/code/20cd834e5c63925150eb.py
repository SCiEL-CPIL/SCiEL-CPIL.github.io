import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = 2, 3, 3
    U_size = n * m  # 6
    V_size = m * p  # 9
    W_size = p * n  # 6
    
    # Target tensor T of shape (n*m, m*p, p*n)
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    T = np.zeros((U_size, V_size, W_size))
    indices_u = []
    indices_v = []
    indices_w = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                indices_u.append(u_idx)
                indices_v.append(v_idx)
                indices_w.append(w_idx)
                
    indices_u = np.array(indices_u)
    indices_v = np.array(indices_v)
    indices_w = np.array(indices_w)
    
    start_time = time.time()
    deadline = start_time + 110.0

    def verify(U, V, W):
        rank = U.shape[1]
        T_calc = np.zeros((U_size, V_size, W_size))
        for c in range(rank):
            T_calc += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        if not np.allclose(T_calc, T, atol=1e-9):
            return False
        return True

    def snap_and_verify(U, V, W):
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def polish_and_verify(U, V, W, rank):
        """High-precision polish: fix one factor at a time with least squares, then snap."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for _ in range(30):
            if time.time() > deadline:
                break
            # Update U: for each column r, solve V[:,r]^T ⊗ W[:,r] * U[:,r] = target_r
            # target_r[u] = sum over (v,w) where T[u,v,w]=1 of ... 
            # Actually: T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
            # Fix V,W: for each u, sum_r U[u,r] * (V⊗W)[:,r][u] = T[u,:,:].flatten()
            # Better: reshape into matrix form
            
            # Matrix form: T_{u,(v,w)} = sum_r U[u,r] * (V[:,r] ⊗ W[:,r])[v*w_idx]
            # Let's use the standard ALS approach
            
            # Update U: A_u = sum_r U[u,r] * outer(V[:,r], W[:,r])
            # We want A to match T
            # For fixed V,W: T.reshape(U_size, -1) = U @ (V⊗W).T where (V⊗W) is V_size*W_size x rank
            
            VW = np.zeros((V_size * W_size, rank))
            for r in range(rank):
                VW[:, r] = np.outer(V[:, r], W[:, r]).flatten()
            
            T_mat = T.reshape(U_size, V_size * W_size)
            # T_mat = U @ VW.T  =>  U = T_mat @ (VW.T)^+ 
            try:
                U_new = np.linalg.lstsq(VW.T, T_mat.T, rcond=None)[0].T
                # Check if it's an improvement
                if np.allclose(U_new, U, atol=1e-12):
                    pass
                else:
                    U = U_new
            except:
                pass
            
            # Update V
            UW = np.zeros((U_size * W_size, rank))
            for r in range(rank):
                UW[:, r] = np.outer(U[:, r], W[:, r]).flatten()
            
            T_mat_v = T.transpose(1, 0, 2).reshape(V_size, U_size * W_size)
            try:
                V_new = np.linalg.lstsq(UW.T, T_mat_v.T, rcond=None)[0].T
                if np.allclose(V_new, V, atol=1e-12):
                    pass
                else:
                    V = V_new
            except:
                pass
            
            # Update W
            UV = np.zeros((U_size * V_size, rank))
            for r in range(rank):
                UV[:, r] = np.outer(U[:, r], V[:, r]).flatten()
            
            T_mat_w = T.transpose(2, 0, 1).reshape(W_size, U_size * V_size)
            try:
                W_new = np.linalg.lstsq(UV.T, T_mat_w.T, rcond=None)[0].T
                if np.allclose(W_new, W, atol=1e-12):
                    pass
                else:
                    W = W_new
            except:
                pass
        
        # Final snap to half-integers
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        if verify(U_snap, V_snap, W_snap):
            return U_snap, V_snap, W_snap
        return None

    def residual(params, R):
        # params is a flat array of length R * (n*m + m*p + p*n)
        U = params[:R*U_size].reshape(U_size, R)
        V = params[R*U_size:R*U_size + R*V_size].reshape(V_size, R)
        W = params[R*U_size + R*V_size:].reshape(W_size, R)
        
        # Compute sum_r U[u,r] V[v,r] W[w,r] for the 18 points
        U_sel = U[indices_u]
        V_sel = V[indices_v]
        W_sel = W[indices_w]
        
        # Product element-wise then sum over R
        prod = U_sel * V_sel * W_sel
        result = prod.sum(axis=1)
        
        # Target is 1 for all 18 points
        return result - 1.0

    # Try ranks from 14 upwards
    # Rank 14 is the known minimum for 2x3x3
    for rank in [14, 15, 16]:
        if time.time() > deadline:
            break
        
        # Number of restarts: give rank 14 the most chances
        # Reduce restarts slightly to leave time for polish
        max_restarts = 40 if rank == 14 else 15
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(restart * 1000 + rank)
            
            # Random initialization
            # Scaling is important. Entries in optimal solutions are often small integers or half-integers.
            # Initializing with small random values helps convergence.
            U_init = np.random.randn(U_size, rank) * 0.5
            V_init = np.random.randn(V_size, rank) * 0.5
            W_init = np.random.randn(W_size, rank) * 0.5
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = least_squares(
                    residual, 
                    x0, 
                    args=(rank,),
                    method='lm', # Levenberg-Marquardt is good for least squares
                    max_nfev=200
                )
                
                x_opt = res.x
                U_opt = x_opt[:rank*U_size].reshape(U_size, rank)
                V_opt = x_opt[rank*U_size:rank*U_size + rank*V_size].reshape(V_size, rank)
                W_opt = x_opt[rank*U_size + rank*V_size:].reshape(W_size, rank)
                
                result = snap_and_verify(U_opt, V_opt, W_opt)
                if result:
                    U_final, V_final, W_final = result
                    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
            except Exception:
                continue

    # Fallback to schoolbook
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
