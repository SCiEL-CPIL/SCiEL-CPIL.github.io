import numpy as np
import time

def construct():
    n, m, p = 3, 3, 6
    deadline = time.time() + 110
    
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    # Construct the 3x3x6 tensor directly
    # T[a, b, c] where:
    # a = i*3 + j (A index: i=row of A, j=col of A / row of B)
    # b = j*6 + k (B index: j=row of B, k=col of B)
    # c = k*3 + i (C index: k=col of B / row of C, i=row of C)
    # T[a,b,c] = 1 if consistent, else 0
    
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def check_exact(U, V, W, atol=1e-9):
        """Check if decomposition is exact."""
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_recon, T, atol=atol)

    # 1. Construct Schoolbook Rank 54 Solution
    # This is guaranteed to work and provides a baseline.
    U_base = np.zeros((nm, 54))
    V_base = np.zeros((mp, 54))
    W_base = np.zeros((pn, 54))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_base[a, r] = 1.0
                V_base[b, r] = 1.0
                W_base[c, r] = 1.0
                r += 1
                
    if not check_exact(U_base, V_base, W_base):
        # This should never happen, but if it does, we have a problem.
        pass

    best_U, best_V, best_W = U_base, V_base, W_base
    best_rank = 54
    
    # 2. Try to find a lower rank solution (e.g., 53)
    # We will try a few random restarts for rank 53.
    # If time allows, we might try 52, etc.
    
    def try_find_rank(target_rank, max_restarts=20, max_iters=20):
        if time.time() > deadline - 10:
            return None
            
        T_flat_U = T.reshape(nm, mp * pn)
        T_flat_V = T.transpose(1, 0, 2).reshape(mp, nm * pn)
        T_flat_W = T.transpose(2, 0, 1).reshape(pn, nm * mp)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
                
            U = np.random.randn(nm, target_rank)
            V = np.random.randn(mp, target_rank)
            W = np.random.randn(pn, target_rank)
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Update U
                D_U = np.zeros((mp * pn, target_rank))
                for r_idx in range(target_rank):
                    D_U[:, r_idx] = (V[:, r_idx].reshape(mp, 1) * W[:, r_idx].reshape(1, pn)).reshape(-1)
                U_new = np.zeros((nm, target_rank))
                for a in range(nm):
                    try:
                        U_new[a, :] = np.linalg.lstsq(D_U, T_flat_U[a, :], rcond=None)[0]
                    except:
                        break
                U = U_new
                
                # Update V
                D_V = np.zeros((nm * pn, target_rank))
                for r_idx in range(target_rank):
                    D_V[:, r_idx] = (U[:, r_idx].reshape(nm, 1) * W[:, r_idx].reshape(1, pn)).reshape(-1)
                V_new = np.zeros((mp, target_rank))
                for b in range(mp):
                    try:
                        V_new[b, :] = np.linalg.lstsq(D_V, T_flat_V[b, :], rcond=None)[0]
                    except:
                        break
                V = V_new
                
                # Update W
                D_W = np.zeros((nm * mp, target_rank))
                for r_idx in range(target_rank):
                    D_W[:, r_idx] = (U[:, r_idx].reshape(nm, 1) * V[:, r_idx].reshape(1, mp)).reshape(-1)
                W_new = np.zeros((pn, target_rank))
                for c in range(pn):
                    try:
                        W_new[c, :] = np.linalg.lstsq(D_W, T_flat_W[c, :], rcond=None)[0]
                    except:
                        break
                W = W_new
                
                # Check convergence / validity
                # We need integer/half-integer entries.
                # ALS converges to real values. We must round and check.
                # This is a heuristic.
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if check_exact(U_r, V_r, W_r):
                    return U_r, V_r, W_r
                    
                U_i = np.round(U)
                V_i = np.round(V)
                W_i = np.round(W)
                if check_exact(U_i, V_i, W_i):
                    return U_i, V_i, W_i
        return None

    # Try rank 53
    if time.time() < deadline - 30:
        res = try_find_rank(53, max_restarts=10, max_iters=15)
        if res:
            best_U, best_V, best_W = res
            best_rank = 53
            
    # Try rank 52
    if time.time() < deadline - 30:
        res = try_find_rank(52, max_restarts=5, max_iters=15)
        if res:
            best_U, best_V, best_W = res
            best_rank = 52

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
