import numpy as np
import time

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    # Schoolbook rank
    rank_school = n * m * p
    best_result = None
    best_rank = rank_school
    
    # Deadline
    deadline = time.time() + 140

    # Try to find a lower rank
    # We'll try ranks from 84 down to 60.
    
    for rank in range(84, 59, -1):
        if time.time() > deadline - 10:
            break
            
        found = False
        for attempt in range(30):
            if time.time() > deadline - 10:
                break
                
            rng = np.random.default_rng(attempt * 1000 + rank)
            U = rng.standard_normal((nm, rank)) * 0.5
            V = rng.standard_normal((mp, rank)) * 0.5
            W = rng.standard_normal((pn, rank)) * 0.5
            
            # ALS
            for it in range(100):
                if time.time() > deadline - 10:
                    break
                    
                # Update U
                # M_U[b, c, r] = V[b, r] * W[c, r]
                M_U = np.einsum('br,cr->bcr', V, W).reshape(-1, rank)
                # G_U = M_U^T M_U
                G_U = M_U.T @ M_U
                # B_U[i, r] = sum_{b,c} T[i, b, c] M_U[b*c, r]
                # T_flat[i, b*c] = T[i, b, c]
                T_flat = T.reshape(nm, -1)
                B_U = T_flat @ M_U
                # Solve U G_U = B_U
                # U = B_U G_U^{-1}
                try:
                    G_U_inv = np.linalg.inv(G_U)
                    U = B_U @ G_U_inv
                except:
                    pass

                # Update V
                M_V = np.einsum('ar,cr->acr', U, W).reshape(-1, rank)
                G_V = M_V.T @ M_V
                # B_V[j, r] = sum_{a,c} T[a, j, c] M_V[a*c, r]
                # T is (nm, mp, pn). We need to extract slices for each j.
                # T[:, j, :] has shape (nm, pn). Flatten to (nm*pn,).
                B_V = np.zeros((mp, rank))
                for j in range(mp):
                    T_slice = T[:, j, :].reshape(-1)
                    B_V[j] = T_slice @ M_V
                try:
                    G_V_inv = np.linalg.inv(G_V)
                    V = B_V @ G_V_inv
                except:
                    pass

                # Update W
                M_W = np.einsum('ar,br->abr', U, V).reshape(-1, rank)
                G_W = M_W.T @ M_W
                # B_W[k, r] = sum_{a,b} T[a, b, k] M_W[a*b, k]
                B_W = np.zeros((pn, rank))
                for k in range(pn):
                    T_slice = T[:, :, k].reshape(-1)
                    B_W[k] = T_slice @ M_W
                try:
                    G_W_inv = np.linalg.inv(G_W)
                    W = B_W @ G_W_inv
                except:
                    pass

                # Check error
                if it % 10 == 0:
                    R = np.einsum('ar,br,cr->abc', U, V, W)
                    err = np.linalg.norm(R - T)
                    if err < 1e-8:
                        break

            # Final check
            R = np.einsum('ar,br,cr->abc', U, V, W)
            err = np.linalg.norm(R - T)
            if err < 1e-8:
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                R_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
                err_r = np.linalg.norm(R_r - T)
                if err_r < 1e-8:
                    best_rank = rank
                    best_result = {
                        "U": U_r.tolist(),
                        "V": V_r.tolist(),
                        "W": W_r.tolist()
                    }
                    found = True
                    break
        
        if found:
            break

    if best_result is not None:
        return best_result
    else:
        # Fall back to schoolbook
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
