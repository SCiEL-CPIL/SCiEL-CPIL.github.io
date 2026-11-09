import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 4, 4, 6

    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    R_init = 77
    U = np.zeros((16, R_init), dtype=float)
    V = np.zeros((24, R_init), dtype=float)
    W = np.zeros((24, R_init), dtype=float)

    col = 0
    for s in range(7):
        if time.time() > deadline:
            break
        for si in range(7):
            for a in range(4):
                if alpha[si][a] == 0: continue
                x_row, x_col = a // 2, a % 2
                for r in range(4):
                    if alpha[s][r] == 0: continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                    U[u_idx, col] += alpha[si][a] * alpha[s][r]

            for b in range(4):
                if beta[si][b] == 0: continue
                y_row, y_inner_col = b // 2, b % 2
                for c in range(4):
                    if beta[s][c] == 0: continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                    V[v_idx, col] += beta[si][b] * beta[s][c]

            for (a, b, gamma) in recon[s]:
                for (za, zb, zgamma) in recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            col += 1

        for u in range(2):
            for v in range(2):
                for r in range(4):
                    if alpha[s][r] == 0: continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                    U[u_idx, col] += alpha[s][r]

                for c in range(4):
                    if beta[s][c] == 0: continue
                    ac, bc = c // 2, c % 2
                    v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                    V[v_idx, col] += beta[s][c]

                for (a, b, gamma) in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                col += 1

    # Remove zero columns
    nonzero_mask = np.any(U != 0, axis=0) & np.any(V != 0, axis=0) & np.any(W != 0, axis=0)
    if not np.all(nonzero_mask):
        keep = np.where(nonzero_mask)[0]
        U = U[:, keep]
        V = V[:, keep]
        W = W[:, keep]

    # Remove proportional columns
    # If T_j = lambda * T_k, we can merge them.
    # T_j = U_j o V_j o W_j.
    # U_j = a U_k, V_j = b V_k, W_j = c W_k => T_j = abc T_k.
    # We keep the merged term in the slot of k (or j) and remove the other.
    
    R_current = U.shape[1]
    if R_current == 0:
        return {"U": [], "V": [], "W": []}
        
    kept_indices = []
    U_kept = []
    V_kept = []
    W_kept = []
    
    for j in range(R_current):
        u_j = U[:, j]
        v_j = V[:, j]
        w_j = W[:, j]
        
        # Check against kept columns
        merged = False
        for k_idx in range(len(kept_indices)):
            u_k = U_kept[k_idx]
            v_k = V_kept[k_idx]
            w_k = W_kept[k_idx]
            
            # Check proportionality
            # Find non-zero in u_j
            nz_u = np.nonzero(u_j)[0]
            if len(nz_u) == 0:
                continue # Should not happen due to zero removal
            
            idx_u = nz_u[0]
            a = u_k[idx_u] / u_j[idx_u]
            
            # Check if U_k is proportional to U_j
            if not np.allclose(u_k, a * u_j, atol=1e-9):
                continue
                
            # Find non-zero in v_j
            nz_v = np.nonzero(v_j)[0]
            if len(nz_v) == 0:
                continue
                
            idx_v = nz_v[0]
            b = v_k[idx_v] / v_j[idx_v]
            
            if not np.allclose(v_k, b * v_j, atol=1e-9):
                continue
                
            # Find non-zero in w_j
            nz_w = np.nonzero(w_j)[0]
            if len(nz_w) == 0:
                continue
                
            idx_w = nz_w[0]
            c = w_k[idx_w] / w_j[idx_w]
            
            if not np.allclose(w_k, c * w_j, atol=1e-9):
                continue
                
            # They are proportional.
            # T_j = (a*b*c) T_k ?
            # Wait. U_j = (1/a) U_k ? No.
            # u_k = a u_j => u_j = (1/a) u_k.
            # v_k = b v_j => v_j = (1/b) v_k.
            # w_k = c w_j => w_j = (1/c) w_k.
            # T_j = u_j o v_j o w_j = (1/a)(1/b)(1/c) (u_k o v_k o w_k) = (1/(abc)) T_k.
            # So T_j = lambda T_k where lambda = 1/(abc).
            # We want to merge T_k + T_j = T_k + lambda T_k = (1+lambda) T_k.
            # New vector for U: (1+lambda) u_k.
            # We can update the kept column k_idx.
            
            lambda_val = 1.0 / (a * b * c)
            
            U_kept[k_idx] = (1.0 + lambda_val) * u_k
            # V and W can remain u_k, v_k, w_k?
            # No, the term is U_new o V_k o W_k.
            # (1+lambda) u_k o v_k o w_k = (1+lambda) T_k = T_k + T_j.
            # So we just scale U_kept[k_idx].
            
            merged = True
            break
            
        if not merged:
            kept_indices.append(j)
            U_kept.append(u_j.copy())
            V_kept.append(v_j.copy())
            W_kept.append(w_j.copy())

    if len(kept_indices) < R_current:
        U = np.column_stack(U_kept)
        V = np.column_stack(V_kept)
        W = np.column_stack(W_kept)
    else:
        # No merges found, keep original
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
