import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _build_rank55_decomp():
    """Build the rank-55 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 55
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Strassen coefficients for 2x2x2
    L = [
        {0: 1, 3: 1},       # M1: a00, a11
        {2: 1, 3: 1},       # M2: a10, a11
        {0: 1},             # M3: a00
        {3: 1},             # M4: a11
        {0: 1, 1: 1},       # M5: a00, a01
        {2: 1, 0: -1},      # M6: a10, -a00
        {1: 1, 3: -1}       # M7: a01, -a11
    ]
    R = [
        {0: 1, 3: 1},       # M1: b00, b11
        {0: 1},             # M2: b00
        {1: 1, 3: -1},      # M3: b01, -b11
        {2: 1, 0: -1},      # M4: b10, -b00
        {3: 1},             # M5: b11
        {0: 1, 1: 1},       # M6: b00, b01
        {2: 1, 3: 1}        # M7: b10, b11
    ]
    Wc = [
        {0: 1, 3: 1},       # M1: c00, c11
        {2: 1, 3: -1},      # M2: c10, -c11
        {1: 1, 3: 1},       # M3: c01, c11
        {0: 1, 2: 1},       # M4: c00, c10
        {1: 1, 0: -1},      # M5: c01, -c00
        {3: 1},             # M6: c11
        {0: 1}              # M7: c00
    ]
    
    # 5 Strassen blocks for 2x2 * 2x2 subproblems
    for block in range(5):
        k_offset = block * 2
        for s in range(7):
            r = col
            for local_idx, val in L[s].items():
                i = local_idx // 2
                j = local_idx % 2
                U[i * m + j, r] = val
            for local_idx, val in R[s].items():
                j = local_idx // 2
                k = local_idx % 2
                V[j * p + (k_offset + k), r] = val
            for local_idx, val in Wc[s].items():
                i = local_idx // 2
                k = local_idx % 2
                W[(k_offset + k) * n + i, r] = val
            col += 1
    
    # Schoolbook for 2x1 * 1x10 part
    for i in range(2):
        for k in range(10):
            r = col
            U[i * m + 2, r] = 1
            V[2 * p + k, r] = 1
            W[k * n + i, r] = 1
            col += 1
    
    return U, V, W

def _check_decomposition(U, V, W):
    """Check if the decomposition is valid."""
    n, m, p = DIMENSIONS
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.max(np.abs(T - 1.0))

def _try_reduce_rank(U, V, W, deadline):
    """Try to reduce rank by finding linear dependencies."""
    n, m, p = DIMENSIONS
    current_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = current_rank
    
    # Build the target tensor
    U3 = U.reshape(n, m, current_rank)
    V3 = V.reshape(m, p, current_rank)
    W3 = W.reshape(p, n, current_rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    
    # Try to find a lower rank decomposition using ALS
    for target_rank in range(current_rank - 1, 40, -1):
        if time.time() > deadline - 5:
            break
        
        success = False
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            rng = np.random.RandomState(42 + attempt)
            Ur = rng.randn(n * m, target_rank) * 0.1
            Vr = rng.randn(m * p, target_rank) * 0.1
            Wr = rng.randn(p * n, target_rank) * 0.1
            
            for it in range(30):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                M = np.einsum('jkr,kir->ijk', Vr.reshape(m, p, target_rank), Wr.reshape(p, n, target_rank))
                U_new = np.zeros((n * m, target_rank))
                for i in range(n):
                    for j in range(m):
                        Mij = M[i, j, :]
                        Tij = T[i, j, :]
                        try:
                            u, _, _, _ = np.linalg.lstsq(Mij, Tij, rcond=None)
                            U_new[i * m + j, :] = u
                        except:
                            pass
                Ur = U_new
                
                # Update V
                N = np.einsum('ijr,kir->jki', Ur.reshape(n, m, target_rank), Wr.reshape(p, n, target_rank))
                V_new = np.zeros((m * p, target_rank))
                for j in range(m):
                    for k in range(p):
                        Njk = N[j, k, :]
                        Tjk = T[:, j, k]
                        try:
                            v, _, _, _ = np.linalg.lstsq(Njk, Tjk, rcond=None)
                            V_new[j * p + k, :] = v
                        except:
                            pass
                Vr = V_new
                
                # Update W
                P = np.einsum('ijr,jkr->kij', Ur.reshape(n, m, target_rank), Vr.reshape(m, p, target_rank))
                W_new = np.zeros((p * n, target_rank))
                for k in range(p):
                    for i in range(n):
                        Pki = P[k, i, :]
                        Tki = T[i, :, k]
                        try:
                            w, _, _, _ = np.linalg.lstsq(Pki, Tki, rcond=None)
                            W_new[k * n + i, :] = w
                        except:
                            pass
                Wr = W_new
                
                # Check error
                U3_t = Ur.reshape(n, m, target_rank)
                V3_t = Vr.reshape(m, p, target_rank)
                W3_t = Wr.reshape(p, n, target_rank)
                T_recon = np.einsum('ijr,jkr,kir->ijk', U3_t, V3_t, W3_t)
                err = np.max(np.abs(T_recon - T))
                
                if err < 1e-10:
                    success = True
                    break
            
            if success:
                # Round to half-integers
                Ur_r = np.round(Ur * 2) / 2
                Vr_r = np.round(Vr * 2) / 2
                Wr_r = np.round(Wr * 2) / 2
                err_rounded = _check_decomposition(Ur_r, Vr_r, Wr_r)
                if err_rounded < 1e-6:
                    best_U, best_V, best_W = Ur_r, Vr_r, Wr_r
                    best_rank = target_rank
                break
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Build initial rank-55 decomposition
    U, V, W = _build_rank55_decomp()
    
    err = _check_decomposition(U, V, W)
    if err > 1e-6:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    if time.time() < deadline - 10:
        U2, V2, W2, new_rank = _try_reduce_rank(U, V, W, deadline)
        if new_rank < U.shape[1]:
            U, V, W = U2, V2, W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
