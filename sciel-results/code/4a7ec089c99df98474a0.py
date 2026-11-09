import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 45
    
    # Construct the target tensor T of shape (n*m, m*p, p*n) = (9, 15, 15)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    T_flat = T.reshape(n * m, m * p * p * n)  # (9, 225)
    
    def als_step_U(V, W, r):
        """Solve for U given V, W. Returns U of shape (9, r)."""
        # A[r, b*p*n + c] = V[b,r] * W[c,r]
        A = np.einsum('br,cr->bcr', V, W).reshape(r, -1)  # (r, 225)
        AA = A @ A.T  # (r, r)
        AT = A @ T_flat.T  # (r, 9)
        try:
            U_T = np.linalg.solve(AA, AT)
        except np.linalg.LinAlgError:
            U_T = np.linalg.lstsq(AA, AT, rcond=None)[0]
        return U_T.T  # (9, r)
    
    def als_step_V(U, W, r):
        """Solve for V given U, W. Returns V of shape (15, r)."""
        T3 = T_flat.reshape(n * m, m * p, p * n)
        V_new = np.zeros((m * p, r))
        for b in range(m * p):
            # N_b[r, a*(p*n) + c] = U[a,r] * W[c,r]
            N_b = np.einsum('ar,cr->rac', U, W).reshape(r, -1)  # (r, 135)
            target = T3[:, b, :].reshape(-1)  # (135,)
            NN = N_b @ N_b.T  # (r, r)
            NT = N_b @ target  # (r,)
            try:
                V_b = np.linalg.solve(NN, NT)
            except np.linalg.LinAlgError:
                V_b = np.linalg.lstsq(NN, NT, rcond=None)[0]
            V_new[b, :] = V_b
        return V_new
    
    def als_step_W(U, V, r):
        """Solve for W given U, V. Returns W of shape (15, r)."""
        T3 = T_flat.reshape(n * m, m * p, p * n)
        W_new = np.zeros((p * n, r))
        for c in range(p * n):
            # P_c[r, a*(m*p) + b] = U[a,r] * V[b,r]
            P_c = np.einsum('ar,br->rab', U, V).reshape(r, -1)  # (r, 135)
            target = T3[:, :, c].reshape(-1)  # (135,)
            PP = P_c @ P_c.T  # (r, r)
            PT = P_c @ target  # (r,)
            try:
                W_c = np.linalg.solve(PP, PT)
            except np.linalg.LinAlgError:
                W_c = np.linalg.lstsq(PP, PT, rcond=None)[0]
            W_new[c, :] = W_c
        return W_new
    
    def compute_error(U, V, W):
        R = np.einsum('ak,bk,ck->abc', U, V, W)
        return np.linalg.norm(R - T)
    
    # Schoolbook initialization
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook
    start_time = time.time()
    
    # Try ranks from 44 down to 28
    for r in range(rank_schoolbook - 1, 27, -1):
        if time.time() - start_time > 130:
            break
        
        found = False
        for restart in range(15):
            if time.time() - start_time > 135:
                break
            
            if restart == 0:
                U = U_school[:, :r].copy()
                V = V_school[:, :r].copy()
                W = W_school[:, :r].copy()
            elif restart == 1:
                U = U_school[:, :r].copy()
                V = V_school[:, :r].copy()
                W = W_school[:, :r].copy()
                U += 0.05 * np.random.randn(*U.shape)
                V += 0.05 * np.random.randn(*V.shape)
                W += 0.05 * np.random.randn(*W.shape)
            elif restart == 2:
                U = U_school[:, :r].copy()
                V = V_school[:, :r].copy()
                W = W_school[:, :r].copy()
                U += 0.2 * np.random.randn(*U.shape)
                V += 0.2 * np.random.randn(*V.shape)
                W += 0.2 * np.random.randn(*W.shape)
            else:
                U = np.random.randn(n * m, r)
                V = np.random.randn(m * p, r)
                W = np.random.randn(p * n, r)
            
            # ALS iterations
            for it in range(100):
                if time.time() - start_time > 135:
                    break
                U = als_step_U(V, W, r)
                V = als_step_V(U, W, r)
                W = als_step_W(U, V, r)
            
            err = compute_error(U, V, W)
            if err < 1e-8:
                best_U, best_V, best_W = U, V, W
                best_rank = r
                found = True
                break
        
        if found:
            pass
    
    # Final polish if we found a solution
    if best_rank < rank_schoolbook:
        r = best_rank
        for it in range(300):
            if time.time() - start_time > 140:
                break
            best_U = als_step_U(best_V, best_W, r)
            best_V = als_step_V(best_U, best_W, r)
            best_W = als_step_W(best_U, best_V, r)
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
