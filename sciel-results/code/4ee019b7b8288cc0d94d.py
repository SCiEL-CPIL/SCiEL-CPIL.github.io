import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Create the target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                T[I, J, K] = 1.0
                
    def check_validity(U, V, W):
        U_check = np.round(U * 2) / 2
        V_check = np.round(V * 2) / 2
        W_check = np.round(W * 2) / 2
        
        if np.max(np.abs(U - U_check)) > 1e-9:
            return False
        if np.max(np.abs(V - V_check)) > 1e-9:
            return False
        if np.max(np.abs(W - W_check)) > 1e-9:
            return False
            
        T_recon_check = np.einsum('it,jt,kt->ijk', U_check, V_check, W_check, optimize=True)
        
        if np.max(np.abs(T_recon_check - T)) > 1e-9:
            return False
            
        return True

    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_schoolbook = np.zeros((nm, rank_schoolbook))
    V_schoolbook = np.zeros((mp, rank_schoolbook))
    W_schoolbook = np.zeros((pn, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I = i * m + j
                J = j * p + k
                K = k * n + i
                U_schoolbook[I, col] = 1
                V_schoolbook[J, col] = 1
                W_schoolbook[K, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_schoolbook, V_schoolbook, W_schoolbook
    best_rank = rank_schoolbook
    
    start_time = time.time()
    time_limit = 110.0 

    def run_als(rank, T, max_iters=50, tol=1e-12):
        U = np.random.randn(nm, rank)
        V = np.random.randn(mp, rank)
        W = np.random.randn(pn, rank)
        
        U /= np.linalg.norm(U)
        V /= np.linalg.norm(V)
        W /= np.linalg.norm(W)
        
        prev_error = np.inf
        for it in range(max_iters):
            M_VW = np.einsum('jt,kt->jkt', V, W)
            M_VW_flat = M_VW.reshape(mp * pn, rank)
            T_flat = T.reshape(nm, mp * pn)
            U = np.linalg.lstsq(M_VW_flat.T, T_flat.T, rcond=None)[0].T
            
            M_UW = np.einsum('it,kt->ikt', U, W)
            M_UW_flat = M_UW.reshape(nm * pn, rank)
            T_flat_V = T.reshape(nm, mp, pn).transpose(1, 0, 2).reshape(mp, nm * pn)
            V = np.linalg.lstsq(M_UW_flat.T, T_flat_V.T, rcond=None)[0].T
            
            M_UV = np.einsum('it,jt->ijt', U, V)
            M_UV_flat = M_UV.reshape(nm * mp, rank)
            T_flat_W = T.reshape(nm, mp, pn).transpose(2, 0, 1).reshape(pn, nm * mp)
            W = np.linalg.lstsq(M_UV_flat.T, T_flat_W.T, rcond=None)[0].T
            
            T_recon = np.einsum('it,jt,kt->ijk', U, V, W, optimize=True)
            error = np.linalg.norm(T_recon - T)
            
            if abs(prev_error - error) < tol:
                break
            prev_error = error
            
            if time.time() - start_time > time_limit:
                return U, V, W, error
                
        return U, V, W, error

    # Search for lower rank
    for rank in range(rank_schoolbook - 1, 50, -1):
        if time.time() - start_time > time_limit:
            break
            
        found = False
        for restart in range(10):
            if time.time() - start_time > time_limit:
                break
            U, V, W, error = run_als(rank, T, max_iters=20)
            if error < 1e-9:
                if check_validity(U, V, W):
                    best_U, best_V, best_W = U, V, W
                    best_rank = rank
                    found = True
                    break
                    
        if found:
            continue

    def clean_matrix(M):
        M_clean = []
        for row in M:
            new_row = []
            for val in row:
                rounded = round(val * 2) / 2
                if abs(rounded - round(rounded)) < 1e-9:
                    new_row.append(int(round(rounded)))
                else:
                    new_row.append(rounded)
            M_clean.append(new_row)
        return M_clean

    U_out = clean_matrix(best_U)
    V_out = clean_matrix(best_V)
    W_out = clean_matrix(best_W)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
