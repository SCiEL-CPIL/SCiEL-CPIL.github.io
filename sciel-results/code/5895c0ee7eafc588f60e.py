import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    nm, mp, pn = n * m, m * p, p * n

    # Build the target tensor
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def to_list(mat):
        return [[float(x) for x in row] for row in mat]

    def verify_solution(U, V, W, T, tol=1e-9):
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, T, atol=tol)

    def is_half_integer(mat, tol=1e-6):
        scaled = 2.0 * mat
        rounded = np.round(scaled)
        if np.all(np.abs(scaled - rounded) < tol):
            return True
        return False

    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()

    start_time = time.time()
    deadline = start_time + 135

    # Strategy 1: Find rank-16 decomposition for 2x3x3 and construct rank-48 for 2x3x9
    # This is the most likely improvement.
    
    def find_decomposition_233(deadline_inner):
        n_d, m_d, p_d = 2, 3, 3
        nm_d, mp_d, pn_d = n_d * m_d, m_d * p_d, p_d * n_d
        T_d = np.zeros((nm_d, mp_d, pn_d))
        for i in range(n_d):
            for j in range(m_d):
                for k in range(p_d):
                    T_d[i*m_d+j, j*p_d+k, k*n_d+i] = 1.0
        
        # Known rank is 16. We search for it.
        # We use random V, W and solve for U.
        for r in range(16, 17): # Only try 16
            for trial in range(200):
                if time.time() > deadline_inner:
                    return None
                np.random.seed(42 + r * 1000 + trial)
                V = np.random.randint(-2, 3, size=(mp_d, r))
                W = np.random.randint(-2, 3, size=(pn_d, r))
                
                M = (V[:, None, :] * W[None, :, :]).reshape(mp_d * pn_d, r)
                
                # Check rank of M
                if np.linalg.matrix_rank(M) < r:
                    continue
                    
                U = np.zeros((nm_d, r))
                valid = True
                for a in range(nm_d):
                    target = T_d[a, :, :].flatten()
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M, target, rcond=None)
                        if residuals and residuals[0] > 1e-6:
                            valid = False
                            break
                        U[a, :] = sol
                    except:
                        valid = False
                        break
                        
                if not valid:
                    continue
                
                # Check half-integer constraint
                if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                    U_snapped = np.round(2.0 * U) / 2.0
                    V_snapped = np.round(2.0 * V) / 2.0
                    W_snapped = np.round(2.0 * W) / 2.0
                    if verify_solution(U_snapped, V_snapped, W_snapped, T_d):
                        return U_snapped, V_snapped, W_snapped, r
        return None

    # Try to find 2x3x3 decomposition
    deadline_233 = time.time() + 30
    res_233 = find_decomposition_233(deadline_233)
    
    if res_233 is not None:
        U_233, V_233, W_233, r_233 = res_233
        new_rank = 3 * r_233
        if new_rank < best_rank:
            U_new = np.zeros((nm, new_rank))
            V_new = np.zeros((mp, new_rank))
            W_new = np.zeros((pn, new_rank))
            
            for block in range(3):
                for r_local in range(r_233):
                    col_global = block * r_233 + r_local
                    U_new[:, col_global] = U_233[:, r_local]
                    
                    for j in range(3):
                        for k_local in range(3):
                            v_idx_local = j * 3 + k_local
                            v_idx_global = j * 9 + (block * 3 + k_local)
                            V_new[v_idx_global, col_global] = V_233[v_idx_local, r_local]
                    
                    for k_local in range(3):
                        for i in range(2):
                            w_idx_local = k_local * 2 + i
                            w_idx_global = (block * 3 + k_local) * 2 + i
                            W_new[w_idx_global, col_global] = W_233[w_idx_local, r_local]
                
            if verify_solution(U_new, V_new, W_new, T):
                best_rank = new_rank
                best_U, best_V, best_W = U_new, V_new, W_new
                print(f"Found decomposition with rank {best_rank} from 2x3x3 blocks")

    # Strategy 2: Direct search for 2x3x9 if time remains and rank > 40
    if best_rank > 40 and time.time() < deadline - 10:
        for r in range(40, best_rank):
            if time.time() > deadline - 10:
                break
            found = False
            for trial in range(50):
                if time.time() > deadline - 10:
                    break
                np.random.seed(100 + r * 1000 + trial)
                V = np.random.randint(-2, 3, size=(mp, r))
                W = np.random.randint(-2, 3, size=(pn, r))
                
                M = (V[:, None, :] * W[None, :, :]).reshape(mp * pn, r)
                
                if np.linalg.matrix_rank(M) < r:
                    continue
                    
                U = np.zeros((nm, r))
                valid = True
                for a in range(nm):
                    target = T[a, :, :].flatten()
                    try:
                        sol, residuals, rank, sv = np.linalg.lstsq(M, target, rcond=None)
                        if residuals and residuals[0] > 1e-6:
                            valid = False
                            break
                        U[a, :] = sol
                    except:
                        valid = False
                        break
                        
                if not valid:
                    continue
                    
                if is_half_integer(U) and is_half_integer(V) and is_half_integer(W):
                    U_snapped = np.round(2.0 * U) / 2.0
                    V_snapped = np.round(2.0 * V) / 2.0
                    W_snapped = np.round(2.0 * W) / 2.0
                    if verify_solution(U_snapped, V_snapped, W_snapped, T):
                        best_rank = r
                        best_U, best_V, best_W = U_snapped, V_snapped, W_snapped
                        found = True
                        break
            if found:
                break

    U_list = to_list(best_U)
    V_list = to_list(best_V)
    W_list = to_list(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
