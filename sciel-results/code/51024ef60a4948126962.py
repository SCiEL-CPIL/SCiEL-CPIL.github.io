import numpy as np
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0

    nm, mp, pn = n * m, m * p, p * n

    # Build target tensor
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def check_decomposition(U, V, W, tol=1e-8):
        T_full = np.einsum('ur,vr,wr->uvw', U, V, W)
        return np.allclose(T_full, T_target, atol=tol)

    def try_rank_peeling(T_res, max_rank, max_iter=50, n_restarts=5, seed=42):
        """
        Peeling approach with restarts.
        """
        best_result = None
        best_rank = max_rank + 1

        for restart in range(n_restarts):
            if time.time() - start_time > time_limit - 10:
                break
                
            np.random.seed(seed + restart)
            residual = T_res.copy()
            U_list = []
            V_list = []
            W_list = []

            for rank_idx in range(max_rank):
                if time.time() - start_time > time_limit - 10:
                    break

                # Initialize u, v, w
                u = np.random.randn(nm)
                v = np.random.randn(mp)
                w = np.random.randn(pn)
                
                # Normalize
                u /= (np.linalg.norm(u) + 1e-15)
                v /= (np.linalg.norm(v) + 1e-15)
                w /= (np.linalg.norm(w) + 1e-15)

                prev_norm = np.inf
                for it in range(max_iter):
                    # Update u
                    S_u = np.einsum('ijk,j,k->i', residual, v, w)
                    u_new = S_u / (np.linalg.norm(S_u) + 1e-15)

                    # Update v
                    S_v = np.einsum('ijk,i,k->j', residual, u_new, w)
                    v_new = S_v / (np.linalg.norm(S_v) + 1e-15)

                    # Update w
                    S_w = np.einsum('ijk,i,j->k', residual, u_new, v_new)
                    w_new = S_w / (np.linalg.norm(S_w) + 1e-15)

                    # Check convergence
                    cur_norm = np.linalg.norm(u_new - u) + np.linalg.norm(v_new - v) + np.linalg.norm(w_new - w)
                    u, v, w = u_new, v_new, w_new

                    if cur_norm < 1e-10:
                        break

                alpha = np.einsum('ijk,i,j,k', residual, u, v, w)
                if abs(alpha) < 1e-12:
                    break

                residual = residual - alpha * np.einsum('i,j,k->ijk', u, v, w)
                U_list.append(alpha * u)
                V_list.append(v)
                W_list.append(w)

                if np.linalg.norm(residual) < 1e-10:
                    break

            R = len(U_list)
            if R <= max_rank:
                U = np.column_stack(U_list)
                V = np.column_stack(V_list)
                W = np.column_stack(W_list)
                if check_decomposition(U, V, W, tol=1e-6):
                    if R < best_rank:
                        best_rank = R
                        best_result = (U_list, V_list, W_list)
                        if R == 1: # Unlikely for this size
                            return best_result

        return best_result

    # Split T into T1 (k=0,1,2) and T2 (k=3,4,5,6)
    T1 = np.zeros_like(T_target)
    T2 = np.zeros_like(T_target)
    
    # Fill T1 and T2 based on k index
    # w = k * n + i. So k = w // n.
    # v = j * p + k. So k = v % p.
    # We can just loop and assign.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                if k < 3:
                    T1[u, v, w] = 1.0
                else:
                    T2[u, v, w] = 1.0

    # Try to find decomposition for T1 (rank ~15)
    result1 = try_rank_peeling(T1, 16, max_iter=40, n_restarts=3, seed=100)
    rank1 = 16 # Fallback
    if result1:
        rank1 = len(result1[0])

    # Try to find decomposition for T2 (rank ~17-19)
    # Try ranks 17, 18, 19
    result2 = None
    rank2 = 20 # Fallback
    for target_rank in [17, 18, 19, 20]:
        if time.time() - start_time > time_limit - 20:
            break
        res = try_rank_peeling(T2, target_rank, max_iter=40, n_restarts=3, seed=200 + target_rank)
        if res:
            R = len(res[0])
            if R <= target_rank:
                result2 = res
                rank2 = R
                break

    if result1 and result2:
        U1, V1, W1 = result1
        U2, V2, W2 = result2
        
        U = np.column_stack([np.column_stack(U1), np.column_stack(U2)])
        V = np.column_stack([np.column_stack(V1), np.column_stack(V2)])
        W = np.column_stack([np.column_stack(W1), np.column_stack(W2)])
        
        if check_decomposition(U, V, W, tol=1e-6):
            total_rank = U.shape[1]
            if total_rank < n * m * p:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to full tensor search for 41, 40
    for target_rank in [41, 40]:
        if time.time() - start_time > time_limit - 15:
            break
        result = try_rank_peeling(T_target, target_rank, max_iter=30, n_restarts=2, seed=300 + target_rank)
        if result:
            U_list, V_list, W_list = result
            U = np.column_stack(U_list)
            V = np.column_stack(V_list)
            W = np.column_stack(W_list)
            if check_decomposition(U, V, W, tol=1e-6):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return get_schoolbook()
