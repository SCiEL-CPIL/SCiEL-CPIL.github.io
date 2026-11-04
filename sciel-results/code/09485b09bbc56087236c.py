import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    T_A = T.reshape(n * m, m * p * p * n)
    T_B = T.transpose(1, 0, 2).reshape(m * p, n * m * p * n)
    T_C = T.transpose(2, 0, 1).reshape(p * n, n * m * m * p)

    def verify(U, V, W, atol=1e-9):
        T_computed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_computed, T, atol=atol)

    def round_to_half(x):
        return np.round(x * 2) / 2

    def try_solve_for_U(V, W, rank):
        M = np.einsum('br,cr->bcr', V, W)
        M_flat = M.reshape(m * p * p * n, rank)
        try:
            U = np.linalg.pinv(M_flat) @ T_A.T
        except np.linalg.LinAlgError:
            return None
        U_r = round_to_half(U)
        if verify(U_r, V, W):
            return U_r
        return None

    def try_solve_for_V(U, W, rank):
        M = np.einsum('ar,cr->acr', U, W)
        M_flat = M.reshape(n * m * p * n, rank)
        try:
            V = np.linalg.pinv(M_flat) @ T_B.T
        except np.linalg.LinAlgError:
            return None
        V_r = round_to_half(V)
        if verify(U, V_r, W):
            return V_r
        return None

    def try_solve_for_W(U, V, rank):
        M = np.einsum('ar,br->abr', U, V)
        M_flat = M.reshape(n * m * m * p, rank)
        try:
            W = np.linalg.pinv(M_flat) @ T_C.T
        except np.linalg.LinAlgError:
            return None
        W_r = round_to_half(W)
        if verify(U, V, W_r):
            return W_r
        return None

    def random_init_sparse(shape, max_val=3):
        # Heavily biased towards 0
        return np.random.choice([-max_val, -max_val+1, -1, 0, 0, 0, 1, max_val-1, max_val], size=shape).astype(np.float64)

    best_rank = n * m * p
    U_sch = np.zeros((n * m, best_rank), dtype=np.float64)
    V_sch = np.zeros((m * p, best_rank), dtype=np.float64)
    W_sch = np.zeros((p * n, best_rank), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U_sch[i * m + j, r] = 1.0
                V_sch[j * p + k, r] = 1.0
                W_sch[k * n + i, r] = 1.0
    best_U, best_V, best_W = U_sch, V_sch, W_sch

    # Search
    for target_rank in range(best_rank - 1, 55, -1):
        if time.time() > deadline:
            break
        
        found = False
        # Increase restarts
        restarts = 500
        for r in range(restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + target_rank * 1000 + r)
            
            # Strategy 1
            V_init = random_init_sparse((m * p, target_rank), 4)
            W_init = random_init_sparse((p * n, target_rank), 4)
            U_sol = try_solve_for_U(V_init, W_init, target_rank)
            if U_sol is not None:
                best_rank = target_rank
                best_U, best_V, best_W = U_sol, V_init, W_init
                found = True
                break
            
            # Strategy 2
            U_init = random_init_sparse((n * m, target_rank), 4)
            W_init2 = random_init_sparse((p * n, target_rank), 4)
            V_sol = try_solve_for_V(U_init, W_init2, target_rank)
            if V_sol is not None:
                best_rank = target_rank
                best_U, best_V, best_W = U_init, V_sol, W_init2
                found = True
                break

            # Strategy 3
            U_init2 = random_init_sparse((n * m, target_rank), 4)
            V_init2 = random_init_sparse((m * p, target_rank), 4)
            W_sol = try_solve_for_W(U_init2, V_init2, target_rank)
            if W_sol is not None:
                best_rank = target_rank
                best_U, best_V, best_W = U_init2, V_init2, W_sol
                found = True
                break

        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
