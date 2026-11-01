import numpy as np
import time

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
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0

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

    def als_decompose(rank, max_iter=100, tol=1e-10):
        # Initialize randomly
        U = np.random.randn(U_size, rank)
        V = np.random.randn(V_size, rank)
        W = np.random.randn(W_size, rank)
        
        prev_cost = np.inf
        for it in range(max_iter):
            # Update U
            # T_flat (U, V*W) = U_new (U, rank) @ M_T (rank, V*W)
            # M (V*W, rank) where M[(j,k), c] = V[j,c] * W[k,c]
            M = np.einsum('jc,kc->jkc', V, W).reshape(V_size * W_size, rank)
            # Solve M @ U_new.T = T_flat.T
            # T_flat is (U, V*W). T_flat.T is (V*W, U).
            # lstsq(M, T_flat.T) gives U_new.T
            try:
                U_new = np.linalg.lstsq(M, T.reshape(U_size, -1).T, rcond=None)[0].T
            except:
                return None

            # Update V
            # For each j, T[:, j, :] (U, W) = V_new[j] (rank) @ N_T (rank, U*W) ? No.
            # T[i, j, k] = sum_c V[j,c] * (U[i,c] * W[k,c])
            # Let N (U*W, rank) where N[(i,k), c] = U[i,c] * W[k,c]
            # T[:, j, :].T (U*W, 1) = N @ V_new[j].T (rank, 1)
            N = np.einsum('ic,kc->ikc', U_new, W).reshape(U_size * W_size, rank)
            V_new = np.zeros((V_size, rank))
            for j in range(V_size):
                t_j = T[:, j, :].reshape(-1, 1)
                try:
                    V_new[j] = np.linalg.lstsq(N, t_j, rcond=None)[0].flatten()
                except:
                    return None

            # Update W
            # For each k, T[:, :, k] (U*V, 1) = P @ W_new[k].T (rank, 1)
            # P (U*V, rank) where P[(i,j), c] = U[i,c] * V[j,c]
            P = np.einsum('ic,jc->ijc', U_new, V_new).reshape(U_size * V_size, rank)
            W_new = np.zeros((W_size, rank))
            for k in range(W_size):
                t_k = T[:, :, k].reshape(-1, 1)
                try:
                    W_new[k] = np.linalg.lstsq(P, t_k, rcond=None)[0].flatten()
                except:
                    return None

            U, V, W = U_new, V_new, W_new
            
            # Check convergence
            cost = 0
            for c in range(rank):
                cost += np.sum((np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c]) - T)**2)
            
            if abs(prev_cost - cost) < tol:
                break
            prev_cost = cost
                
        return U, V, W

    # Try ranks from 14 upwards
    # Prioritize 14 and 15 with more restarts, as they are the most likely improvements
    # Reduce max_iter to allow more restarts within the time limit
    rank_configs = [
        (14, 50, 30),
        (15, 50, 30),
        (16, 20, 30),
        (17, 10, 30)
    ]
    
    for rank, max_restarts, max_iter in rank_configs:
        if time.time() > deadline:
            break
            
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(restart * 1000 + rank)
            
            U, V, W = als_decompose(rank, max_iter=max_iter)
            if U is None:
                continue

            result = snap_and_verify(U, V, W)
            if result:
                U_final, V_final, W_final = result
                return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

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
