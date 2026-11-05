import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    neighbors_u = [[] for _ in range(n*m)]
    neighbors_v = [[] for _ in range(m*p)]
    neighbors_w = [[] for _ in range(p*n)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                neighbors_u[u].append((v, w))
                neighbors_v[v].append((u, w))
                neighbors_w[w].append((u, v))
                
    def check_exact(U, V, W):
        T_rec = np.zeros((n*m, m*p, p*n))
        for c in range(U.shape[1]):
            T_rec += np.einsum('u,v,w->uvw', U[:, c], V[:, c], W[:, c])
            
        T_target = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m + j
                    v = j*p + k
                    w = k*n + i
                    T_target[u, v, w] = 1.0
                    
        err = np.linalg.norm(T_rec - T_target)
        return err

    def snap_to_half_integer(mat):
        return np.round(mat * 2) / 2.0

    def try_rank(r, time_limit):
        max_restarts = 100
        max_iters = 100
        
        for restart in range(max_restarts):
            if time.time() > time_limit:
                break
                
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                    
                # Update U
                Num_U = np.zeros((n*m, r))
                Den_U = np.zeros((n*m, r))
                for u in range(n*m):
                    for v, w in neighbors_u[u]:
                        Num_U[u] += V[v] * W[w]
                        Den_U[u] += (V[v] * W[w]) ** 2
                mask = Den_U > 1e-12
                U_new = np.zeros_like(U)
                U_new[mask] = Num_U[mask] / Den_U[mask]
                U = U_new
                
                # Update V
                Num_V = np.zeros((m*p, r))
                Den_V = np.zeros((m*p, r))
                for v in range(m*p):
                    for u, w in neighbors_v[v]:
                        Num_V[v] += U[u] * W[w]
                        Den_V[v] += (U[u] * W[w]) ** 2
                mask = Den_V > 1e-12
                V_new = np.zeros_like(V)
                V_new[mask] = Num_V[mask] / Den_V[mask]
                V = V_new
                
                # Update W
                Num_W = np.zeros((p*n, r))
                Den_W = np.zeros((p*n, r))
                for w in range(p*n):
                    for u, v in neighbors_w[w]:
                        Num_W[w] += U[u] * V[v]
                        Den_W[w] += (U[u] * V[v]) ** 2
                mask = Den_W > 1e-12
                W_new = np.zeros_like(W)
                W_new[mask] = Num_W[mask] / Den_W[mask]
                W = W_new
                
            err = check_exact(U, V, W)
            if err < 1e-9:
                return U, V, W
                
            U_snap = snap_to_half_integer(U)
            V_snap = snap_to_half_integer(V)
            W_snap = snap_to_half_integer(W)
            err_snap = check_exact(U_snap, V_snap, W_snap)
            if err_snap < 1e-9:
                return U_snap, V_snap, W_snap

        return None

    # Allocate time for each rank
    # Start with 18, then 19, etc.
    ranks_to_try = [18, 19, 20, 21, 22, 23, 24]
    time_per_rank = 15  # seconds
    
    for r in ranks_to_try:
        current_time = time.time()
        if current_time > deadline:
            break
        limit = min(deadline - 5, current_time + time_per_rank)
        res = try_rank(r, limit)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback
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
