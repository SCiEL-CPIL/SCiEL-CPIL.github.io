import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115
    
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    T_target = np.ones((n, m, p))
    
    def check_decomposition(U, V, W, rank):
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        T_recon = np.einsum('ir,jr,kr->ijk', U_arr.reshape(n, m, rank), V_arr.reshape(m, p, rank), W_arr.reshape(p, n, rank))
        
        if np.allclose(T_recon, T_target, atol=1e-6):
            return True
        return False
        
    def round_and_check(U, V, W, rank):
        U_round = np.round(np.array(U) * 2) / 2
        V_round = np.round(np.array(V) * 2) / 2
        W_round = np.round(np.array(W) * 2) / 2
        
        if check_decomposition(U_round, V_round, W_round, rank):
            return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
        return None

    for rank in range(71, 39, -1):
        if time.time() > deadline:
            break
            
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            for it in range(50):
                # Solve for U
                # U[i, j, r] = argmin sum_k (T[i, j, k] - sum_r' U[i, j, r'] V[j, k, r'] W[k, i, r'])^2
                # This is a linear least squares problem for each (i, j).
                # M_U[i, j, k, r] = V[j, k, r] * W[k, i, r]
                # We want M_U[i, j, :, :] @ U[i, j, :] = T[i, j, :]
                
                # Construct M_U efficiently
                # V is (m, p, rank), W is (p, n, rank)
                # V[j, :, r] * W[:, i, r] -> (p, rank)
                # We need to do this for all i, j.
                
                # Reshape V to (1, m, p, rank)
                # Reshape W to (p, n, 1, rank)
                # Product: (1, m, p, rank) * (p, n, 1, rank) -> (1, m, p, n, rank)
                # This is too large.
                
                # Instead, loop over i, j.
                U_new = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # M is (p, rank)
                        # M[k, r] = V[j, k, r] * W[k, i, r]
                        M = V[j * p : (j+1) * p, :] * W[0::n, :].T # W is (p*n, rank), W[k*n+i, r]
                        # Wait, W is stored as (p*n, rank). Index k*n+i.
                        # So W[k, i] corresponds to row k*n+i.
                        # We need to extract W[k, i] for all k.
                        # This is W[0:n*p:n + i, :]
                        W_subset = W[range(0, p*n, n) + i, :] # This is not valid syntax for list of indices in numpy if i is scalar
                        # Better:
                        W_subset = W[::n, :] # This gives (p, rank) but with wrong i.
                        # We need W[k*n+i, r] for k=0..p-1.
                        # Indices: i, i+n, i+2n, ..., i+(p-1)n.
                        idx = np.arange(p) * n + i
                        W_subset = W[idx, :]
                        M = V[j * p : (j+1) * p, :] * W_subset
                        b = np.ones(p)
                        U_new[i * m + j, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                U = U_new
                
                V_new = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        idx_U = np.arange(n) * m + j
                        U_subset = U[idx_U, :]
                        idx_W = np.arange(p) * n + i # Wait, i is not fixed here.
                        # For V[j, k, r], we fix j, k.
                        # M[i, r] = U[i, j, r] * W[k, i, r]
                        # i ranges 0..n-1.
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        V_new[j * p + k, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                V = V_new
                
                W_new = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        W_new[k * n + i, :], _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                W = W_new
                
            if check_decomposition(U, V, W, rank):
                best_rank = rank
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                return best_result
                
            rounded = round_and_check(U, V, W, rank)
            if rounded:
                best_rank = rank
                best_result = rounded
                return best_result

    return best_result
