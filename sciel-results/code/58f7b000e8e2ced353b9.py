import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit

    # Schoolbook decomposition
    rank_school = n * m * p  # 63
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_rank = rank_school
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()

    # Efficient error calculation using vectorized operations
    def compute_error_and_grads(U, V, W, rank):
        # Precompute slices
        # U[i*m:(i+1)*m, :] for i in range(n)
        # V[j*p+k, :] for j in range(m), k in range(p)
        # W[k*n+i, :] for k in range(p), i in range(n)
        
        U_slices = [U[i*m:(i+1)*m, :] for i in range(n)]  # list of (m, rank)
        V_slices = []  # V_slices[j][k] = V[j*p+k, :] shape (rank,)
        for j in range(m):
            V_slices.append([V[j*p+k, :] for k in range(p)])
        W_slices = [W[k*n:(k*n+n), :] for k in range(p)]  # W_slices[k][i] = W[k*n+i, :]
        
        errors = np.zeros((n, p, m, m))
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            U_i = U_slices[i]  # (m, rank)
            for k in range(p):
                w_val = W_slices[k][i]  # (rank,)
                V_k = np.array([V_slices[j][k] for j in range(m)])  # (m, rank)
                
                # result[j1,j2] = sum_c U_i[j1,c] * V_k[j2,c] * w_val[c]
                Uw = U_i * w_val  # (m, rank)
                result = Uw @ V_k.T  # (m, m)
                err_ik = result - np.eye(m)
                errors[i, k] = err_ik
                
                # Gradients
                # dU_i[j1,c] = sum_j2 V_k[j2,c] * w_val[c] * err_ik[j1,j2]
                # = (err_ik @ V_k) * w_val
                dU_i = (err_ik @ V_k) * w_val  # (m, rank)
                dU[i*m:(i+1)*m, :] = dU_i
                
                # dV_k[j2,c] = sum_j1 U_i[j1,c] * w_val[c] * err_ik[j1,j2]
                # = (U_i * w_val).T @ err_ik ... let me think
                # dV[j2,c] = sum_j1 U_i[j1,c] * w_val[c] * err_ik[j1,j2]
                # = (w_val * U_i.T @ err_ik)[j2] ... 
                # Actually: dV_k[j2,c] = sum_j1 U_i[j1,c] * w_val[c] * err_ik[j1,j2]
                # = w_val[c] * sum_j1 U_i[j1,c] * err_ik[j1,j2]
                # Let M[j1,c] = U_i[j1,c], then dV_k[j2,c] = w_val[c] * (M.T @ err_ik)[j2]... no
                # dV_k[j2,c] = sum_j1 M[j1,c] * err_ik[j1,j2] * w_val[c]
                # = w_val[c] * (M.T @ err_ik)[c,j2]? No, M is (m,rank), M.T is (rank,m)
                # sum_j1 M[j1,c] * err_ik[j1,j2] = (M.T @ err_ik)[c,j2]... 
                # Actually M.T has shape (rank, m), err_ik has shape (m, m)
                # M.T @ err_ik has shape (rank, m), and [c,j2] = sum_j1 M[j1,c]*err_ik[j1,j2]
                # So dV_k[j2,c] = w_val[c] * (M.T @ err_ik)[c,j2]
                # dV_k = (w_val[:,None] * (U_i.T @ err_ik).T)  # (m, rank)
                dV_k = (w_val[:, None] * (U_i.T @ err_ik).T)  # (m, rank)
                for j in range(m):
                    dV[j*p+k, :] = dV_k[j]
                
                # dW[k*n+i,c] = sum_j1,j2 U_i[j1,c] * V_k[j2,c] * err_ik[j1,j2]
                # = sum_j1,j2 U_i[j1,c] * V_k[j2,c] * err_ik[j1,j2]
                # = (U_i * err_ik.sum(axis=0, keepdims=False)) ... no
                # Let's compute: for each c, dW[c] = sum_j1,j2 U_i[j1,c]*V_k[j2,c]*err_ik[j1,j2]
                # = sum_j1 U_i[j1,c] * (sum_j2 V_k[j2,c]*err_ik[j1,j2])
                # = sum_j1 U_i[j1,c] * (V_k.T @ err_ik.T)[j2,j1]... 
                # Actually: sum_j2 V_k[j2,c]*err_ik[j1,j2] = (V_k.T @ err_ik.T)[c,j1]? 
                # V_k.T is (rank,m), err_ik.T is (m,m), product is (rank,m), [c,j1] = sum_j2 V_k[j2,c]*err_ik[j2,j1]... no
                # Let me just compute directly:
                # dW_val[c] = sum_j1,j2 U_i[j1,c]*V_k[j2,c]*err_ik[j1,j2]
                # = sum_j1 U_i[j1,c] * (sum_j2 V_k[j2,c]*err_ik[j1,j2])
                # inner = sum_j2 V_k[j2,c]*err_ik[j1,j2] for each j1,c
                # = (V_k.T @ err_ik.T) ... V_k.T is (rank,m), err_ik.T is (m,m)
                # V_k.T @ err_ik.T has shape (rank,m), entry [c,j1] = sum_j2 V_k[j2,c]*err_ik.T[j2,j1] = sum_j2 V_k[j2,c]*err_ik[j1,j2]
                # So inner[c,j1] = sum_j2 V_k[j2,c]*err_ik[j1,j2]
                # Then dW_val[c] = sum_j1 U_i[j1,c] * inner[c,j1] = sum_j1 inner[c,j1]*U_i[j1,c]
                # = (inner * U_i.T).sum(axis=1) where inner is (rank,m), U_i.T is (rank,m)
                inner = V_k.T @ err_ik.T  # (rank, m)
                dW_val = (inner * U_i.T).sum(axis=1)  # (rank,)
                dW[k*n+i, :] = dW_val
        
        return errors, dU, dV, dW

    def try_rank(target_rank, max_iterations=200, n_restarts=3):
        nonlocal best_rank, best_U, best_V, best_W
        
        if target_rank >= best_rank:
            return False
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                return False
            
            # Initialize
            np.random.seed(restart * 1000 + target_rank)
            
            if target_rank <= rank_school:
                U = U_school[:, :target_rank].copy()
                V = V_school[:, :target_rank].copy()
                W = W_school[:, :target_rank].copy()
            else:
                U = np.random.randn(n * m, target_rank) * 0.5
                V = np.random.randn(m * p, target_rank) * 0.5
                W = np.random.randn(p * n, target_rank) * 0.5
            
            # Add noise
            U += np.random.randn(n * m, target_rank) * 0.3
            V += np.random.randn(m * p, target_rank) * 0.3
            W += np.random.randn(p * n, target_rank) * 0.3
            
            # Gradient descent
            lr = 0.05
            for it in range(max_iterations):
                if time.time() > deadline - 10:
                    return False
                
                errors, dU, dV, dW = compute_error_and_grads(U, V, W, target_rank)
                err_norm = np.linalg.norm(errors)
                
                if err_norm < 1e-8:
                    best_rank = target_rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    return True
                
                # Update with momentum-like decay
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Decay learning rate
                lr *= 0.99
        
        return False

    # Try ranks in decreasing order from schoolbook-1 down to a reasonable bound
    for target_rank in range(rank_school - 1, 45, -1):
        if time.time() > deadline - 15:
            break
        if try_rank(target_rank, max_iterations=150, n_restarts=2):
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
