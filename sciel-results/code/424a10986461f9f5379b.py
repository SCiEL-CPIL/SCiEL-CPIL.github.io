"""Matrix multiplication tensor decomposition search for 3x4x6."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 4, 6]
START_TIME = time.time()
DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    global START_TIME
    start_time = time.time()
    n, m, p = DIMENSIONS
    N, M, P = n, m, p
    NM, MP, PN = N * M, M * P, P * N
    
    # Build target tensor T of shape (N*M, M*P, P*N)
    # T[u, v, w] = 1 if u=i*M+j, v=j*P+k, w=k*N+i
    T = np.zeros((NM, MP, PN))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                u = i * M + j
                v = j * P + k
                w = k * N + i
                T[u, v, w] = 1.0
    T_flat = T.flatten()
    
    # Schoolbook fallback
    def schoolbook():
        rank = N * M * P
        U = [[0] * rank for _ in range(NM)]
        V = [[0] * rank for _ in range(MP)]
        W = [[0] * rank for _ in range(PN)]
        col = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][col] = 1
                    V[j * P + k][col] = 1
                    W[k * N + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = N * M * P
    
    def check_exact(U_list, V_list, W_list):
        """Check if the decomposition exactly reconstructs T."""
        if not U_list:
            return False
        r = len(U_list[0])
        U = np.array(U_list)
        V = np.array(V_list)
        W = np.array(W_list)
        
        # Compute reconstructed tensor T_rec[u, v, w] = sum_c U[u, c] V[v, c] W[w, c]
        T_rec = np.einsum('uc,vc,wc->uvw', U, V, W)
        
        # Compare with the target tensor T
        return np.allclose(T_rec, T, atol=1e-9)
    
    def objective_and_grad(x, r):
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U + size_V]
        W_flat = x[size_U + size_V:]
        
        U = U_flat.reshape(N, M, r)
        V = V_flat.reshape(M, P, r)
        W = W_flat.reshape(P, N, r)
        
        T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        T_3d = np.ones((N, M, P))
        err = T_rec - T_3d
        loss = np.sum(err ** 2)
        
        grad_U = 2 * np.einsum('ijk,jkc,kic->ijc', err, V, W)
        grad_V = 2 * np.einsum('ijk,ijc,kic->jkc', err, U, W)
        grad_W = 2 * np.einsum('ijk,ijc,jkc->kic', err, U, V)
        
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return loss, grad

    def try_rank(r, num_seeds=10, max_iter=500):
        """Try to find a decomposition of rank r."""
        size_U = N * M * r
        size_V = M * P * r
        size_W = P * N * r
        total_size = size_U + size_V + size_W
        
        for seed in range(num_seeds):
            if time.time() - start_time > DEADLINE:
                return None
            
            # Initialization
            if seed == 0:
                # Use schoolbook initialization truncated to r
                sb = schoolbook()
                U_sb = np.array(sb["U"])
                V_sb = np.array(sb["V"])
                W_sb = np.array(sb["W"])
                
                U_init = U_sb[:, :r].flatten()
                V_init = V_sb[:, :r].flatten()
                W_init = W_sb[:, :r].flatten()
                x0 = np.concatenate([U_init, V_init, W_init])
            else:
                np.random.seed(seed * 1000 + r)
                # Random perturbation of schoolbook truncated
                sb = schoolbook()
                U_sb = np.array(sb["U"])
                V_sb = np.array(sb["V"])
                W_sb = np.array(sb["W"])
                
                U_init = U_sb[:, :r].flatten() + np.random.randn(size_U) * 0.5
                V_init = V_sb[:, :r].flatten() + np.random.randn(size_V) * 0.5
                W_init = W_sb[:, :r].flatten() + np.random.randn(size_W) * 0.5
                x0 = np.concatenate([U_init, V_init, W_init])
            
            try:
                res = minimize(
                    objective_and_grad,
                    x0,
                    args=(r,),
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )
            except Exception:
                continue
            
            if res.fun < 1e-4:
                # Extract and round to half-integers
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U:size_U + size_V]
                W_flat = res.x[size_U + size_V:]
                
                U = U_flat.reshape(NM, r)
                V = V_flat.reshape(MP, r)
                W = W_flat.reshape(PN, r)
                
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if check_exact(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # If close, try a local repair step
                # Calculate residual
                U_r = np.array(U_list)
                V_r = np.array(V_list)
                W_r = np.array(W_list)
                T_rec = np.einsum('uc,vc,wc->uvw', U_r, V_r, W_r)
                residual = T - T_rec
                
                # If residual is very small, maybe we just need to flip a few 0.5s to 1.0 or similar
                # This is a heuristic repair
                if np.linalg.norm(residual) < 1e-6:
                     # Try flipping signs or small integers
                     # This is complex, so we skip for now and rely on more seeds
                     pass
        
        return None
    
    # Try ranks from low to high, starting from just below schoolbook
    # Schoolbook is 72. Try 71, 70, ...
    for r in range(71, 60, -1):
        if time.time() - start_time > DEADLINE:
            break
        
        result = try_rank(r, num_seeds=10, max_iter=500)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # If no better found, return schoolbook
    if best_rank >= N * M * P:
        return schoolbook()
    
    return best_result
