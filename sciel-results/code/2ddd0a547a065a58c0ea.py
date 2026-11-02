import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 2, 3]

def construct():
    n, m, p = DIMENSIONS
    U_size = n * m  # 4
    V_size = m * p  # 6
    W_size = p * n  # 6
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Build target tensor: T[u, v, w] = 1 if u=i*m+k, v=k*p+j, w=j*n+i for some valid (i,k,j)
    # For 2x2 * 2x3:
    # i in {0,1}, k in {0,1}, j in {0,1,2}
    # u = i*2 + k, v = k*3 + j, w = j*2 + i
    nonzero_set = set()
    for k in range(m):
        for i in range(n):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                nonzero_set.add((u_idx, v_idx, w_idx))
    
    target_arr = np.zeros(U_size * V_size * W_size)
    for (u, v, w) in nonzero_set:
        target_arr[(u * V_size + v) * W_size + w] = 1.0

    def compute_tensor(U, V, W, rank):
        """Compute the tensor from factor matrices."""
        T = np.zeros((U_size, V_size, W_size))
        for r in range(rank):
            T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return T

    def verify_solution(U, V, W, rank, tol=1e-9):
        """Verify the decomposition exactly matches the target tensor."""
        T = compute_tensor(U, V, W, rank)
        return np.all(np.abs(T.flatten() - target_arr) < tol)

    def round_and_verify(U, V, W, rank, tol=1e-9):
        """Round to half-integers and verify."""
        U_r = np.round(U * 2) / 2.0
        V_r = np.round(V * 2) / 2.0
        W_r = np.round(W * 2) / 2.0
        if verify_solution(U_r, V_r, W_r, rank, tol):
            return True, U_r, V_r, W_r
        return False, None, None

    # Rank 11 fallback solution (Strassen for 2x2 block + schoolbook for 3rd column)
    rank_fallback = 11
    U_fb = np.zeros((U_size, rank_fallback))
    V_fb = np.zeros((V_size, rank_fallback))
    W_fb = np.zeros((W_size, rank_fallback))
    
    # Strassen part for columns 0 and 1 of B
    U_fb[0, 0] = 1; U_fb[3, 0] = 1
    V_fb[0, 0] = 1; V_fb[4, 0] = 1
    U_fb[2, 1] = 1; U_fb[3, 1] = 1
    V_fb[0, 1] = 1
    U_fb[0, 2] = 1
    V_fb[1, 2] = 1; V_fb[4, 2] = -1
    U_fb[3, 3] = 1
    V_fb[3, 3] = 1; V_fb[0, 3] = -1
    U_fb[0, 4] = 1; U_fb[1, 4] = 1
    V_fb[4, 4] = 1
    U_fb[2, 5] = 1; U_fb[0, 5] = -1
    V_fb[0, 5] = 1; V_fb[1, 5] = 1
    U_fb[1, 6] = 1; U_fb[3, 6] = -1
    V_fb[3, 6] = 1; V_fb[4, 6] = 1
    
    W_fb[0, 0] = 1; W_fb[0, 3] = 1; W_fb[0, 4] = -1; W_fb[0, 6] = 1
    W_fb[2, 2] = 1; W_fb[2, 4] = 1
    W_fb[1, 1] = 1; W_fb[1, 3] = 1
    W_fb[3, 0] = 1; W_fb[3, 1] = -1; W_fb[3, 2] = 1; W_fb[3, 5] = 1
    
    # Schoolbook part for column 2 of B
    U_fb[0, 7] = 1
    V_fb[2, 7] = 1
    U_fb[1, 8] = 1
    V_fb[5, 8] = 1
    U_fb[2, 9] = 1
    V_fb[2, 9] = 1
    U_fb[3, 10] = 1
    V_fb[5, 10] = 1
    
    W_fb[4, 7] = 1; W_fb[4, 8] = 1
    W_fb[5, 9] = 1; W_fb[5, 10] = 1

    # Verify the fallback
    if not verify_solution(U_fb, V_fb, W_fb, rank_fallback):
        # Fallback broken; use schoolbook rank 12
        rank_sb = 12
        U_sb = np.zeros((U_size, rank_sb))
        V_sb = np.zeros((V_size, rank_sb))
        W_sb = np.zeros((W_size, rank_sb))
        for k in range(m):
            for i in range(n):
                for j in range(p):
                    r = k * n * p + i * p + j
                    U_sb[i * m + k, r] = 1
                    V_sb[k * p + j, r] = 1
                    W_sb[j * n + i, r] = 1
        return {"U": [[float(x) for x in row] for row in U_sb],
                "V": [[float(x) for x in row] for row in V_sb],
                "W": [[float(x) for x in row] for row in W_sb]}

    # Try to find rank 10
    rank = 10
    best_result = None

    def objective(p, rank):
        """Objective function for optimization: sum of squared errors."""
        U_f = p[:U_size * rank].reshape(U_size, rank)
        V_f = p[U_size * rank : U_size * rank + V_size * rank].reshape(V_size, rank)
        W_f = p[U_size * rank + V_size * rank :].reshape(W_size, rank)
        T = np.zeros((U_size, V_size, W_size))
        for r in range(rank):
            T += np.einsum('i,j,k->ijk', U_f[:, r], V_f[:, r], W_f[:, r])
        return np.sum((T.flatten() - target_arr) ** 2)

    # Strategy 1: Drop one term from rank-11 and optimize
    for drop_col in range(rank_fallback):
        if time.time() > deadline:
            break
        
        U_init = np.zeros((U_size, rank))
        V_init = np.zeros((V_size, rank))
        W_init = np.zeros((W_size, rank))
        
        for c in range(rank):
            src = c if c < drop_col else c + 1
            U_init[:, c] = U_fb[:, src]
            V_init[:, c] = V_fb[:, src]
            W_init[:, c] = W_fb[:, src]
        
        p_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        res = minimize(objective, p_init, args=(rank,), method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})
        
        if res.fun < 1e-10:
            U_opt = res.x[:U_size * rank].reshape(U_size, rank)
            V_opt = res.x[U_size * rank : U_size * rank + V_size * rank].reshape(V_size, rank)
            W_opt = res.x[U_size * rank + V_size * rank :].reshape(W_size, rank)
            found, U_r, V_r, W_r = round_and_verify(U_opt, V_opt, W_opt, rank)
            if found:
                best_result = (U_r, V_r, W_r)
                break
    
    # Strategy 2: Random restarts with perturbation from rank-11
    if best_result is None:
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            drop_col = random.randint(0, rank_fallback - 1)
            U_init = np.zeros((U_size, rank))
            V_init = np.zeros((V_size, rank))
            W_init = np.zeros((W_size, rank))
            
            for c in range(rank):
                src = c if c < drop_col else c + 1
                U_init[:, c] = U_fb[:, src]
                V_init[:, c] = V_fb[:, src]
                W_init[:, c] = W_fb[:, src]
            
            noise_scale = 0.1
            U_init += noise_scale * np.random.randn(U_size, rank)
            V_init += noise_scale * np.random.randn(V_size, rank)
            W_init += noise_scale * np.random.randn(W_size, rank)
            
            p_init = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            res = minimize(objective, p_init, args=(rank,), method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
            
            if res.fun < 1e-10:
                U_opt = res.x[:U_size * rank].reshape(U_size, rank)
                V_opt = res.x[U_size * rank : U_size * rank + V_size * rank].reshape(V_size, rank)
                W_opt = res.x[U_size * rank + V_size * rank :].reshape(W_size, rank)
                found, U_r, V_r, W_r = round_and_verify(U_opt, V_opt, W_opt, rank)
                if found:
                    best_result = (U_r, V_r, W_r)
                    break

    if best_result is not None:
        U_r, V_r, W_r = best_result
        return {"U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]}
    else:
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
