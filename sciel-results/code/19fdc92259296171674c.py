"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product


def _find_m223_rank10():
    """Find rank-10 decomposition for M_{2,2,3} via optimization."""
    n, m, p = 2, 2, 3
    rank = 10
    np.random.seed(42)
    
    # Initialize with random half-integer values
    U = np.random.randint(-2, 3, size=(n*m, rank)) * 0.5
    V = np.random.randint(-2, 3, size=(m*p, rank)) * 0.5
    W = np.random.randint(-2, 3, size=(p*n, rank)) * 0.5
    
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want T[i,j,k] = 1 if this is a valid multiplication entry, else 0
    # For matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 for all i,j,k (each term A[i,j]*B[j,k] contributes to C[i,k])
    # Actually the tensor is: T_{i,j,k} where the multiplication is sum_j A_{i,j} B_{j,k}
    # The tensor entries are: T[i,j,k] = delta (it's 1 for all i,j,k since each product term exists)
    
    # Wait, the tensor for matrix multiplication M_{n,m,p} has entries:
    # T[i,j,k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # So all entries are 1.
    
    # Actually no. The tensor decomposition means:
    # C[i,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] summed over j
    # The target tensor has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    def objective(params):
        U_p = params[:n*m*rank].reshape(n*m, rank)
        V_p = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
        W_p = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
        
        # Compute tensor: for each (i,j,k), compute sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        tensor_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_p[i*m+j] * V_p[j*p+k] * W_p[k*n+i])
                    tensor_err += (val - 1.0) ** 2
        return tensor_err
    
    # Use Nelder-Mead or similar
    from scipy.optimize import minimize
    
    params_init = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    best_obj = float('inf')
    best_params = params_init.copy()
    
    # Try multiple random initializations
    n_attempts = 50
    for attempt in range(n_attempts):
        np.random.seed(42 + attempt)
        U_r = np.random.randint(-3, 4, size=(n*m, rank)) * 0.5
        V_r = np.random.randint(-3, 4, size=(m*p, rank)) * 0.5
        W_r = np.random.randint(-3, 4, size=(p*n, rank)) * 0.5
        params_r = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
        
        res = minimize(objective, params_r, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x.copy()
        
        if best_obj < 1e-6:
            break
    
    U_final = best_params[:n*m*rank].reshape(n*m, rank)
    V_final = best_params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
    W_final = best_params[n*m*rank+m*p*rank:].reshape(p*n, rank)
    
    # Round to nearest half-integer
    U_half = np.round(U_final * 2) / 2
    V_half = np.round(V_final * 2) / 2
    W_half = np.round(W_final * 2) / 2
    
    # Verify
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_half[i*m+j] * V_half[j*p+k] * W_half[k*n+i])
                max_err = max(max_err, abs(val - 1.0))
    
    return U_half, V_half, W_half, max_err


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140

    # Fallback: Rank 53 solution
    rank_fallback = 53
    U_fb = [[0] * rank_fallback for _ in range(n * m)]
    V_fb = [[0] * rank_fallback for _ in range(m * p)]
    W_fb = [[0] * rank_fallback for _ in range(p * n)]

    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    U_school = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    V_school = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    W_school = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]

    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U_fb[i * m + j][c] = U_strassen[i * 2 + j][r]
            for j in range(2):
                V_fb[j * p + k0][c] = V_strassen[j * 2 + 0][r]
                V_fb[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W_fb[kk * n + i][c] = W_strassen[k * 2 + i][r]
        col_offset += 7

    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U_fb[i * m + j][c] = U_school[i * 2 + j][r]
        for j in range(2):
            V_fb[j * p + k_last][c] = V_school[j][r]
        for i in range(2):
            W_fb[k_last * n + i][c] = W_school[i][r]

    # Try to find rank-10 M_{2,2,3} and tile it for rank 50
    try:
        if time.time() < deadline - 60:
            U_223, V_223, W_223, err = _find_m223_rank10()
            if err < 1e-6:
                rank_223 = 10
                U_new = [[0] * rank_223 for _ in range(n * m)]
                V_new = [[0] * rank_223 for _ in range(m * p)]
                W_new = [[0] * rank_223 for _ in range(p * n)]
                
                n_blocks = p // 3  # 5 blocks of 3 columns
                for block in range(n_blocks):
                    k0 = 3 * block
                    for r in range(rank_223):
                        c = block * rank_223 + r
                        for i in range(n):
                            for j in range(m):
                                U_new[i * m + j][c] = U_223[i * m + j, r]
                        for j in range(m):
                            for kk in range(3):
                                V_new[j * p + k0 + kk][c] = V_223[j * 3 + kk, r]
                        for kk in range(3):
                            for i in range(n):
                                W_new[(k0 + kk) * n + i][c] = W_223[kk * n + i, r]
                
                # Verify the full decomposition
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = sum(U_new[i*m+j][r] * V_new[j*p+k][r] * W_new[k*n+i][r] for r in range(rank_223))
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U_new, "V": V_new, "W": W_new}
    except Exception:
        pass

    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
