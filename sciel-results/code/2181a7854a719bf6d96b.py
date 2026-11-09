"""Improved program with structured search for lower rank decomposition."""
import time
import numpy as np
import scipy.linalg

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill
    
    # 1. Baseline: Schoolbook Decomposition (Rank 60)
    rank_baseline = n * m * p
    U_base = [[0] * rank_baseline for _ in range(n * m)]
    V_base = [[0] * rank_baseline for _ in range(m * p)]
    W_base = [[0] * rank_baseline for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_base[i * m + j][col] = 1
                V_base[j * p + k][col] = 1
                W_base[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_base, "V": V_base, "W": W_base}
    best_rank = rank_baseline
    deadline = start_time + time_limit - 3.0
    
    # Pre-compute index arrays for faster access
    U_indices = np.arange(n * m).reshape(n, m)  # U_indices[i,j] = i*m+j
    V_indices = np.arange(m * p).reshape(m, p)  # V_indices[j,k] = j*p+k
    
    # Try to find lower rank using structured approaches
    for R in range(rank_baseline - 1, 39, -1):
        if time.time() > deadline:
            break
            
        # Number of trials per rank
        trials = 500 if R > 50 else 300
        
        for t in range(trials):
            if time.time() > deadline:
                break
                
            # Generate random U and V with entries from {-1, 0, 1}
            U = np.random.randint(-1, 2, size=(n * m, R))
            V = np.random.randint(-1, 2, size=(m * p, R))
            
            U_float = U.astype(np.float64)
            V_float = V.astype(np.float64)
            
            # For each (i, k) pair, we need to solve:
            # For each j: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is A_{ik} * w_{ik} = 1_m where A_{ik}[j, c] = U[i*m+j, c] * V[j*p+k, c]
            
            W_candidate = np.zeros((p * n, R), dtype=np.float64)
            valid = True
            
            for i in range(n):
                for k in range(p):
                    # Build A matrix of shape (m, R)
                    # A[j, :] = U[U_indices[i,j], :] * V[V_indices[j,k], :]
                    A = U_float[U_indices[i, :], :] * V_float[V_indices[:, k], :]
                    
                    # Check if system A * w = ones(m) is consistent
                    # Use SVD for both consistency check and solution
                    try:
                        U_svd, s_vals, Vt_svd = np.linalg.svd(A, full_matrices=False)
                        rank_A = np.sum(s_vals > 1e-10)
                        
                        # Check consistency: ones(m) must be in column space of A
                        # Project ones(m) onto column space
                        proj = U_svd[:, :rank_A] @ (U_svd[:, :rank_A].T @ np.ones(m))
                        if not np.allclose(proj, np.ones(m), atol=1e-8):
                            valid = False
                            break
                        
                        # Find minimum norm solution
                        if rank_A > 0:
                            vec_u = U_svd[:, :rank_A].T @ np.ones(m)
                            vec_s_inv_u = vec_u / s_vals[:rank_A]
                            w_min = Vt_svd[:rank_A].T @ vec_s_inv_u
                        else:
                            valid = False
                            break
                        
                        # Check if minimum norm solution is close to half-integer
                        w_half = np.round(w_min * 2) / 2.0
                        if np.allclose(A.dot(w_half), np.ones(m), atol=1e-8):
                            W_candidate[k * n + i, :] = w_half
                        else:
                            # Try null space search using scipy for robustness
                            found = False
                            # Only search if there is a null space
                            if R > rank_A:
                                try:
                                    from scipy.linalg import null_space
                                    ns_basis = null_space(A)
                                    if ns_basis.size > 0:
                                        for _ in range(30):
                                            y = np.random.randn(ns_basis.shape[1])
                                            w_try = w_min + ns_basis @ y
                                            w_try_half = np.round(w_try * 2) / 2.0
                                            if np.allclose(A.dot(w_try_half), np.ones(m), atol=1e-8):
                                                W_candidate[k * n + i, :] = w_try_half
                                                found = True
                                                break
                                except Exception:
                                    pass
                            
                            if not found:
                                valid = False
                                break
                    except np.linalg.LinAlgError:
                        valid = False
                        break
                        
                if not valid:
                    break
            
            if valid:
                # Verify the full tensor equation using vectorized computation
                # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
                # Build tensor of products
                # U_expanded: (n, m, R)
                # V_expanded: (m, p, R)  
                # W_expanded: (p, n, R)
                
                U_exp = U_float.reshape(n, m, R)
                V_exp = V_float.reshape(m, p, R)
                W_exp = W_candidate.reshape(p, n, R)
                
                # For each (i,j,k): sum over c of U_exp[i,j,c] * V_exp[j,k,c] * W_exp[k,i,c]
                # Use einsum: result[i,j,k] = sum_c U_exp[i,j,c] * V_exp[j,k,c] * W_exp[k,i,c]
                result = np.einsum('ijc,jkc,kic->ijk', U_exp, V_exp, W_exp)
                
                if np.allclose(result, 1.0, atol=1e-7):
                    best_solution = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W_candidate.tolist()
                    }
                    best_rank = R
                    break  # Found solution for this R, move to R-1

    return best_solution
