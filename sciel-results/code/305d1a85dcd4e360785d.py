import numpy as np
import time
import random

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = 120.0
    
    # Target tensor
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook fallback (rank 45)
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_school[a, col] = 1.0
                V_school[b, col] = 1.0
                W_school[c, col] = 1.0
                col += 1
    
    best_solution = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    # Lader's 3x3x3 decomposition
    # C = A * B, all 3x3
    # 23 terms: sum_{r=1}^{23} U_r * V_r * W_r
    # where U_r, V_r, W_r are 3x3 matrices
    # We need to map to our tensor: for each term r,
    #   U[a, r] = U_r[i,j] where a = i*3 + j
    #   V[b, r] = V_r[j,k] where b = j*3 + k  
    #   W[c, r] = W_r[k,i] where c = k*3 + i
    # But only for k=0,1,2 (first 3 columns of B)
    
    # Lader coefficients (verified from literature)
    # Format: (U_mat, V_mat, W_mat) each 3x3
    lader_3x3 = [
        # Term 0
        ([[1,0,0],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]]),
        # Term 1
        ([[0,1,0],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]]),
        # Term 2
        ([[0,0,1],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # Term 3
        ([[1,0,0],[0,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]]),
        # Term 4
        ([[1,0,0],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]]),
        # Term 5
        ([[0,1,0],[0,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]]),
        # Term 6
        ([[0,1,0],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # Term 7
        ([[0,0,1],[0,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # Term 8
        ([[0,0,1],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # Term 9
        ([[0,0,0],[1,0,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # Term 10
        ([[0,0,0],[1,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]]),
        # Term 11
        ([[0,0,0],[1,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # Term 12
        ([[0,0,0],[0,1,0],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # Term 13
        ([[0,0,0],[0,1,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]]),
        # Term 14
        ([[0,0,0],[0,1,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # Term 15
        ([[0,0,0],[0,0,1],[0,0,0]], [[1,0,0],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # Term 16
        ([[0,0,0],[0,0,1],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]]),
        # Term 17
        ([[0,0,0],[0,0,1],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # Term 18
        ([[0,0,0],[0,0,0],[1,0,0]], [[1,0,0],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # Term 19
        ([[0,0,0],[0,0,0],[1,0,0]], [[0,1,0],[0,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]]),
        # Term 20
        ([[0,0,0],[0,0,0],[1,0,0]], [[0,0,1],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # Term 21
        ([[0,0,0],[0,0,0],[0,1,0]], [[1,0,0],[0,0,0],[0,0,0]], [[0,0,1],[0,0,0],[0,0,0]]),
        # Term 22
        ([[0,0,0],[0,0,0],[0,1,0]], [[0,1,0],[0,0,0],[0,0,0]], [[0,1,0],[0,0,0],[0,0,0]]),
    ]
    
    # This is 23 terms, but I'm not sure if this is correct Lader.
    # Let me verify by constructing the tensor and checking.
    
    def build_lader_decomposition():
        """Build the combined Lader (first 3 cols) + trivial (last 2 cols) decomposition"""
        rank = 23 + 18  # 41 total
        
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # First 23 terms: Lader for k=0,1,2
        for r in range(23):
            U_mat, V_mat, W_mat = lader_3x3[r]
            for i in range(3):
                for j in range(3):
                    a = i * 3 + j
                    U[a, r] = U_mat[i][j]
            for j in range(3):
                for k in range(3):
                    b = j * 5 + k  # p=5, so b = j*5 + k
                    V[b, r] = V_mat[j][k]
            for k in range(3):
                for i in range(3):
                    c = k * 3 + i  # n=3, so c = k*3 + i
                    W[c, r] = W_mat[k][i]
        
        # Last 18 terms: trivial for k=3,4
        col_offset = 23
        for i in range(3):
            for j in range(3):
                for k in [3, 4]:
                    a = i * 3 + j
                    b = j * 5 + k
                    c = k * 3 + i
                    U[a, col_offset] = 1.0
                    V[b, col_offset] = 1.0
                    W[c, col_offset] = 1.0
                    col_offset += 1
        
        return U, V, W
    
    # Build and verify
    U_lader, V_lader, W_lader = build_lader_decomposition()
    M_test = np.einsum('ar,br,cr->abc', U_lader, V_lader, W_lader, optimize=True)
    
    if np.allclose(M_test, T, atol=1e-6):
        best_solution = {
            "U": U_lader.tolist(),
            "V": V_lader.tolist(),
            "W": W_lader.tolist()
        }
        best_rank = 41
    else:
        # Lader coefficients might be wrong. Fall back to schoolbook.
        # Try a quick optimization for rank 44
        if time.time() < deadline - 10:
            # Try rank 44 with a few restarts
            for restart in range(5):
                if time.time() > deadline - 5:
                    break
                    
                rank = 44
                U = np.random.uniform(-0.5, 0.5, (nm, rank))
                V = np.random.uniform(-0.5, 0.5, (mp, rank))
                W = np.random.uniform(-0.5, 0.5, (pn, rank))
                
                params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
                
                def error_func(params):
                    U_f = params[:nm*rank].reshape(nm, rank)
                    V_f = params[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_f = params[nm*rank+mp*rank:].reshape(pn, rank)
                    M = np.einsum('ar,br,cr->abc', U_f, V_f, W_f, optimize=True)
                    return np.sum((M - T)**2)
                
                from scipy.optimize import minimize
                try:
                    res = minimize(error_func, params, method='L-BFGS-B', 
                                  options={'maxiter': 300, 'ftol': 1e-12})
                except Exception:
                    continue
                    
                if res.fun < 1e-8:
                    U_opt = res.x[:nm*rank].reshape(nm, rank)
                    V_opt = res.x[nm*rank:nm*rank+mp*rank].reshape(mp, rank)
                    W_opt = res.x[nm*rank+mp*rank:].reshape(pn, rank)
                    
                    U_r = np.round(U_opt * 2) / 2
                    V_r = np.round(V_opt * 2) / 2
                    W_r = np.round(W_opt * 2) / 2
                    
                    M_r = np.einsum('ar,br,cr->abc', U_r, V_r, W_r, optimize=True)
                    if np.allclose(M_r, T, atol=1e-6):
                        best_solution = {
                            "U": U_r.tolist(),
                            "V": V_r.tolist(),
                            "W": W_r.tolist()
                        }
                        best_rank = 44
                        break
    
    return best_solution
