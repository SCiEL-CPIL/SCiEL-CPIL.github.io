import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Target rank 21
    target_rank = 21
    
    # Tensor dimensions
    dim_u = n * m  # 10
    dim_v = m * p  # 30
    dim_w = p * n  # 12
    
    # Build the target tensor T[a, b, c]
    # a = i*m + j, b = j*p + k, c = k*n + i
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Reshape T for ALS: T_mat[a, b*dim_w + c]
    T_mat = T.reshape(dim_u, dim_v * dim_w)
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    # Try multiple random restarts
    for attempt in range(30):
        if time.time() > deadline - 10:
            break
        
        # Random initialization
        rng = np.random.RandomState(attempt * 12345 + 67890)
        U = rng.randn(dim_u, target_rank)
        V = rng.randn(dim_v, target_rank)
        W = rng.randn(dim_w, target_rank)
        
        # ALS iterations
        for step in range(200):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each column c, U[:, c] = (V[:, c] ⊗ W[:, c])^T T / ||V[:, c] ⊗ W[:, c]||^2
            for c in range(target_rank):
                vw = np.outer(V[:, c], W[:, c])
                norm_sq = np.dot(vw.ravel(), vw.ravel())
                if norm_sq < 1e-12:
                    U[:, c] = 0
                    continue
                # U[:, c] = sum over (b,c_idx) of T[a, b, c_idx] * vw[a, b] / norm_sq
                # = (T_mat * vw) / norm_sq  where T_mat is (dim_u, dim_v*dim_w)
                U[:, c] = (T_mat @ vw.ravel()) / norm_sq
            
            # Update V: for each column c, V[:, c] = (U[:, c] ⊗ W[:, c])^T T / ||U[:, c] ⊗ W[:, c]||^2
            for c in range(target_rank):
                uw = np.outer(U[:, c], W[:, c])
                norm_sq = np.dot(uw.ravel(), uw.ravel())
                if norm_sq < 1e-12:
                    V[:, c] = 0
                    continue
                # T_mat is (dim_u, dim_v*dim_w), uw is (dim_u, dim_v)
                # V[:, c] = sum over (a, c_idx) of T[a, b, c_idx] * uw[a, c_idx] / norm_sq
                # = (T_mat.T @ uw.ravel()) / norm_sq
                V[:, c] = (T_mat.T @ uw.ravel()) / norm_sq
            
            # Update W: for each column c, W[:, c] = (U[:, c] ⊗ V[:, c])^T T / ||U[:, c] ⊗ V[:, c]||^2
            for c in range(target_rank):
                uv = np.outer(U[:, c], V[:, c])
                norm_sq = np.dot(uv.ravel(), uv.ravel())
                if norm_sq < 1e-12:
                    W[:, c] = 0
                    continue
                # W[:, c] = sum over (a, b) of T[a, b, c_idx] * uv[a, b] / norm_sq
                # = (T_mat @ uv.ravel()) / norm_sq
                W[:, c] = (T_mat @ uv.ravel()) / norm_sq
        
        # Check error
        err = 0.0
        for c in range(target_rank):
            contribution = np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
            err += np.sum((contribution - T)**2)
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_err < 1e-8:
            break
    
    # If we found a good decomposition, round to half-integers
    if best_U is not None and best_err < 1e-6:
        # Round to nearest half-integer
        def round_to_half(x):
            return np.round(x * 2) / 2.0
        
        U_rounded = round_to_half(best_U)
        V_rounded = round_to_half(best_V)
        W_rounded = round_to_half(best_W)
        
        # Verify the rounded version
        err_rounded = 0.0
        for c in range(target_rank):
            contribution = np.outer(np.outer(U_rounded[:, c], V_rounded[:, c]), W_rounded[:, c])
            err_rounded += np.sum((contribution - T)**2)
        
        if err_rounded < 1e-6:
            # Return as lists of fractions
            U_list = [[float(x) for x in row] for row in U_rounded]
            V_list = [[float(x) for x in row] for row in V_rounded]
            W_list = [[float(x) for x in row] for row in W_rounded]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook decomposition (rank 60)
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
