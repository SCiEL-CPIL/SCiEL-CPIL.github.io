import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    u_size = n * m
    v_size = m * p
    w_size = p * n
    
    # Precompute indices for non-zero and zero constraints
    # Non-zero: (i, j, k) -> u_idx, v_idx, w_idx
    non_zero_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                non_zero_indices.append((u_idx, v_idx, w_idx))
                
    # Zero constraints: For each u_idx, w_idx, the only non-zero v_idx is the one matching.
    # We can generate a subset of zero constraints to check during optimization,
    # but for verification we check all.
    # For ALS, we minimize the error on non-zero terms and hope zeros are small.
    
    def als_step(U, V, W, r, max_iters=100, tol=1e-8):
        # U: (u_size, r), V: (v_size, r), W: (w_size, r)
        # Equation: U[u, c] * V[v, c] * W[w, c] sum over c = Target[u, v, w]
        
        # This is not linear in all variables simultaneously.
        # We can treat it as:
        # For each c, the term is U[:, c] outer V[:, c] outer W[:, c].
        # This is not standard ALS because of the product of three matrices.
        
        # We can rewrite:
        # Let M_c = U[:, c] * V[:, c] (element-wise product, shape u_size x v_size? No, U is u_size x r, V is v_size x r).
        # U[:, c] is a vector of size u_size.
        # V[:, c] is a vector of size v_size.
        # The product U[u, c] * V[v, c] is a matrix of size u_size x v_size.
        # Then multiplied by W[w, c] (scalar for each w).
        
        # We can optimize W given U, V:
        # For each w, Target[:, :, w] (flattened u, v) = sum_c W[w, c] * (U[:, c] * V[:, c])
        # This is linear in W[w, :].
        
        # Let A_w be a matrix where A_w[u * v_size + v, c] = U[u, c] * V[v, c].
        # Then Target_w (vector of size u_size * v_size) = A_w @ W[w, :].
        # We can solve for W[w, :] using least squares.
        
        # Similarly for U and V.
        
        best_error = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            # Update W
            for w in range(w_size):
                # Build A_w: (u_size * v_size, r)
                A_w = np.zeros((u_size * v_size, r))
                for u in range(u_size):
                    for v in range(v_size):
                        A_w[u * v_size + v, :] = U[u, :] * V[v, :]
                
                # Target vector for this w
                # Target is 1 if (u, v) corresponds to a non-zero term with this w, else 0.
                # Non-zero terms: u = i*m+j, v = j*p+k, w = k*n+i.
                # For a fixed w (k, i), the non-zero u, v are those where j is such that u=(i,j), v=(j,k).
                # So for fixed k, i, the non-zero pairs are (u=(i,j), v=(j,k)) for all j.
                
                target_w = np.zeros(u_size * v_size)
                for j in range(m):
                    u_idx = i * m + j  # Wait, i is derived from w
                    # w = k * n + i. So i = w % n, k = w // n.
                    i_val = w % n
                    k_val = w // n
                    u_idx = i_val * m + j
                    v_idx = j * p + k_val
                    target_w[u_idx * v_size + v_idx] = 1.0
                
                # Solve A_w @ W[w, :] = target_w
                # Use lstsq
                try:
                    W[w, :], residuals, rank, sv = np.linalg.lstsq(A_w, target_w, rcond=None)
                except:
                    continue
            
            # Update U
            for u in range(u_size):
                # Target for u: sum_c U[u, c] * (V[:, c] * W[:, c]) ...
                # For fixed u (i, j), the non-zero terms are (v=(j,k), w=(k,i)) for all k.
                A_u = np.zeros((v_size * w_size, r))
                for v in range(v_size):
                    for w in range(w_size):
                        A_u[v * w_size + w, :] = V[v, :] * W[w, :]
                
                target_u = np.zeros(v_size * w_size)
                i_val = u // m
                j_val = u % m
                for k in range(p):
                    v_idx = j_val * p + k
                    w_idx = k * n + i_val
                    target_u[v_idx * w_size + w_idx] = 1.0
                
                try:
                    U[u, :], residuals, rank, sv = np.linalg.lstsq(A_u, target_u, rcond=None)
                except:
                    continue

            # Update V
            for v in range(v_size):
                # For fixed v (j, k), non-zero terms are (u=(i,j), w=(k,i)) for all i.
                A_v = np.zeros((u_size * w_size, r))
                for u in range(u_size):
                    for w in range(w_size):
                        A_v[u * w_size + w, :] = U[u, :] * W[w, :]
                
                target_v = np.zeros(u_size * w_size)
                j_val = v // p
                k_val = v % p
                for i in range(n):
                    u_idx = i * m + j_val
                    w_idx = k_val * n + i
                    target_v[u_idx * w_size + w_idx] = 1.0
                
                try:
                    V[v, :], residuals, rank, sv = np.linalg.lstsq(A_v, target_v, rcond=None)
                except:
                    continue
            
            # Check convergence / error
            # Compute error on non-zero terms
            err = 0.0
            for u_idx, v_idx, w_idx in non_zero_indices:
                val = 0.0
                for c in range(r):
                    val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                err += (val - 1.0) ** 2
            
            if err < best_error:
                best_error = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < tol:
                break
        
        return best_U, best_V, best_W, best_error

    def verify_solution(U, V, W, r):
        # Check non-zero
        for u_idx, v_idx, w_idx in non_zero_indices:
            val = 0.0
            for c in range(r):
                val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
            if abs(val - 1.0) > 1e-9:
                return False
        
        # Check zeros
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    w_idx = k * n + i
                    correct_v_idx = j * p + k
                    
                    for j_prime in range(m):
                        for k_prime in range(p):
                            v_idx = j_prime * p + k_prime
                            if v_idx == correct_v_idx:
                                continue
                            
                            val = 0.0
                            for c in range(r):
                                val += U[u_idx, c] * V[v_idx, c] * W[w_idx, c]
                            if abs(val) > 1e-9:
                                return False
        return True

    start_time = time.time()
    
    # Try ranks from 40 to 60
    for r in range(40, rank_school + 1):
        if time.time() - start_time > 120:
            break
        
        found = False
        attempts = 10
        
        for attempt in range(attempts):
            if time.time() - start_time > 120:
                break
            
            np.random.seed(attempt)
            U0 = np.random.randn(u_size, r)
            V0 = np.random.randn(v_size, r)
            W0 = np.random.randn(w_size, r)
            
            U_sol, V_sol, W_sol, err = als_step(U0, V0, W0, r, max_iters=50)
            
            # Round to nearest 0.5
            U_round = np.round(U_sol * 2) / 2
            V_round = np.round(V_sol * 2) / 2
            W_round = np.round(W_sol * 2) / 2
            
            if verify_solution(U_round, V_round, W_round, r):
                return {
                    "U": U_round.tolist(),
                    "V": V_round.tolist(),
                    "W": W_round.tolist()
                }
        
    # Fallback to schoolbook
    rank = rank_school
    U = [[0] * rank for _ in range(u_size)]
    V = [[0] * rank for _ in range(v_size)]
    W = [[0] * rank for _ in range(w_size)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
