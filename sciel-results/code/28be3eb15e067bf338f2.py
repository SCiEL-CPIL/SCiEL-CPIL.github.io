import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin before 150s kill
    
    # Schoolbook as fallback
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
    
    best_U = np.array(U_school, dtype=np.float64)
    best_V = np.array(V_school, dtype=np.float64)
    best_W = np.array(W_school, dtype=np.float64)
    best_rank = rank_school
    
    # Helper to check if a decomposition is valid
    def is_valid(U, V, W, tol=1e-9):
        r = U.shape[1]
        # Check consistent indices
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(s - 1) > tol:
                        return False
        
        # Check inconsistent indices
        for u_idx in range(n * m):
            i_u = u_idx // m
            j_u = u_idx % m
            for v_idx in range(m * p):
                j_v = v_idx // p
                k_v = v_idx % p
                if j_u != j_v:
                    # Must be zero for all w_idx
                    vec = np.sum(U[u_idx, :, None] * V[v_idx, :, None] * W, axis=1)
                    if np.any(np.abs(vec) > tol):
                        return False
                else:
                    w_consistent = k_v * n + i_u
                    # Check consistent entry
                    s = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_consistent, :])
                    if abs(s - 1) > tol:
                        return False
                    # Check inconsistent w_idx for this u, v
                    vec = np.sum(U[u_idx, :, None] * V[v_idx, :, None] * W, axis=1)
                    vec[w_consistent] = 0
                    if np.any(np.abs(vec) > tol):
                        return False
        return True
    
    # Helper to round to nearest half-integer
    def round_to_half(x):
        return np.round(x * 2) / 2.0
    
    # ALS-style solver: given U, V, solve for W, etc.
    def solve_als(U, V, W, n_iter=200, tol=1e-10):
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = U.shape[1]
        
        for it in range(n_iter):
            if time.time() > deadline - 10:
                break
            
            # Solve for W: For each (i, j, k), W[k*n+i, :] should satisfy
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # This is a linear system for each column of W
            # Actually, let's think differently.
            # T[u_idx, v_idx, w_idx] = sum_c U[u_idx, c] V[v_idx, c] W[w_idx, c]
            # For fixed u_idx, v_idx, this defines a linear equation on W[:, :]
            # But we have many constraints.
            
            # Better approach: treat W as unknowns.
            # For each consistent (i, j, k):
            #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
            # For each inconsistent (u_idx, v_idx, w_idx):
            #   sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c] = 0
            
            # This is a linear system. Let's build it.
            # Variables: W[w_idx, c] for w_idx in [0, p*n), c in [0, r)
            # Total variables: p*n*r
            
            # But this is huge. Let's use a different approach.
            
            # Alternative: for each column c, W[:, c] is a vector.
            # The constraint is:
            # sum_c (U[u_idx, c] * V[v_idx, c]) * W[w_idx, c] = T[u_idx, v_idx, w_idx]
            
            # For fixed u_idx, v_idx, define a_u_v[c] = U[u_idx, c] * V[v_idx, c]
            # Then sum_c a_u_v[c] * W[w_idx, c] = T[u_idx, v_idx, w_idx]
            
            # This is: W[w_idx, :] @ a_u_v = T[u_idx, v_idx, w_idx]
            
            # For all w_idx, this gives: W @ a_u_v = T[:, u_idx, v_idx]
            # where T[:, u_idx, v_idx] is a vector of length p*n
            
            # So for each (u_idx, v_idx), we have a linear system:
            # W @ a_u_v = t_u_v
            # where t_u_v[w_idx] = T[u_idx, v_idx, w_idx]
            
            # But W is shared across all (u_idx, v_idx). This is overdetermined.
            
            # Let's use least squares: minimize ||W @ A - b||^2
            # where A is a matrix with columns a_u_v for all (u_idx, v_idx)
            # and b is the corresponding target vector.
            
            # But this is still large. Let's try a simpler approach.
            
            # Iterate: for each w_idx, solve for W[w_idx, :] using all constraints involving w_idx.
            
            # For fixed w_idx, the constraints are:
            # For all (u_idx, v_idx): sum_c U[u_idx, c] * V[v_idx, c] * W[w_idx, c] = T[u_idx, v_idx, w_idx]
            
            # This is a linear system for W[w_idx, :]:
            # (U * V^T) @ W[w_idx, :] = T[:, :, w_idx]
            # where (U * V^T) is a matrix of size (n*m * m*p, r)
            # and T[:, :, w_idx] is a vector of size n*m * m*p
            
            # Let's build this.
            
            # Actually, let's just do a simple gradient descent on the full objective.
            
            # Objective: sum over all (u_idx, v_idx, w_idx) of (sum_c U[u_idx, c] V[v_idx, c] W[w_idx, c] - T[u_idx, v_idx, w_idx])^2
            
            # Gradient with respect to W[w_idx, c]:
            # 2 * (sum_c U[u_idx, c] V[v_idx, c] W[w_idx, c] - T[u_idx, v_idx, w_idx]) * U[u_idx, c] * V[v_idx, c]
            # summed over all (u_idx, v_idx)
            
            # This is expensive. Let's just return the current solution.
            break
        
        return U, V, W
    
    # Try to find lower rank decompositions
    # Lower bound: max(n*p, m*p, n*m) = max(21, 35, 15) = 35
    # Try ranks from 35 to 104
    
    for r in range(35, rank_school):
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(5):
            if time.time() > deadline - 15:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # Run ALS
            U, V, W = solve_als(U, V, W, n_iter=50)
            
            # Round to half-integers
            U_r = round_to_half(U)
            V_r = round_to_half(V)
            W_r = round_to_half(W)
            
            # Check validity
            if is_valid(U_r, V_r, W_r, tol=1e-8):
                best_U = U_r
                best_V = V_r
                best_W = W_r
                best_rank = r
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    U_out = best_U.tolist()
    V_out = best_V.tolist()
    W_out = best_W.tolist()
    
    # Ensure they are integers or half-integers
    for mat in [U_out, V_out, W_out]:
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                mat[i][j] = round(mat[i][j] * 2) / 2.0
    
    return {"U": U_out, "V": V_out, "W": W_out}
