import numpy as np
import time

DIMENSIONS = [2, 2, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
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

    # Tensor T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    # Target ranks to try
    # Based on knowledge, 18 is good. Try 17, 18.
    target_ranks = [17, 18, 19, 20]
    
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    for r in target_ranks:
        if time.time() > deadline:
            break
        # Try to find a rank-r decomposition
        # Use multiple random restarts
        for restart in range(10):
            if time.time() > deadline:
                break
            # Initialize U, V, W randomly
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS iterations
            for it in range(100):
                # Update U
                # T is n x m x p
                # Residual R = T - sum_c U[:, c] outer V[:, c] outer W[:, c]
                # This is expensive to compute directly.
                # Instead, solve linear system for U given V, W.
                # T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
                
                # We can vectorize the update.
                # For fixed V, W, the problem is linear in U.
                # Let A_{(i,j), (j',k')} be the coefficient matrix for U?
                # Actually, it's easier to use the mode-n unfolding or just solve the least squares.
                
                # Let's construct the linear system for U.
                # T_{i,j,k} = sum_c U_{i*m+j, c} * (V_{j*p+k, c} * W_{k*n+i, c})
                # This couples i,j,k.
                
                # Alternative: Use the fact that T is small.
                # We can just compute the residual and use gradient descent? No, ALS is better.
                
                # Let's implement a simple ALS step.
                # For each row of U (index i*m+j), we need to fit the slice.
                # This is complex. Let's use a generic ALS implementation for 3-way tensors.
                
                # Unfold T to matrix T_(1) of size n*m x p? No, dimensions are mixed.
                # Let's stick to a simpler, robust method: Randomized SVD on the tensor unfolded?
                # Or just use `scipy.optimize`? No, too slow.
                
                # Let's use a standard ALS loop for tensor decomposition.
                # We need to solve:
                # min_U || T - U * V * W ||_F
                # where * denotes the tensor product.
                
                # We can compute the "contraction" of T with V and W to get a matrix for U.
                # Let M_U be the matrix such that U = argmin ||T - U M_U||
                # This is not straightforward because V and W change.
                
                # Let's try a different approach:
                # Since n, m, p are small, we can solve the linear system for each factor directly.
                
                # Update U:
                # For each (i, j), we want to find U[i*m+j, :] such that
                # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] approximates T[i, j, k] for all k.
                # This is a linear system in U[i*m+j, :].
                # Let B_{(i,j), (k, c)} = V[j*p+k, c] * W[k*n+i, c]
                # Then T[i, j, :] = U[i*m+j, :] * B_{(i,j), :}^T ?
                # No, sum_c U_{ij, c} B_{ik, c} = T_{ijk}.
                # So for fixed i, j, we solve:
                # U_{ij, c} * sum? No.
                # sum_c U_{ij, c} * (V_{jk, c} * W_{ki, c}) = T_{ijk}
                # Let A_{k, c} = V_{jk, c} * W_{ki, c}
                # Then A @ U_{ij, :}^T = T_{ij, :}^T
                # So U_{ij, :}^T = pinv(A) @ T_{ij, :}^T
                
                # We can do this for all i, j.
                
                # 1. Update U
                new_U = np.zeros_like(U)
                for i in range(n):
                    for j in range(m):
                        # Construct A of shape (p, r)
                        # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        A = V[j*p:(j+1)*p, :] * W[(i):n*p:n, :] # W indices: k*n+i for k=0..p-1
                        # Solve A @ u = T[i, j, :]
                        # u is shape (r,)
                        # T[i, j, :] is shape (p,)
                        try:
                            u, *_ = np.linalg.lstsq(A, T[i, j, :], rcond=None)
                            new_U[i*m+j, :] = u
                        except:
                            new_U[i*m+j, :] = U[i*m+j, :]
                U = new_U
                
                # 2. Update V
                new_V = np.zeros_like(V)
                for j in range(m):
                    for k in range(p):
                        # Construct A of shape (n, r)
                        # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        # Indices for U: i*m+j for i=0..n-1
                        # Indices for W: k*n+i for i=0..n-1
                        U_slice = U[0:n*m:m, :] # This is wrong, U is (n*m, r). Row i*m+j.
                        # We need U[i*m+j, c] for i=0..n-1.
                        # So rows are j, m+j, 2m+j ...
                        U_rows = [i*m + j for i in range(n)]
                        U_sub = U[U_rows, :]
                        W_rows = [k*n + i for i in range(n)]
                        W_sub = W[W_rows, :]
                        A = U_sub * W_sub
                        # Solve A @ v = T[:, j, k]
                        # T[:, j, k] is shape (n,)
                        try:
                            v, *_ = np.linalg.lstsq(A, T[:, j, k], rcond=None)
                            new_V[j*p+k, :] = v
                        except:
                            new_V[j*p+k, :] = V[j*p+k, :]
                V = new_V
                
                # 3. Update W
                new_W = np.zeros_like(W)
                for k in range(p):
                    for i in range(n):
                        # Construct A of shape (m, r)
                        # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        U_rows = [i*m + j for j in range(m)]
                        U_sub = U[U_rows, :]
                        V_rows = [j*p + k for j in range(m)]
                        V_sub = V[V_rows, :]
                        A = U_sub * V_sub
                        # Solve A @ w = T[i, :, k]
                        # T[i, :, k] is shape (m,)
                        try:
                            w, *_ = np.linalg.lstsq(A, T[i, :, k], rcond=None)
                            new_W[k*n+i, :] = w
                        except:
                            new_W[k*n+i, :] = W[k*n+i, :]
                W = new_W
                
                # Check convergence
                # Compute error
                # Reconstruct T_rec
                # T_rec[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
                # We can do this efficiently
                # Let's just compute it for verification at the end of restart.
                
            # After ALS, round and verify
            U_rounded = np.round(U * 2) / 2 # Half integers
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify exactness
            # T_rec[i, j, k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                        if not np.isclose(val, 1.0, atol=1e-9):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Convert to list of lists
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
                
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
