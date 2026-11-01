import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) else 0
    # a = i*m + j
    # b = j*p + k
    # c = k*n + i
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
                
    indices = np.argwhere(T == 1.0)
    
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds, leave 40s for safety
    
    # Try ranks from 18 upwards
    for R in range(18, 30):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        for init_idx in range(50):
            if time.time() > deadline:
                break
                
            np.random.seed(42 + init_idx)
            
            # Initialize U, V, W
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.random.randn(p * n, R)
            
            # ALS Iterations
            for it in range(50):
                if time.time() > deadline:
                    break
                    
                # Update U
                # For each a, solve U[a, :] * (V * W)^T = T[a, :, :]
                # Reshape V and W to form the linear operator
                # V is (m*p, R), W is (p*n, R)
                # We want M such that U[a, :] @ M = T[a, :, :].reshaped
                # M has shape (m*p * p*n, R)? No.
                # Sum_r U[a,r] V[b,r] W[c,r] = T[a,b,c]
                # Let Z[b,c,r] = V[b,r] W[c,r]
                # Then Sum_r U[a,r] Z[b,c,r] = T[a,b,c]
                # This is a linear system for U[a, :].
                
                # Efficiently:
                # Flatten T[a] to vector of length m*p * p*n
                # Construct matrix A of size (m*p * p*n, R) where A[flat(b,c), r] = V[b,r] W[c,r]
                # This is large: (27 * 18, R) = (486, R).
                
                # We can do this for all a at once?
                # U is (n*m, R).
                # T is (n*m, m*p, p*n).
                
                # Let's just loop over a (only 6 values)
                for a in range(n * m):
                    # Target vector t_a of length m*p * p*n
                    t_a = T[a].flatten()
                    
                    # Build matrix A of shape (m*p * p*n, R)
                    # A[idx, r] = V[b, r] * W[c, r] where idx = b * p*n + c
                    # We can use broadcasting
                    # V is (m*p, R), W is (p*n, R)
                    # V[:, :, None] * W[None, :, :] -> (m*p, p*n, R)
                    # Transpose to (m*p, R, p*n) ? No.
                    # We need columns to be r.
                    # Z = V[:, :, None] * W[None, :, :] # (m*p, p*n, R)
                    # A = Z.transpose(0, 2, 1).reshape(m*p * p*n, R) # (m*p, R, p*n) -> (m*p * p*n, R)
                    
                    Z = V[:, :, None] * W[None, :, :]
                    A = Z.transpose(0, 2, 1).reshape(m * p * p * n, R)
                    
                    # Solve U[a, :] @ A = t_a
                    # U[a, :] = t_a @ A^+
                    # Use lstsq
                    sol, _, _, _ = np.linalg.lstsq(A, t_a, rcond=None)
                    U[a, :] = sol
                    
                # Update V
                # For each b, solve V[b, :] * (U * W)^T = T[:, b, :]
                for b in range(m * p):
                    t_b = T[:, b, :].flatten() # length n*m * p*n
                    
                    # A[idx, r] = U[a, r] * W[c, r] where idx = a * p*n + c
                    Z = U[:, :, None] * W[None, :, :] # (n*m, p*n, R)
                    A = Z.transpose(0, 2, 1).reshape(n * m * p * n, R)
                    
                    sol, _, _, _ = np.linalg.lstsq(A, t_b, rcond=None)
                    V[b, :] = sol
                    
                # Update W
                # For each c, solve W[c, :] * (U * V)^T = T[:, :, c]
                for c in range(p * n):
                    t_c = T[:, :, c].flatten() # length n*m * m*p
                    
                    # A[idx, r] = U[a, r] * V[b, r] where idx = a * m*p + b
                    Z = U[:, :, None] * V[None, :, :] # (n*m, m*p, R)
                    A = Z.transpose(0, 2, 1).reshape(n * m * m * p, R)
                    
                    sol, _, _, _ = np.linalg.lstsq(A, t_c, rcond=None)
                    W[c, :] = sol
                    
            # After ALS, check if we can snap to half-integers
            # Verify current solution
            # Compute tensor product
            # U (6, R), V (27, R), W (18, R)
            # T_recon[a,b,c] = Sum_r U[a,r] V[b,r] W[c,r]
            
            # Check max error
            # We only need to check non-zero entries for speed, but let's check all for safety in final step
            # For now, check a few random entries
            
            error = 0
            for idx in indices:
                a, b, c = idx
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                error += abs(val - 1.0)
                
            # Also check zero entries
            for _ in range(100):
                a = np.random.randint(n * m)
                b = np.random.randint(m * p)
                c = np.random.randint(p * n)
                # Check if it's a zero entry
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                if not (j == j2 and i == i2 and k == k2):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    error += abs(val)
                    
            if error < 1e-4:
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                # Verify snapped solution
                valid = True
                for idx in indices:
                    a, b, c = idx
                    val = np.sum(U_snap[a, :] * V_snap[b, :] * W_snap[c, :])
                    if abs(val - 1.0) > 1e-9:
                        valid = False
                        break
                        
                if valid:
                    # Check zero entries
                    for _ in range(200):
                        a = np.random.randint(n * m)
                        b = np.random.randint(m * p)
                        c = np.random.randint(p * n)
                        i, j = divmod(a, m)
                        j2, k = divmod(b, p)
                        k2, i2 = divmod(c, n)
                        if not (j == j2 and i == i2 and k == k2):
                            val = np.sum(U_snap[a, :] * V_snap[b, :] * W_snap[c, :])
                            if abs(val) > 1e-9:
                                valid = False
                                break
                                
                if valid:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
                        
    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
