import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Target Tensor T
    # T[a, b, d] where a in [0, n*m), b in [0, m*p), d in [0, n*p)
    T = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = i * p + k
                T[a, b, d] = 1.0

    best_rank = rank_schoolbook
    best_U = None
    best_V = None
    best_W = None

    # Try ranks from schoolbook-1 downwards
    # 24 is schoolbook. Try 23, 22...
    # It is unlikely we go much lower.
    
    for r in range(rank_schoolbook - 1, 15, -1): # Try down to 16
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        num_restarts = 20 if r > 20 else 10
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
                
            # Initialize U, V, W randomly
            # Use small integers to help convergence to rational solutions?
            # Or just random floats.
            U = np.random.randint(-2, 3, (n*m, r)).astype(float)
            V = np.random.randint(-2, 3, (m*p, r)).astype(float)
            W = np.random.randint(-2, 3, (n*p, r)).astype(float)
            
            # ALS Refinement
            converged = False
            for it in range(50):
                if time.time() > deadline:
                    break
                    
                # Update U
                # For each a, solve LHS_a U[a] = RHS_a
                # LHS_a shape (m*p * n*p, r)
                # RHS_a shape (m*p * n*p, 1)
                new_U = np.zeros_like(U)
                for a in range(n*m):
                    # Construct LHS
                    # Indices b, d
                    # LHS[b, d, c] = V[b, c] * W[d, c]
                    # We can vectorize this?
                    # V is (m*p, r), W is (n*p, r)
                    # Outer product of V and W?
                    # V[:, np.newaxis, :] * W[np.newaxis, :, :] -> (m*p, n*p, r)
                    # Reshape to (m*p * n*p, r)
                    LHS = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, r)
                    RHS = T[a, :, :].reshape(-1)
                    
                    # Solve least squares
                    # np.linalg.lstsq
                    try:
                        sol, _, _, _ = np.linalg.lstsq(LHS, RHS, rcond=None)
                        new_U[a] = sol
                    except:
                        pass
                        
                U = new_U
                
                # Update V
                new_V = np.zeros_like(V)
                for b in range(m*p):
                    # LHS[a, d, c] = U[a, c] * W[d, c]
                    LHS = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(-1, r)
                    RHS = T[:, b, :].reshape(-1)
                    
                    try:
                        sol, _, _, _ = np.linalg.lstsq(LHS, RHS, rcond=None)
                        new_V[b] = sol
                    except:
                        pass
                        
                V = new_V
                
                # Update W
                new_W = np.zeros_like(W)
                for d in range(n*p):
                    # LHS[a, b, c] = U[a, c] * V[b, c]
                    LHS = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(-1, r)
                    RHS = T[:, :, d].reshape(-1)
                    
                    try:
                        sol, _, _, _ = np.linalg.lstsq(LHS, RHS, rcond=None)
                        new_W[d] = sol
                    except:
                        pass
                        
                W = new_W
                
                # Check residual
                # Residual = || sum_c U[:, c] V[:, c] W[:, c] - T ||
                # Compute tensor product sum
                # This is expensive. 
                # T_rec = np.einsum('ac,bc,dc->abd', U, V, W)
                # But we can check specific non-zero entries?
                # No, we need all entries to be correct.
                # Let's compute the residual norm.
                
                if it % 5 == 0:
                    T_rec = np.einsum('ac,bc,dc->abd', U, V, W)
                    res = np.linalg.norm(T_rec - T)
                    if res < 1e-6:
                        converged = True
                        break
            
            if converged:
                # We have a floating point solution.
                # Try to find a rational solution.
                # Round to nearest half-integer?
                # The problem says "integer/half-integer".
                # Let's try rounding to nearest 0.5.
                
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exactness
                T_rec_r = np.einsum('ac,bc,dc->abd', U_r, V_r, W_r)
                if np.allclose(T_rec_r, T, atol=1e-9):
                    # Found a valid rank r solution
                    best_rank = r
                    best_U = U_r
                    best_V = V_r
                    best_W = W_r
                    break # Break restarts loop, try lower rank?
                    # Actually, if we found rank r, we can try r-1?
                    # The loop goes downwards.
                    # So we break restarts, and the outer loop continues to r-1.
        
        # If we found a solution for this r, we can break if we want the minimal?
        # The loop is for r in range(...).
        # If we find a solution, we store it and continue to lower r?
        # Yes, we want rank-minimal.
        
    # If no lower rank found, use schoolbook
    if best_rank == rank_schoolbook:
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    else:
        # Convert to list of lists of numbers
        # Ensure they are integers or half-integers
        def to_list(mat):
            return [[float(x) for x in row] for row in mat]
        
        return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
