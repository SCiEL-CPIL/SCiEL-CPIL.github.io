<<<<<<< SEARCH
def _construct_rank38():
    """
    Construct rank-38 decomposition: 4 Strassen blocks (8 cols) + 1 rank-10 block (3 cols).
    Uses a known rank-10 decomposition for M_{2,2,3}.
    """
    n, m, p = 2, 2, 11
    rank = 38
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    # 4 Strassen blocks for k=0..7
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        w_c00 = k0 * n
        w_c10 = k0 * n + 1
        w_c01 = k1 * n
        w_c11 = k1 * n + 1
        
        # Strassen terms
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_c10][r] = 1; W[w_c11][r] = -1
        r += 1
        
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_c01][r] = 1; W[w_c11][r] = 1
        r += 1
        
        U[3][r] = 1
        V[v_g][r] = 1; V[v_e][r] = -1
        W[w_c00][r] = 1; W[w_c10][r] = 1
        r += 1
        
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_c00][r] = -1; W[w_c01][r] = 1
        r += 1
        
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_c11][r] = 1
        r += 1
        
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_c00][r] = 1
        r += 1
    
    # Rank 10 block for k=8,9,10
    # We need a rank 10 decomposition for 2x2 x 2x3.
    # This is a known result. I will use a specific decomposition found in literature or derived.
    # For now, I will use a search if I don't have it hardcoded.
    # But to be safe, I'll implement a quick search for this sub-problem.
    
    # Sub-problem: n=2, m=2, p=3.
    # Indices:
    # U: 0..3
    # V: 0..5 (k=0,1,2 for row 0; k=0,1,2 for row 1)
    # W: 0..5 (k=0,1,2 for col 0; k=0,1,2 for col 1)
    
    # I will use a hardcoded rank 10 solution if I can verify it.
    # Otherwise, I'll return None and fall back to rank 39.
    
    # Let's try to construct one.
    # This is complex. I will use a random search for the sub-block.
    
    def find_rank10_223(deadline):
        # Random search for rank 10 M_{2,2,3}
        # This might be slow.
        # I'll skip this and return None to use rank 39 if not found quickly.
        return None
    
    # If we don't have a rank 10 block, we can't make rank 38.
    # So this function returns None.
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Try rank 38
    # Since I don't have a guaranteed rank 10 block, I'll try a local search on the full problem for rank 38.
    # But that is risky.
    
    # I will return the rank 39 solution as it is verified.
    U, V, W = _construct_rank39()
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
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
=======
def _search_low_rank(n, m, p, target_rank, deadline, seed=42):
    """
    Search for a decomposition of rank < target_rank using iterative refinement.
    Uses random initialization and Newton-like updates on the tensor equations.
    """
    import random
    import numpy as np
    
    rng = random.Random(seed)
    start_time = time.time()
    
    # Number of constraints: n*m*p
    n_constraints = n * m * p
    
    # We have rank variables, each with n*m + m*p + p*n entries
    # Total variables: rank * (n*m + m*p + p*n)
    # But we can fix gauge: for each c, set U[0][c]=1 (if nonzero)
    
    # Strategy: random initialization, then iterate:
    # For each constraint (i,j,k), we need sum_c U[i*m+j][c]*V[j*p+k][c]*W[k*n+i][c] = 1
    # This is nonlinear. Use alternating minimization or gradient descent.
    
    # Simpler: try to find a rank-r decomposition by solving a system.
    # For small ranks, we can try random linear combinations.
    
    # Alternative: use the fact that rank-39 works, and try to merge columns.
    # Start with rank-39 and try to eliminate one column by adjusting others.
    
    U39, V39, W39 = _construct_rank39()
    rank39 = len(U39[0])
    
    # Try to reduce rank by 1: set one column to zero and adjust others.
    # This is hard. Instead, do random search for rank < 39.
    
    # For each candidate rank from target_rank to 38:
    for r in range(target_rank, min(39, rank39)):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Random initialization
            U = [[rng.uniform(-1, 1) for _ in range(r)] for _ in range(n * m)]
            V = [[rng.uniform(-1, 1) for _ in range(r)] for _ in range(m * p)]
            W = [[rng.uniform(-1, 1) for _ in range(r)] for _ in range(p * n)]
            
            # Iterative refinement: for each constraint, update variables to satisfy it.
            # Use a simple approach: for each (i,j,k), compute residual and distribute to variables.
            
            converged = False
            for iteration in range(100):
                if time.time() > deadline:
                    break
                
                # Compute residuals
                max_res = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                            res = 1.0 - s
                            max_res = max(max_res, abs(res))
                
                if max_res < 1e-8:
                    converged = True
                    break
                
                # Update: for each constraint, adjust the variables.
                # Simple approach: for each (i,j,k), pick a random c and adjust.
                # Better: gradient descent on sum of squared residuals.
                
                # Compute gradients
                dU = [[0.0] * r for _ in range(n * m)]
                dV = [[0.0] * r for _ in range(m * p)]
                dW = [[0.0] * r for _ in range(p * n)]
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for c in range(r):
                                s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                            res = 1.0 - s
                            
                            for c in range(r):
                                # d/dU[i*m+j][c] of (1-s)^2 = -2*res * V[j*p+k][c]*W[k*n+i][c]
                                dU[i * m + j][c] += 2 * res * V[j * p + k][c] * W[k * n + i][c]
                                dV[j * p + k][c] += 2 * res * U[i * m + j][c] * W[k * n + i][c]
                                dW[k * n + i][c] += 2 * res * U[i * m + j][c] * V[j * p + k][c]
                
                # Gradient descent step
                lr = 0.01
                for i in range(n * m):
                    for c in range(r):
                        U[i][c] -= lr * dU[i][c]
                for j in range(m * p):
                    for c in range(r):
                        V[j][c] -= lr * dV[j][c]
                for k in range(p * n):
                    for c in range(r):
                        W[k][c] -= lr * dW[k][c]
                
                if max_res < 1e-10:
                    converged = True
                    break
            
            if converged:
                # Verify
                if _verify(U, V, W, n, m, p, tol=1e-7):
                    # Round to rationals if possible
                    return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # First, try to find a lower rank solution
    best_U, best_V, best_W = None, None, None
    
    # Try rank 38
    result = _search_low_rank(n, m, p, 38, deadline, seed=123)
    if result is not None:
        best_U, best_V, best_W = result
    else:
        # Try rank 39 (should always work)
        U39, V39, W39 = _construct_rank39()
        if _verify(U39, V39, W39, n, m, p):
            best_U, best_V, best_W = U39, V39, W39
    
    if best_U is not None:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Fallback: schoolbook
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
>>>>>>> REPLACE
