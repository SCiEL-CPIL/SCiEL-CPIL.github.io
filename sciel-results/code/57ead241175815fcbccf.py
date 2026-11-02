"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check reconstruction: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                recon = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(recon - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _vectorized_als(n, m, p, rank, max_iter=1000, tol=1e-12, seed=0):
    """
    Vectorized Alternating Least Squares for tensor decomposition.
    Uses NumPy broadcasting to update U, V, W efficiently.
    """
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Pre-compute indices for vectorization
    # We need to solve for each row of U, V, W.
    # Update U[i*m+j]: min || A @ u - 1 ||^2 where A[k,c] = V[j*p+k,c] * W[k*n+i,c]
    # This is still a loop over i,j for U, but the inner solve is fast.
    # Can we vectorize the solve?
    # A is (p, rank). We need to solve A @ u = ones(p).
    # We can stack all A matrices into a 3D array (n*m, p, rank) and use np.linalg.lstsq on the batch?
    # np.linalg.lstsq does not support batching in older versions, but we can use pinv.
    # u = pinv(A) @ ones(p).
    
    ones_p = np.ones(p)
    ones_n = np.ones(n)
    ones_m = np.ones(m)
    
    for iteration in range(max_iter):
        # Update U
        # For each i, j, construct A[i,j] of shape (p, rank)
        # A[i,j,k,c] = V[j*p+k, c] * W[k*n+i, c]
        # V is (m*p, rank). W is (p*n, rank).
        
        # Vectorized construction of A for all i, j
        # This is memory intensive: n*m*p*rank. For 3*5*6*rank ~ 90*rank.
        # If rank=50, 4500 floats. Small.
        
        # Reshape V to (m, p, rank)
        V_reshaped = V.reshape(m, p, rank)
        # Reshape W to (p, n, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # We want A[i, j, k, c] = V_reshaped[j, k, c] * W_reshaped[k, i, c]
        # V_reshaped is (m, p, rank). We need to broadcast to (n, m, p, rank).
        # V_b = V_reshaped[None, :, :, :] # (1, m, p, rank)
        # W_b = W_reshaped[:, None, :, :] # (p, 1, n, rank) -> Need to permute to (p, n, 1, rank)?
        # Let's use einsum or broadcasting carefully.
        
        # A[i, j, k, c] = V[j, k, c] * W[k, i, c]
        # V_reshaped: (m, p, rank)
        # W_reshaped: (p, n, rank)
        
        # Expand dims:
        # V_exp: (1, m, p, rank)
        # W_exp: (p, n, 1, rank) -> Permute to (1, 1, p, n, rank)? No.
        
        # Let's just loop over i and j, but vectorize over k?
        # No, the loop over i,j is only 15 iterations. The loop over k is 6.
        # The solve is the expensive part.
        
        for i in range(n):
            for j in range(m):
                # Construct A (p, rank)
                # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                V_slice = V[j*p:(j+1)*p, :] # (p, rank)
                W_slice = W[i::n, :] # (p, rank) - Wait, W index is k*n+i.
                # W rows are k*n+i for k in 0..p-1.
                # So W_slice = W[[k*n+i for k in range(p)], :]
                W_indices = [k*n + i for k in range(p)]
                W_slice = W[W_indices, :]
                
                A = V_slice * W_slice # (p, rank)
                
                # Solve A @ u = ones(p)
                # Use pseudo-inverse for speed and stability
                try:
                    u = np.linalg.pinv(A) @ ones_p
                except np.linalg.LinAlgError:
                    u = np.zeros(rank)
                    
                U[i*m+j, :] = u
        
        # Update V
        for j in range(m):
            for k in range(p):
                # B[i, c] = U[i*m+j, c] * W[k*n+i, c]
                U_slice = U[:j*m+m:(m), :] # No, U indices are i*m+j for i in 0..n-1.
                U_indices = [i*m + j for i in range(n)]
                U_slice = U[U_indices, :] # (n, rank)
                
                W_slice = W[[k*n+i for i in range(n)], :] # (n, rank)
                
                B = U_slice * W_slice # (n, rank)
                
                try:
                    v = np.linalg.pinv(B) @ ones_n
                except np.linalg.LinAlgError:
                    v = np.zeros(rank)
                    
                V[j*p+k, :] = v
        
        # Update W
        for k in range(p):
            for i in range(n):
                # C[j, c] = U[i*m+j, c] * V[j*p+k, c]
                U_indices = [i*m + j for j in range(m)]
                U_slice = U[U_indices, :] # (m, rank)
                
                V_slice = V[[j*p+k for j in range(m)], :] # (m, rank)
                
                C = U_slice * V_slice # (m, rank)
                
                try:
                    w = np.linalg.pinv(C) @ ones_m
                except np.linalg.LinAlgError:
                    w = np.zeros(rank)
                    
                W[k*n+i, :] = w
        
        # Compute loss
        # Loss = sum (sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] - 1)^2
        # We can vectorize this check.
        # Tensor T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        # Use einsum: 'ijc,jkc,kic->ijk'
        try:
            T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
            loss = np.sum((T - 1.0)**2)
        except Exception:
            loss = float('inf')
            
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if loss < tol:
            break
    
    return best_U, best_V, best_W, best_loss

def construct():
    n, m, p = DIMENSIONS
    schoolbook_result = _schoolbook(n, m, p)
    
    start_time = time.time()
    deadline = time.time() + 130.0 # 20s margin
    
    # Try ranks from a lower bound up
    # Lower bound: max(n*m, m*p, p*n) = 30
    # Upper bound: 90
    # We will try ranks in descending order of likelihood of success? 
    # No, we want the MINIMAL rank. So we try from low to high.
    # But low ranks are harder.
    # Strategy: Try a few low ranks with many restarts, then higher ranks with fewer.
    
    ranks_to_try = list(range(30, 90))
    
    best_result = schoolbook_result
    best_rank = 90
    
    # Allocate time budget
    # We will spend most time on ranks 40-80.
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # If we already have a result with rank <= current rank, skip?
        # No, we are iterating from low to high.
        # If we find a valid one, we can stop?
        # The task asks for "rank-minimal".
        # If we find a valid rank 50, we should stop and return it?
        # Yes, because we are iterating from low to high.
        # But wait, is 50 the minimal? We started at 30.
        # If we didn't find anything at 30-49, then 50 is the best we found.
        # So we return the first valid one found.
        
        # Time budget per rank
        # More time for lower ranks? Or less?
        # Lower ranks are harder, so they need more time to find a solution if one exists.
        # But if no solution exists, we waste time.
        # Let's give 2 seconds per rank initially, and increase for lower ranks.
        
        time_budget = 2.0
        if rank < 50:
            time_budget = 5.0
        elif rank < 70:
            time_budget = 3.0
        
        max_iter = 200
        num_seeds = int(time_budget / (time_budget / num_seeds)) # Just pick a number
        num_seeds = 10 if rank < 50 else 5
        
        found = False
        for seed in range(num_seeds):
            if time.time() > deadline:
                break
            
            U, V, W, loss = _vectorized_als(n, m, p, rank, max_iter=max_iter, seed=seed)
            
            if loss < 1e-8:
                # Verify
                U_list = [[round(float(x), 6) for x in row] for row in U]
                V_list = [[round(float(x), 6) for x in row] for row in V]
                W_list = [[round(float(x), 6) for x in row] for row in W]
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return schoolbook_result
# EVOLVE-BLOCK-END
