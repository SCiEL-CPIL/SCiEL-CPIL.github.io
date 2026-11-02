import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    nm = n * m  # 6
    mp = m * p  # 15
    pn = p * n  # 10
    
    deadline = time.time() + 140  # Safety margin
    
    def validate(U, V, W, tol=1e-9):
        """
        Check if decomposition is valid.
        U: (nm, rank), V: (mp, rank), W: (pn, rank)
        Condition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        """
        rank = U.shape[1] if U.ndim > 1 else 0
        # Reshape for vectorized check
        # U_reshaped: (n, m, rank)
        # V_reshaped: (m, p, rank)
        # W_reshaped: (p, n, rank)
        
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
        # We need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
        # This is a tensor contraction.
        # For each i, j, k:
        # val = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
        
        # Vectorized approach:
        # Create arrays of indices
        # This might be memory intensive if rank is large, but rank is small here.
        
        # Let's just loop over the small dimensions i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def snap_half_ints(M):
        """Snap matrix entries to nearest half-integer."""
        return np.round(M * 2) / 2

    def snap_ints(M):
        """Snap matrix entries to nearest integer."""
        return np.round(M)

    def try_rank(rank, seed=0, max_iters=100):
        """
        Try to find valid decomposition of given rank using vectorized alternating linear solves.
        """
        rng = random.Random(seed)
        
        # Initialize with small random values
        U = np.random.uniform(-1, 1, (nm, rank))
        V = np.random.uniform(-1, 1, (mp, rank))
        W = np.random.uniform(-1, 1, (pn, rank))
        
        best_err = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), solve A * U_row = ones(p)
            # A[k,c] = V[j*p+k][c] * W[k*n+i][c]
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    # Construct A (p x rank)
                    # V rows: j*p to j*p+p-1
                    # W rows: k*n+i for k in 0..p-1
                    V_slice = V[j*p : (j+1)*p, :] # (p, rank)
                    W_slice = np.array([W[k*n+i, :] for k in range(p)]) # (p, rank)
                    A = V_slice * W_slice
                    b = np.ones(p)
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i*m+j, :] = sol
                    except:
                        pass
            U = U_new
            
            # Update V: for each (j,k), solve A * V_row = ones(n)
            # A[i,c] = U[i*m+j][c] * W[k*n+i][c]
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    # Construct A (n x rank)
                    # U rows: i*m+j for i in 0..n-1
                    # W rows: k*n+i for i in 0..n-1
                    U_slice = np.array([U[i*m+j, :] for i in range(n)]) # (n, rank)
                    W_slice = np.array([W[k*n+i, :] for i in range(n)]) # (n, rank)
                    A = U_slice * W_slice
                    b = np.ones(n)
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        V_new[j*p+k, :] = sol
                    except:
                        pass
            V = V_new
            
            # Update W: for each (k,i), solve A * W_row = ones(m)
            # A[j,c] = U[i*m+j][c] * V[j*p+k][c]
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    # Construct A (m x rank)
                    # U rows: i*m+j for j in 0..m-1
                    # V rows: j*p+k for j in 0..m-1
                    U_slice = np.array([U[i*m+j, :] for j in range(m)]) # (m, rank)
                    V_slice = np.array([V[j*p+k, :] for j in range(m)]) # (m, rank)
                    A = U_slice * V_slice
                    b = np.ones(m)
                    try:
                        sol, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_new[k*n+i, :] = sol
                    except:
                        pass
            W = W_new
            
            # Check error
            # Vectorized error calculation
            U_r = U.reshape(n, m, rank)
            V_r = V.reshape(m, p, rank)
            W_r = W.reshape(p, n, rank)
            
            # err = sum_{i,j,k} (sum_c U[i,j,c] V[j,k,c] W[k,i,c] - 1)^2
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                        err += (val - 1.0) ** 2
            
            if err < best_err:
                best_err = err
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if err < 1e-10:
                # Found a solution! Try to snap to half-integers
                U_r = snap_half_ints(U)
                V_r = snap_half_ints(V)
                W_r = snap_half_ints(W)
                if validate(U_r, V_r, W_r, tol=1e-9):
                    return U_r, V_r, W_r
                
                U_i = snap_ints(U)
                V_i = snap_ints(V)
                W_i = snap_ints(W)
                if validate(U_i, V_i, W_i, tol=1e-9):
                    return U_i, V_i, W_i
                
                # If snapping fails, maybe the solution is not exactly on the grid yet.
                # We can return the best found if it's very close, but the task requires exact integer/half-integer.
                # So we must find one that snaps correctly.
        
        # If best_err is very small, try snapping the best solution found
        if best_err < 1e-6 and best_U is not None:
            U_r = snap_half_ints(best_U)
            V_r = snap_half_ints(best_V)
            W_r = snap_half_ints(best_W)
            if validate(U_r, V_r, W_r, tol=1e-9):
                return U_r, V_r, W_r
            
            U_i = snap_ints(best_U)
            V_i = snap_ints(best_V)
            W_i = snap_ints(best_W)
            if validate(U_i, V_i, W_i, tol=1e-9):
                return U_i, V_i, W_i
        
        return None

    # Fallback: return schoolbook decomposition (rank 30)
    def get_schoolbook():
        rank = n * m * p  # 30
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, r] = 1.0
                    V[j*p+k, r] = 1.0
                    W[k*n+i, r] = 1.0
                    r += 1
        return U, V, W

    best_result = None
    
    # Try ranks from lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = 15
    for R in range(15, 31):
        if time.time() > deadline - 10:
            break
        
        # Try multiple seeds
        for seed in range(10):
            if time.time() > deadline - 5:
                break
            
            result = try_rank(R, seed=seed, max_iters=50)
            if result is not None:
                U, V, W = result
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback
    U, V, W = get_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
