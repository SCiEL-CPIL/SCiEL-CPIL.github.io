"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 125  # safety margin before 150s kill

    # Schoolbook gives rank 60.
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    # Try to find lower rank via structured search
    # Strategy: use the fact that for small matrices, we can try
    # various rank values with different initialization strategies
    
    # First, try a few known lower-bound constructions
    # For 2x5*5x6, the rank is at most 60 (schoolbook)
    # Lower bounds: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Also: n*p = 12, so rank >= 12
    
    # Try ranks from 30 up to 59, looking for a valid decomposition
    # Use multiple random seeds and ALS iterations
    
    for rank in range(30, best_rank):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random initializations for this rank
        found = False
        for seed in range(20):
            if time.time() > deadline - 10:
                break
            
            result = _try_rank_als(n, m, p, rank, seed, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                found = True
                break
        
        if found:
            # Found a valid decomposition at this rank, try lower
            # But we're going upward, so actually we should go downward
            # Let me restructure: try from lower to upper
            break
    
    # If we didn't find anything better, also try from 59 down to 30
    if best_rank == n * m * p:
        for rank in range(best_rank - 1, 29, -1):
            if time.time() > deadline - 15:
                break
            
            for seed in range(15):
                if time.time() > deadline - 10:
                    break
                
                result = _try_rank_als(n, m, p, rank, seed, deadline)
                if result is not None:
                    best_result = result
                    best_rank = rank
                    break
            
            if best_rank < n * m * p:
                break
    
    return best_result


def _schoolbook(n, m, p):
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


def _try_rank_als(n, m, p, rank, seed, deadline):
    """Try to find a rank-`rank` decomposition using alternating least squares."""
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    rng = np.random.RandomState(seed)
    
    # Initialize with small random values
    U = rng.randn(dim_u, rank) * 0.1
    V = rng.randn(dim_v, rank) * 0.1
    W = rng.randn(dim_w, rank) * 0.1
    
    max_iter = 80
    tol = 1e-10
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            return None
        
        # Update U: for each (i,j), solve linear system
        new_U = np.zeros((dim_u, rank))
        for i in range(n):
            for j in range(m):
                # G[r, k] = V[j*p+k, r] * W[k*n+i, r]
                G = np.array([[V[j * p + k, r] * W[k * n + i, r] for k in range(p)] for r in range(rank)])
                # Solve: G.T @ new_U[i*m+j, :] = ones(p)
                if G.shape[0] >= p:
                    try:
                        new_U[i * m + j, :] = np.linalg.lstsq(G.T, np.ones(p), rcond=None)[0]
                    except:
                        new_U[i * m + j, :] = np.linalg.pinv(G.T) @ np.ones(p)
                else:
                    # Underdetermined, use minimum norm solution
                    try:
                        new_U[i * m + j, :] = np.linalg.pinv(G.T) @ np.ones(p)
                    except:
                        new_U[i * m + j, :] = np.ones(rank) / rank
        
        U = new_U
        
        # Update V
        new_V = np.zeros((dim_v, rank))
        for j in range(m):
            for k in range(p):
                # H[r, i] = U[i*m+j, r] * W[k*n+i, r]
                H = np.array([[U[i * m + j, r] * W[k * n + i, r] for i in range(n)] for r in range(rank)])
                if H.shape[0] >= n:
                    try:
                        new_V[j * p + k, :] = np.linalg.lstsq(H.T, np.ones(n), rcond=None)[0]
                    except:
                        new_V[j * p + k, :] = np.linalg.pinv(H.T) @ np.ones(n)
                else:
                    try:
                        new_V[j * p + k, :] = np.linalg.pinv(H.T) @ np.ones(n)
                    except:
                        new_V[j * p + k, :] = np.ones(rank) / rank
        
        V = new_V
        
        # Update W
        new_W = np.zeros((dim_w, rank))
        for k in range(p):
            for i in range(n):
                # K[r, j] = U[i*m+j, r] * V[j*p+k, r]
                K = np.array([[U[i * m + j, r] * V[j * p + k, r] for j in range(m)] for r in range(rank)])
                if K.shape[0] >= m:
                    try:
                        new_W[k * n + i, :] = np.linalg.lstsq(K.T, np.ones(m), rcond=None)[0]
                    except:
                        new_W[k * n + i, :] = np.linalg.pinv(K.T) @ np.ones(m)
                else:
                    try:
                        new_W[k * n + i, :] = np.linalg.pinv(K.T) @ np.ones(m)
                    except:
                        new_W[k * n + i, :] = np.ones(rank) / rank
        
        W = new_W
        
        # Check convergence
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    error += abs(val - 1.0)
        
        if error < tol:
            # Try to rationalize
            result = _rationalize(U, V, W, n, m, p)
            if result is not None:
                return result
        
        # Damping to help convergence
        if iteration > 0 and iteration % 10 == 0:
            U = U * 0.95
            V = V * 0.95
            W = W * 0.95
    
    return None


def _rationalize(U, V, W, n, m, p):
    """Try to find rational/half-integer approximation to the decomposition."""
    rank = U.shape[1]
    
    # Try various denominators
    for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]:
        scale = float(denom)
        
        U_int = np.round(U * scale)
        V_int = np.round(V * scale)
        W_int = np.round(W * scale)
        
        # Check if the integer approximation works
        # We need: sum_r (U_int[i*m+j]*V_int[j*p+k]*W_int[k*n+i]) / scale^3 = 1
        # i.e., sum_r U_int[i*m+j]*V_int[j*p+k]*W_int[k*n+i] = scale^3
        
        target = scale ** 3
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_int[i * m + j, :] * V_int[j * p + k, :] * W_int[k * n + i, :])
                    error += abs(val - target)
        
        if error < 1e-6:
            # Convert to list format with proper scaling
            U_list = [[0] * rank for _ in range(n * m)]
            V_list = [[0] * rank for _ in range(m * p)]
            W_list = [[0] * rank for _ in range(p * n)]
            
            for i in range(n * m):
                for r in range(rank):
                    U_list[i][r] = U_int[i, r] / scale
            for i in range(m * p):
                for r in range(rank):
                    V_list[i][r] = V_int[i, r] / scale
            for i in range(p * n):
                for r in range(rank):
                    W_list[i][r] = W_int[i, r] / scale
            
            # Verify
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = sum(U_list[i * m + j][r] * V_list[j * p + k][r] * W_list[k * n + i][r] for r in range(rank))
                        if abs(val - 1.0) > 1e-9:
                            return None
            
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

# EVOLVE-BLOCK-END
