"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    """
    Schoolbook tensor decomposition for 3x4 x 4x4 matrix multiplication.
    
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    U: shape (n*m, rank) = (12, 48)
    V: shape (m*p, rank) = (16, 48)  
    W: shape (p*n, rank) = (12, 48)
    
    For each (i,j,k) triple, one rank-1 term:
      U[i*m+j, c] = 1
      V[j*p+k, c] = 1
      W[k*n+i, c] = 1
    """
    n, m, p = DIMENSIONS  # 3, 4, 4
    rank = n * m * p  # 48
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify exact decomposition of the multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Compute T[i,k,j] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t]
    T = np.zeros((n, p, m), dtype=np.float64)
    
    for t in range(rank):
        u_col = U_arr[:, t]
        v_col = V_arr[:, t]
        w_col = W_arr[:, t]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, k, j] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    
    # The multiplication tensor should be all-ones
    expected = np.ones((n, p, m), dtype=np.float64)
    return np.allclose(T, expected, atol=1e-10)

def _round_to_half_int(arr):
    """Round numpy array to nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _try_optimize_rank(R, deadline, seed=0):
    """
    Attempt to find a rank-R decomposition using gradient descent and rounding.
    """
    n, m, p = DIMENSIONS
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n * m, R) * 0.1
    V = np.random.randn(m * p, R) * 0.1
    W = np.random.randn(p * n, R) * 0.1
    
    # Precompute indices for faster evaluation
    # We need to minimize sum_{i,j,k} (sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] - 1)^2
    
    # Simple Gradient Descent
    lr = 0.01
    steps = 500 # Keep small to fit in time
    
    for step in range(steps):
        if time.time() > deadline:
            return None
            
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # Calculate error and gradients
        # T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
        # dE/dU[i,j,t] = 2 * (T[i,j,k] - 1) * V[j,k,t] * W[k,i,t]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute T[i,j,k]
                    t_val = 0.0
                    for t in range(R):
                        t_val += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                    
                    err = t_val - 1.0
                    if abs(err) < 1e-6:
                        continue
                        
                    for t in range(R):
                        dU[i*m+j, t] += 2 * err * V[j*p+k, t] * W[k*n+i, t]
                        dV[j*p+k, t] += 2 * err * U[i*m+j, t] * W[k*n+i, t]
                        dW[k*n+i, t] += 2 * err * U[i*m+j, t] * V[j*p+k, t]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Optional: Regularize towards sparsity or small values?
        # Not strictly necessary for rounding, but helps convergence.
        
    # Round to half-integers
    U_r = _round_to_half_int(U)
    V_r = _round_to_half_int(V)
    W_r = _round_to_half_int(W)
    
    # Convert to lists
    U_list = U_r.tolist()
    V_list = V_r.tolist()
    W_list = W_r.tolist()
    
    # Verify
    if _verify_decomposition(U_list, V_list, W_list):
        return {"U": U_list, "V": V_list, "W": W_list}
        
    return None

def _try_find_lower_rank(deadline):
    """Attempt to find a rank lower than 48 using optimization."""
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook()
    current_rank = len(schoolbook["U"][0])
    
    # Try ranks from 47 down to 19 (known lower bound for 4x4x4 is 19, so 3x4x4 <= 19 is possible but unlikely to be much lower)
    # We start from 47 and go down.
    for R in range(current_rank - 1, 18, -1):
        if time.time() > deadline - 5:
            break
            
        # Try a few seeds for this rank
        for seed in range(3):
            if time.time() > deadline - 5:
                break
            result = _try_optimize_rank(R, deadline, seed)
            if result is not None:
                return result
                
    return None

def construct():
    """Construct the tensor decomposition with deadline guard."""
    start_time = time.time()
    deadline = start_time + 110  # 120s limit, leave 10s margin for safety
    
    # Start with schoolbook (rank 48)
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try to find a lower rank
    if time.time() < deadline - 5:
        result = _try_find_lower_rank(deadline)
        if result is not None:
            U, V, W = result
            new_rank = len(U[0])
            if new_rank < best_rank and _verify_decomposition(U, V, W):
                best = {"U": U, "V": V, "W": W}
                best_rank = new_rank
    
    return best
# EVOLVE-BLOCK-END
