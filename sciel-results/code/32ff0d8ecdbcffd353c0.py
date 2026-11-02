import numpy as np
from scipy.optimize import least_squares
import time
import random

START_TIME = time.time()
BUDGET_SECONDS = 140.0

def check_time():
    return time.time() - START_TIME < BUDGET_SECONDS

def construct():
    n, m, p = 3, 4, 5
    # Dimensions of factors
    dim_u = n * m  # 12
    dim_v = m * p  # 20
    dim_w = p * n  # 15
    
    # Target tensor T[i, j, k] = 1 if i corresponds to (row, col1), j to (col1, col2), k to (col2, row)
    # Wait, the index convention is:
    # U: (n*m, rank) -> row i*m + j
    # V: (m*p, rank) -> row j*p + k
    # W: (p*n, rank) -> row k*n + i
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # We will try to find a rank 59 decomposition.
    # If that fails, we return rank 60.
    
    # Precompute the target tensor values for non-zero entries
    # There are n*m*p = 60 non-zero entries in the tensor.
    # The tensor is sparse.
    
    def objective(params, rank):
        # params is a flat vector of length rank * (dim_u + dim_v + dim_w)
        U_flat = params[:rank * dim_u]
        V_flat = params[rank * dim_u : rank * dim_u + rank * dim_v]
        W_flat = params[rank * dim_u + rank * dim_v :]
        
        U = U_flat.reshape(rank, dim_u).T # (dim_u, rank)
        V = V_flat.reshape(rank, dim_v).T # (dim_v, rank)
        W = W_flat.reshape(rank, dim_w).T # (dim_w, rank)
        
        # Compute error
        # We only check the 60 non-zero positions
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx_u = i * m + j
                    idx_v = j * p + k
                    idx_w = k * n + i
                    # Sum_r U[idx_u, r] * V[idx_v, r] * W[idx_w, r]
                    val = np.sum(U[idx_u, :] * V[idx_v, :] * W[idx_w, :])
                    # Target is 1
                    errors.append(val - 1)
        return np.array(errors)

    def try_find_decomposition(rank, num_restarts=5):
        for _ in range(num_restarts):
            if not check_time():
                return None
            
            # Initialize with small random values
            # Or initialize with a subset of schoolbook terms
            # For rank 59, we can start with 59 random vectors
            
            size = rank * (dim_u + dim_v + dim_w)
            # Random initialization
            x0 = np.random.randn(size) * 0.1
            
            try:
                result = least_squares(objective, x0, args=(rank,), method='lm', max_nfev=1000)
            except Exception:
                continue
            
            if result.success or result.cost < 1e-6:
                # Extract U, V, W
                U_flat = result.x[:rank * dim_u]
                V_flat = result.x[rank * dim_u : rank * dim_u + rank * dim_v]
                W_flat = result.x[rank * dim_u + rank * dim_v :]
                
                U = U_flat.reshape(rank, dim_u).T
                V = V_flat.reshape(rank, dim_v).T
                W = W_flat.reshape(rank, dim_w).T
                
                # Round to nearest integer or half-integer
                # The problem allows integer/half-integer entries.
                # Let's try rounding to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exactness
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            idx_u = i * m + j
                            idx_v = j * p + k
                            idx_w = k * n + i
                            val = np.sum(U_r[idx_u, :] * V_r[idx_v, :] * W_r[idx_w, :])
                            if abs(val - 1) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    # Convert to lists
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        return None

    # Try Rank 59
    result_59 = try_find_decomposition(59, num_restarts=10)
    if result_59 is not None:
        return result_59
        
    # Fallback to Rank 60 (Schoolbook)
    rank = 60
    U = [[0] * rank for _ in range(dim_u)]
    V = [[0] * rank for _ in range(dim_v)]
    W = [[0] * rank for _ in range(dim_w)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                U[idx_u][col] = 1
                V[idx_v][col] = 1
                W[idx_w][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
