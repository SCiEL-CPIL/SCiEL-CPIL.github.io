"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from scipy.optimize import minimize
from itertools import product

DIMENSIONS = [2, 5, 5]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 if (i,j,k) is valid."""
    T = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1.0
    return T

def _tensor_to_vector(T, n, m, p):
    """Flatten tensor to vector with index (i*m+j)*p+k."""
    return T.reshape(n*m*p)

def _vector_to_tensor(v, n, m, p):
    """Reshape vector back to tensor."""
    return v.reshape(n, m, p)

def _decompose_from_factors(U, V, W, n, m, p):
    """
    Given factor matrices U (n*m x rank), V (m*p x rank), W (p*n x rank),
    reconstruct the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    T = np.zeros((n, m, p), dtype=np.float64)
    U_flat = U.reshape(n, m, rank)
    V_flat = V.reshape(m, p, rank)
    W_flat = W.reshape(p, n, rank)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U_flat[i, j, :], V_flat[j, k, :]) * 0  # This is wrong
                # Correct: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
                # Let's fix this
    # Redo properly
    T = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += u_c[i, j] * v_c[j, k] * w_c[k, i]
    return T

def _error_from_factors(U, V, W, T_target, n, m, p):
    """Compute squared error between decomposed tensor and target."""
    T_decomp = _decompose_from_factors(U, V, W, n, m, p)
    return np.sum((T_decomp - T_target) ** 2)

def _random_init(n, m, p, rank, scale=1.0):
    """Random initialization of factors."""
    U = np.random.randn(n * m, rank) * scale
    V = np.random.randn(m * p, rank) * scale
    W = np.random.randn(p * n, rank) * scale
    return U, V, W

def _flatten_factors(U, V, W):
    """Flatten factors into a single vector for optimization."""
    return np.concatenate([U.flatten(), V.flatten(), W.flatten()])

def _unflatten_factors(x, n, m, p, rank):
    """Unflatten vector back into factors."""
    size_U = n * m * rank
    size_V = m * p * rank
    U = x[:size_U].reshape(n * m, rank)
    V = x[size_U:size_U + size_V].reshape(m * p, rank)
    W = x[size_U + size_V:].reshape(p * n, rank)
    return U, V, W

def _objective(x, T_target, n, m, p, rank):
    """Objective function for optimization."""
    U, V, W = _unflatten_factors(x, n, m, p, rank)
    T_decomp = _decompose_from_factors(U, V, W, n, m, p)
    return np.sum((T_decomp - T_target) ** 2)

def construct():
    n, m, p = DIMENSIONS
    T_target = _build_tensor(n, m, p)
    T_vec = _tensor_to_vector(T_target, n, m, p)
    
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    # Start with schoolbook rank
    best_rank = n * m * p
    best_factors = None
    
    # Generate schoolbook factors as initial best
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    best_factors = (U_school, V_school, W_school)
    
    # Try to find lower rank decompositions
    # For 2x5x5, known lower bounds suggest rank < 50 might be possible
    # Let's try ranks from 49 down to some minimum
    
    for rank in range(best_rank - 1, max(10, best_rank - 20), -1):
        if time.time() > deadline - 20:
            break
        
        # Multiple random restarts
        for restart in range(3):
            if time.time() > deadline - 15:
                break
            
            U, V, W = _random_init(n, m, p, rank, scale=0.5)
            x0 = _flatten_factors(U, V, W)
            
            # Try to optimize
            try:
                result = minimize(
                    _objective,
                    x0,
                    args=(T_target, n, m, p, rank),
                    method='L-BFGS-B',
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < 1e-8:
                    U_opt, V_opt, W_opt = _unflatten_factors(result.x, n, m, p, rank)
                    err = _error_from_factors(U_opt, V_opt, W_opt, T_target, n, m, p)
                    if err < 1e-6:
                        best_rank = rank
                        best_factors = (U_opt, V_opt, W_opt)
                        break  # Found valid decomposition for this rank, try lower
            except Exception:
                continue
    
    # Convert best factors to lists
    U, V, W = best_factors
    rank = U.shape[1]
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
