"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import random

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _validate(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly represents matrix multiplication."""
    rank = U.shape[1]
    # Build the tensor and check
    # T[i*m+j, j*p+k, k*n+i] should be 1 for valid (i,j,k), 0 otherwise
    # We check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    # And for invalid index combos, it should be 0.
    
    # Check all valid (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False, (i, j, k, val)
    
    # Check some invalid combinations to ensure no spurious terms
    # For example, U[i*m+j] with V[j'*p+k] where j' != j should not contribute
    # when multiplied with appropriate W
    # This is hard to check exhaustively; we'll rely on the valid checks
    # plus checking a few random invalid combos
    rng = np.random.RandomState(42)
    for _ in range(100):
        i = rng.randint(0, n)
        j = rng.randint(0, m)
        k = rng.randint(0, p)
        # Pick a different j' for V
        j2 = rng.randint(0, m)
        if j2 == j:
            continue
        val = np.sum(U[i*m+j, :] * V[j2*p+k, :] * W[k*n+i, :])
        if abs(val) > tol:
            # This might be a false positive if the structure allows it,
            # but for exact decomposition we want this to be 0
            return False, ("invalid", i, j, k, j2, val)
    
    return True, None

def _try_decomposition(rank, n, m, p, max_iter=5000, seed=42):
    """Try to find a rank-R decomposition using iterative optimization."""
    rng = np.random.RandomState(seed)
    
    U = rng.randn(n * m, rank)
    V = rng.randn(m * p, rank)
    W = rng.randn(p * n, rank)
    
    # Normalize to avoid scale issues
    for r in range(rank):
        norm = np.sqrt(np.sum(U[:, r]**2) * np.sum(V[:, r]**2) * np.sum(W[:, r]**2))
        if norm > 1e-12:
            U[:, r] /= norm ** (1/3)
            V[:, r] /= norm ** (1/3)
            W[:, r] /= norm ** (1/3)
    
    # Build the target tensor as a set of constraints
    # We need: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    
    # Use a simple iterative method: for each constraint, update the factors
    best_err = float('inf')
    best_factors = (U.copy(), V.copy(), W.copy())
    
    for it in range(max_iter):
        # Randomly pick a constraint (i,j,k)
        i = rng.randint(0, n)
        j = rng.randint(0, m)
        k = rng.randint(0, p)
        
        # Current value
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        
        current = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
        error = 1.0 - current
        
        if abs(error) < 1e-10:
            continue
        
        # Update: gradient descent style
        # dU[u_idx, r] += eta * error * V[v_idx, r] * W[w_idx, r]
        # Similarly for V and W
        eta = 0.01
        
        dU = eta * error * V[v_idx, :] * W[w_idx, :]
        dV = eta * error * U[u_idx, :] * W[w_idx, :]
        dW = eta * error * U[u_idx, :] * V[v_idx, :]
        
        U[u_idx, :] += dU
        V[v_idx, :] += dV
        W[w_idx, :] += dW
        
        # Periodically check error
        if it % 500 == 0 or it == max_iter - 1:
            total_err = 0.0
            for ii in range(n):
                for jj in range(m):
                    for kk in range(p):
                        u_i = ii * m + jj
                        v_i = jj * p + kk
                        w_i = kk * n + ii
                        val = np.sum(U[u_i, :] * V[v_i, :] * W[w_i, :])
                        total_err += (1.0 - val) ** 2
            if total_err < best_err:
                best_err = total_err
                best_factors = (U.copy(), V.copy(), W.copy())
            if total_err < 1e-8:
                break
    
    return best_factors, best_err

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 54
    
    # Try to find a lower rank decomposition
    # Start from a few ranks below schoolbook and work down
    # Use multiple random seeds and try different ranks
    
    best_result = _construct_schoolbook()
    best_rank = schoolbook_rank
    
    # Try ranks from a few below schoolbook down
    for target_rank in range(40, 54):
        for seed in [42, 123, 456, 789, 101]:
            factors, err = _try_decomposition(target_rank, n, m, p, max_iter=2000, seed=seed)
            U, V, W = factors
            valid, info = _validate(U, V, W, n, m, p, tol=1e-8)
            if valid:
                # Convert to list format
                U_list = [[float(U[i, r]) for r in range(target_rank)] for i in range(n * m)]
                V_list = [[float(V[i, r]) for r in range(target_rank)] for i in range(m * p)]
                W_list = [[float(W[i, r]) for r in range(target_rank)] for i in range(p * n)]
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # If no lower rank found, return schoolbook
    return best_result
# EVOLVE-BLOCK-END
