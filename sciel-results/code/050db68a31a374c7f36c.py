"""Program to find a rank-minimal tensor decomposition for 2x3 by 3x4 matrix multiplication."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24

    # Schoolbook solution
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    best_result = {"U": U, "V": V, "W": W}
    best_rank = schoolbook_rank

    start_time = time.time()
    time_limit = 140.0  # seconds (10s safety margin)

    # Try to find a lower rank decomposition
    # Known lower bounds: for 2x3 * 3x4, the rank is at least max(n*m, m*p, p*n) = 12
    # but more refined bounds exist. Let's try ranks from 23 down to 18.
    for r in range(schoolbook_rank - 1, 17, -1):
        if time.time() - start_time > time_limit:
            break
        
        # Try combinatorial search for rank r
        result = _try_find_decomposition(n, m, p, r, time_limit - (time.time() - start_time))
        if result is not None:
            best_result = result
            best_rank = r
            # If we found a low rank, we could stop, but let's see if we can go lower
            # Actually, let's break if we find one since lower is better but harder
            # We'll continue to try lower ranks if time permits
            continue

    return best_result

def _try_find_decomposition(n, m, p, rank, time_budget):
    """
    Attempt to find a decomposition of the given rank using a combinatorial approach.
    We parameterize the tensor decomposition as:
    T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    
    We use a randomized local search / simulated annealing style approach
    with rounding to half-integers.
    """
    start_time = time.time()
    
    # We'll use a more structured approach.
    # Initialize with random sparse matrices.
    
    np.random.seed(rank * 1000 + n * m * p)
    
    best_residual = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    # Number of iterations based on time budget
    # Estimate: each iteration is cheap for small matrices
    max_iters = 5000
    iters_per_check = 100
    
    for it in range(max_iters):
        if it % iters_per_check == 0 and time.time() - start_time > time_budget:
            break
        
        # Initialize or update factors
        if it == 0:
            # Start with a random initialization
            U = np.random.choice([-1, 0, 1], size=(n * m, rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, rank))
            
            # Ensure some non-trivial structure
            # For each column, ensure at least one non-zero in each factor
            for r in range(rank):
                if np.all(U[:, r] == 0):
                    U[np.random.randint(n * m), r] = 1
                if np.all(V[:, r] == 0):
                    V[np.random.randint(m * p), r] = 1
                if np.all(W[:, r] == 0):
                    W[np.random.randint(p * n), r] = 1
        else:
            # Local update: pick a random factor and a random entry, perturb it
            factor_idx = np.random.randint(3)
            if factor_idx == 0:
                idx = np.random.randint(n * m)
                r_idx = np.random.randint(rank)
                U[idx, r_idx] += np.random.choice([-1, 1])
                U[idx, r_idx] = np.clip(U[idx, r_idx], -2, 2)
            elif factor_idx == 1:
                idx = np.random.randint(m * p)
                r_idx = np.random.randint(rank)
                V[idx, r_idx] += np.random.choice([-1, 1])
                V[idx, r_idx] = np.clip(V[idx, r_idx], -2, 2)
            else:
                idx = np.random.randint(p * n)
                r_idx = np.random.randint(rank)
                W[idx, r_idx] += np.random.choice([-1, 1])
                W[idx, r_idx] = np.clip(W[idx, r_idx], -2, 2)
        
        # Compute residual
        residual = _compute_residual(U, V, W, n, m, p)
        
        if residual < best_residual:
            best_residual = residual
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
            if residual < 1e-10:
                # Found exact solution
                return _finalize_solution(best_U, best_V, best_W, n, m, p)
        
        # If residual is very small, try to polish
        if residual < 1e-6 and it > 100:
            polished = _polish_solution(U, V, W, n, m, p)
            if polished is not None:
                return polished
    
    # If we didn't find an exact solution, check if best is close enough to round
    if best_U is not None and best_residual < 1e-4:
        return _finalize_solution(best_U, best_V, best_W, n, m, p)
    
    return None

def _compute_residual(U, V, W, n, m, p):
    """Compute the sum of squared residuals."""
    residual = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # T[i,j,k] should be 1
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                residual += (val - 1.0) ** 2
    return residual

def _polish_solution(U, V, W, n, m, p):
    """
    Try to polish a near-solution by rounding to half-integers and checking.
    """
    # Try rounding to nearest half-integer
    for scale in [1, 2, 4]:
        U_rounded = np.round(U * scale) / scale
        V_rounded = np.round(V * scale) / scale
        W_rounded = np.round(W * scale) / scale
        
        # Check exactness
        exact = True
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_rounded[i * m + j, :], V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        exact = False
                        break
                if not exact:
                    break
            if not exact:
                break
        
        if exact:
            return _finalize_solution(U_rounded, V_rounded, W_rounded, n, m, p)
    
    return None

def _finalize_solution(U, V, W, n, m, p):
    """
    Convert numpy arrays to lists and verify the solution.
    """
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Final verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(len(U_list[0])):
                    val += U_list[i * m + j][r] * V_list[j * p + k][r] * W_list[k * n + i][r]
                if abs(val - 1.0) > 1e-9:
                    return None
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
