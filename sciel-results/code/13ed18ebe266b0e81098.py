"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product
from scipy.optimize import lsq_linear

DIMENSIONS = [4, 4, 7]
DEADLINE_SECONDS = 135.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, num_tests=10, tol=1e-6):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)  # Note: Task says W[k*n+i, r]. So W is (p*n, rank).
                     # In my previous thought, I used W[i,k]. 
                     # Let's map: index in W is k*n + i.
                     # So W[idx, c] corresponds to coefficient for output (i,k) in term c.
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for _ in range(num_tests):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        
        for c in range(rank):
            # Extract U_c as matrix of shape (n, m)
            u_c = U[:, c].reshape(n, m)
            # Extract V_c as matrix of shape (m, p)
            v_c = V[:, c].reshape(m, p)
            
            # L_A_c is a scalar: sum(u_c * A)
            l_a = np.sum(u_c * A)
            # L_B_c is a scalar: sum(v_c * B)
            l_b = np.sum(v_c * B)
            
            # W_c is a matrix of shape (p, n) based on indexing W[k*n+i, c]
            # We need to add l_a * l_b * W_c to C.
            # C[i, k] += l_a * l_b * W[k*n+i, c]
            w_c_flat = W[:, c]
            w_c = w_c_flat.reshape(p, n)
            
            # w_c[k, i] is the coefficient for C[i, k]
            # We need to add to C[i, k] the value: l_a * l_b * w_c[k, i]
            # So C += (l_a * l_b) * w_c.T
            
            C_computed += (l_a * l_b) * w_c.T
        
        if not np.allclose(C_computed, C_expected, atol=tol, rtol=tol):
            return False
    return True

def _build_schoolbook(n, m, p):
    """Build the standard schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Term c computes A[i,j] * B[j,k] and adds to C[i,k]
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                # W index: k*n + i
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_solve_for_w(n, m, p, target_rank, start_time, deadline, attempt_seed=0):
    """
    Try to find U, V, W by randomizing U, V and solving linear systems for W.
    """
    np.random.seed(attempt_seed)
    
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    
    # Generate sparse random U and V with small integer entries
    # Bias towards structured sparsity to make linear systems well-conditioned
    for c in range(target_rank):
        # Randomly pick a few non-zeros
        u_nonzero_idx = np.random.choice(n*m, size=max(1, int(n*m*0.3)), replace=False)
        v_nonzero_idx = np.random.choice(m*p, size=max(1, int(m*p*0.3)), replace=False)
        
        for idx in u_nonzero_idx:
            U[idx, c] = np.random.choice([-2, -1, 1, 2])
        for idx in v_nonzero_idx:
            V[idx, c] = np.random.choice([-2, -1, 1, 2])
            
    # We need to solve for W.
    # W has shape (p*n, target_rank).
    # For each output (i, k), we have a vector W_{:, i, k} of length target_rank.
    # Let's solve for each (i, k) independently.
    
    # Generate random test matrices A, B to form the linear system
    num_samples = 2 * target_rank + 10
    
    A_samples = np.random.randn(num_samples, n, m)
    B_samples = np.random.randn(num_samples, m, p)
    
    W_solution = np.zeros((p*n, target_rank))
    
    valid = True
    
    for i in range(n):
        for k in range(p):
            # Target values y_t for each sample t
            # y_t = A_t[i, :] dot B_t[:, k]
            y = np.zeros(num_samples)
            for t in range(num_samples):
                y[t] = np.dot(A_samples[t, i, :], B_samples[t, :, k])
                
            # Feature matrix X
            # X[t, c] = L_A_c(A_t) * L_B_c(B_t)
            X = np.zeros((num_samples, target_rank))
            for c in range(target_rank):
                u_c = U[:, c].reshape(n, m)
                v_c = V[:, c].reshape(m, p)
                
                for t in range(num_samples):
                    l_a = np.sum(u_c * A_samples[t])
                    l_b = np.sum(v_c * B_samples[t])
                    X[t, c] = l_a * l_b
                    
            # Solve X * W_col = y
            # Use least squares
            try:
                w_col, residuals, rank_x, sv = np.linalg.lstsq(X, y, rcond=None)
            except np.linalg.LinAlgError:
                valid = False
                break
                
            # Check residual
            if residuals[0] > 1e-6:
                valid = False
                break
                
            # Check if w_col is "nice" (integer or half-integer)
            w_doubled = w_col * 2
            w_rounded = np.round(w_doubled)
            if not np.allclose(w_doubled, w_rounded, atol=1e-5):
                valid = False
                break
                
            W_solution[k*n + i, :] = w_rounded / 2.0
            
        if not valid:
            break
            
    if not valid:
        return None
        
    # Verify with held-out samples
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W_solution.tolist()
    
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, num_tests=5, tol=1e-5):
        return {"U": U_list, "V": V_list, "W": W_list}
        
    return None

def _try_strassen_block_construction(n, m, p):
    """
    Construct a decomposition using Strassen for the 4x4x4 part and schoolbook for the rest,
    but optimized to share terms if possible.
    """
    # This is a fallback to ensure we get a valid result if random search fails,
    # but usually random search with the solver is better.
    # For now, return schoolbook.
    return _build_schoolbook(n, m, p)

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from low to high
    # Known lower bound for 4x4x4 is ~21, for 4x4x7 likely higher.
    # Schoolbook is 112.
    # Let's try up to 60.
    for target_rank in range(30, best_rank):
        if time.time() > deadline - 5:
            break
            
        # Try a few random attempts for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
                
            try:
                result = _try_solve_for_w(n, m, p, target_rank, start_time, deadline, attempt_seed=target_rank*1000 + attempt)
                if result is not None:
                    # Verify again just in case
                    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
                        return result
            except Exception:
                continue
                
    return best

# EVOLVE-BLOCK-END
