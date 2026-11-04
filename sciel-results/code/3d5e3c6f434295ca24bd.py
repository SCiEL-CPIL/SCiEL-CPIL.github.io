import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition with rank p*m*n."""
    rank = p * m * n
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

def _compute_C(U, V, W, A, B, n, m, p):
    """Compute C = sum_t W[:,t] * (U[:,t]·A) * (V[:,t]·B) using vectorized ops."""
    r = U.shape[1]
    u_vals = U.T @ A.flatten()  # shape: (r,)
    v_vals = V.T @ B.flatten()  # shape: (r,)
    uv = u_vals * v_vals  # (r,)
    W_reshaped = W.reshape(p, n, r)  # (p, n, r)
    C_reshaped = np.einsum('kit,t->ki', W_reshaped, uv)
    return C_reshaped.T  # (n, p)

def _residual_multi(U, V, W, n, m, p, A_list, B_list):
    """Compute residual over multiple test matrices for robustness."""
    total_res = 0.0
    for A, B in zip(A_list, B_list):
        C_true = A @ B
        C_decomp = _compute_C(U, V, W, A, B, n, m, p)
        total_res += np.sum((C_true - C_decomp) ** 2)
    return total_res

def _try_rank_rational(n, m, p, r, deadline):
    """Try to find a rank-r decomposition with rational entries using structured search."""
    # For small r, we can try a more direct approach
    # The key constraint is that for all A, B:
    # A @ B = sum_t (U[:,t]·A) * (V[:,t]·B) * W[:,t]
    
    # This is equivalent to saying the tensor T[i,j,k] = delta_{i,k} (for matmul)
    # can be written as sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    
    # For exact rational solutions, we need to solve a system of polynomial equations.
    # This is hard in general, but for small dimensions we can try.
    
    # Strategy: Use the fact that if we fix U and V, then W is determined (if possible).
    # Specifically, for each (i,k), we need:
    # delta_{i,k} = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] summed appropriately
    
    # Actually, the correct formulation is:
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # = sum_t (sum_{i',j'} U[i'*m+j', t] * A[i',j']) * (sum_{j',k'} V[j'*p+k', t] * B[j',k']) * W[k*n+i, t]
    
    # This must hold for ALL A, B, which gives us constraints on U, V, W.
    
    # A simpler approach: try random rational initializations and check if they work.
    # For rank r, we have 3*r*(dimensions) variables, which is a lot.
    
    # Alternative: use the schoolbook as a starting point and try to eliminate columns.
    # If column t is a linear combination of other columns in U, V, W simultaneously,
    # we can eliminate it.
    
    U_school, V_school, W_school, _ = _build_schoolbook(n, m, p)
    
    # Try to find a subset of schoolbook columns that still spans the tensor
    # This is equivalent to finding a minimal generating set
    
    # For now, return None if we can't find one quickly
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)
    
    # Start with schoolbook
    U_school, V_school, W_school, rank_school = _build_schoolbook(n, m, p)
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_school
    
    # Generate test matrices for verification
    np.random.seed(12345)
    A_list = [np.random.randn(n, m) for _ in range(10)]
    B_list = [np.random.randn(m, p) for _ in range(10)]
    
    # Try to find lower rank decompositions using a more systematic approach
    # Key insight: try to find rank-r decompositions by solving the tensor equation
    
    # For each candidate rank, try to find a decomposition
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # Upper bound: n*m*p = 84
    
    # Try ranks from 28 to 83
    for target_rank in range(28, rank_school):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations with rational values
        found = False
        for trial in range(50):
            if time.time() > deadline - 3:
                break
            
            np.random.seed(trial * 10000 + target_rank * 100 + 42)
            
            # Initialize with small rational values
            # Try values from {-1, -0.5, 0, 0.5, 1}
            vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
            
            U_init = np.random.choice(vals, size=(n * m, target_rank))
            V_init = np.random.choice(vals, size=(m * p, target_rank))
            W_init = np.random.choice(vals, size=(p * n, target_rank))
            
            # Check if this decomposition works
            res = _residual_multi(U_init, V_init, W_init, n, m, p, A_list, B_list)
            if res < 1e-10:
                best_U = U_init.copy()
                best_V = V_init.copy()
                best_W = W_init.copy()
                best_rank = target_rank
                found = True
                break
        
        if found:
            break
    
    # Convert to lists
    best_U_list = [[float(x) for x in row] for row in best_U]
    best_V_list = [[float(x) for x in row] for row in best_V]
    best_W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": best_U_list, "V": best_V_list, "W": best_W_list}
