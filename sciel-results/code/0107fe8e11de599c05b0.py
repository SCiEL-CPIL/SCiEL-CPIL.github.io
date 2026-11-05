"""Matrix multiplication 2,5,6 - rank-minimal tensor decomposition."""
import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
DEADLINE = START_TIME + 115  # 115s wall-clock limit (5s safety margin)

def _time_remaining():
    return DEADLINE - time.time()

# EVOLVE-BLOCK-START

def _build_rank54():
    """Build the known rank-54 Strassen-based decomposition."""
    n, m, p = 2, 5, 6
    rank = 54
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # M0: (A00+A11)(B00+B11)
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        nonlocal col
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        U[(i_base+1) * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1

    # Apply Strassen blocks for j-pairs (0,1) and (2,3)
    for j_base in [0, 2]:
        for k_base in [0, 2, 4]:
            add_strassen_block(0, j_base, k_base)
    
    # Apply naive blocks for j=4
    for k_base in [0, 2, 4]:
        add_naive_2112_block(0, 4, k_base)
    
    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")
    
    return U, V, W


def _verify_decomposition(U, V, W, n=2, m=5, p=6, tol=1e-9):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)
    
    # Check all non-zero entries of the multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check zero entries
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                
                if j_a == j_b and k_b == k_c and i_a == i_c:
                    continue
                
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val) > tol:
                    return False
    
    return True


def _try_rank_reduction(U, V, W, target_rank=53, max_time=60.0):
    """
    Try to reduce rank from 54 to 53 by finding a linear dependency.
    Uses a smooth optimization approach.
    """
    n, m, p = 2, 5, 6
    start_time = time.time()
    rank = U.shape[1]
    
    if rank <= target_rank:
        return None
    
    # Strategy: try to express one term as a linear combination of others
    # T = sum_r U_r ⊗ V_r ⊗ W_r
    # If T_r = sum_{s≠r} c_s * T_s for some r, we can merge.
    
    # More practically: try to find a 53-term decomposition by 
    # relaxing the structure slightly.
    
    # Approach: parameterize 53 terms and optimize to match the tensor.
    # This is a non-convex problem, so use multiple restarts.
    
    # First, check if there's an obvious linear dependency in the current decomposition
    # by checking if any 53 of the 54 terms span the same space.
    
    # Build the tensor T from the rank-54 decomposition
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    
    # Try removing each term and see if the remaining 53 can still represent T
    # by adjusting coefficients (this won't work for exact decomposition, but let's try)
    
    # Better approach: use the fact that the multiplication tensor has specific structure.
    # Try a direct search for rank-53 by parameterizing the problem.
    
    # For rank 53, we have 53 * (n*m + m*p + p*n) = 53 * (10 + 30 + 12) = 53 * 52 = 2756 parameters
    # The tensor has n*m*p = 60 non-zero entries, so we have 60 equations.
    # This is underdetermined, so there should be solutions if rank 53 is achievable.
    
    # Use a least-squares approach with random initialization
    best_result = None
    best_residual = float('inf')
    
    n_params = target_rank * (n*m + m*p + p*n)
    
    # Flatten the target tensor
    target_flat = []
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_flat.append(1.0)
                target_indices.append((a, b, c))
    
    # Also include some zero entries to constrain the solution
    zero_indices = []
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_a == i_c):
                    zero_indices.append((a, b, c))
    
    # Use a subset of zero entries for speed
    np.random.seed(42)
    if len(zero_indices) > 200:
        zero_subset = [zero_indices[i] for i in np.random.choice(len(zero_indices), 200, replace=False)]
    else:
        zero_subset = zero_indices
    
    all_indices = target_indices + zero_subset
    all_targets = [1.0] * len(target_indices) + [0.0] * len(zero_subset)
    
    def compute_residual(params):
        """Compute the residual for a given parameter vector."""
        U_r = params[:target_rank * n*m].reshape(target_rank, n*m).T  # (n*m, target_rank)
        V_r = params[target_rank * n*m : 2*target_rank * n*m].reshape(target_rank, m*p).T  # (m*p, target_rank)
        W_r = params[2*target_rank * n*m : 3*target_rank * n*m].reshape(target_rank, p*n).T  # (p*n, target_rank)
        
        residual = 0.0
        for idx, (a, b, c) in enumerate(all_indices):
            val = np.sum(U_r[a, :] * V_r[b, :] * W_r[c, :])
            residual += (val - all_targets[idx]) ** 2
        return residual
    
    # Try multiple random initializations
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() - start_time > max_time:
            break
        
        if restart == 0:
            # Initialize from the rank-54 solution, dropping one term
            # Try dropping each term and see which gives the best starting point
            best_init_residual = float('inf')
            best_init_params = None
            
            for drop_idx in range(rank):
                # Build initial parameters from rank-54 minus one term
                keep = [r for r in range(rank) if r != drop_idx]
                
                U_init = U[:, keep].T.flatten()  # (target_rank, n*m)
                V_init = V[:, keep].T.flatten()  # (target_rank, m*p)
                W_init = W[:, keep].T.flatten()  # (target_rank, p*n)
                
                params_init = np.concatenate([U_init, V_init, W_init])
                
                res = compute_residual(params_init)
                if res < best_init_residual:
                    best_init_residual = res
                    best_init_params = params_init
            
            if best_init_params is not None:
                x0 = best_init_params
            else:
                x0 = np.random.randn(n_params) * 0.1
        else:
            # Random initialization
            x0 = np.random.randn(n_params) * 0.1
        
        # Use L-BFGS-B for optimization
        def objective(params):
            return compute_residual(params)
        
        # Gradient via finite differences (expensive but necessary)
        def gradient(params):
            eps = 1e-6
            grad = np.zeros_like(params)
            for i in range(min(100, len(params))):  # Limit for speed
                params_plus = params.copy()
                params_plus[i] += eps
                params_minus = params.copy()
                params_minus[i] -= eps
                grad[i] = (objective(params_plus) - objective(params_minus)) / (2 * eps)
            return grad
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20})
            
            if result.fun < best_residual:
                best_residual = result.fun
                best_result = result.x
                
                # Check if we've found a valid decomposition
                if best_residual < 1e-8:
                    U_r = best_result[:target_rank * n*m].reshape(target_rank, n*m).T
                    V_r = best_result[target_rank * n*m : 2*target_rank * n*m].reshape(target_rank, m*p).T
                    W_r = best_result[2*target_rank * n*m : 3*target_rank * n*m].reshape(target_rank, p*n).T
                    
                    if _verify_decomposition(U_r, V_r, W_r, n, m, p, tol=1e-7):
                        return U_r, V_r, W_r
                        
        except Exception as e:
            pass
    
    return None


def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Start with the rank-54 decomposition
    U, V, W = _build_rank54()
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        raise ValueError("Base rank-54 decomposition failed verification")
    
    # Try to reduce to rank 53
    if _time_remaining() > 70:  # Only attempt if we have enough time
        result = _try_rank_reduction(U, V, W, target_rank=53, max_time=60.0)
        if result is not None:
            U, V, W = result
            # Final verification
            if _verify_decomposition(U, V, W, n, m, p, tol=1e-7):
                pass  # Successfully reduced to rank 53
            else:
                # Fall back to rank 54
                U, V, W = _build_rank54()
    
    # Convert to lists for the return format
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
