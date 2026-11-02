"""Matrix multiplication 3,4,5 decomposition."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """
    Build the target tensor T with shape (n*m, m*p, p*n).
    T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i', j==j', k==k' else 0.
    """
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T[idx_u, idx_v, idx_w] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify if the decomposition U, V, W exactly reproduces the target tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Compute the decomposed tensor D
    # D[x, y, z] = sum_r U[x, r] * V[y, r] * W[z, r]
    D = np.einsum('xr,yr,zr->xyz', U, V, W)
    
    # Build target tensor
    T = _build_target_tensor(n, m, p)
    
    return np.allclose(D, T, atol=tol, rtol=0)

def _round_to_half_integer(arr):
    """Round array values to the nearest integer or half-integer."""
    # Multiply by 2, round to nearest integer, divide by 2
    return (np.round(arr * 2) / 2)

def _search_lower_rank(n, m, p, start_rank, end_rank, time_limit):
    """
    Attempt to find a decomposition with rank between end_rank and start_rank (inclusive)
    using a structured linear search over U and V.
    """
    T_target = _build_target_tensor(n, m, p)
    
    for rank in range(start_rank, end_rank - 1, -1):
        if time.time() > time_limit:
            break
            
        print(f"Trying rank {rank} with structured search...", file=sys.stderr)
        
        # Try a few random restarts for this rank
        # We use a limited number of trials to stay within time limits
        num_trials = 500 
        np.random.seed(rank)
        
        for trial in range(num_trials):
            if time.time() > time_limit:
                break
                
            # Randomly initialize U and V with entries in {0, 1}
            # We use binary initialization to encourage sparsity
            U_init = np.random.randint(0, 2, size=(n * m, rank))
            V_init = np.random.randint(0, 2, size=(m * p, rank))
            
            # Construct the matrix M for the linear system
            # M[i*m+j, r] = U[i*m+j, r] * V[j*p+k, r] ? No.
            # The equation is T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
            # For fixed k, we have T[:, :, k] = sum_r (U[:, r] * V[:, r]) * W[k, r]
            # Let col_r = U[:, r] * V[:, r] (element-wise product, shape n*m)
            # Then T[:, :, k] is a vector of length n*m (flattened)
            # We want to solve M * W_k = T_k
            # where M is n*m x rank, with M[i, r] = U[i, r] * V[i, r] ?
            # Wait, V index is j*p+k.
            # T[i, j, k] depends on U[i, r] and V[j, r] and W[k, r].
            # The index of V is j.
            # So for a fixed k, the term is sum_r U[i, r] * V[j, r] * W[k, r].
            # This is not a simple linear system in W_k unless we fix i, j.
            # The vector T_k has entries T[i, j, k] for all i, j.
            # The coefficient of W[k, r] in T[i, j, k] is U[i, r] * V[j, r].
            # So M[i, j, r] = U[i, r] * V[j, r].
            # This is a 3D tensor, not a matrix.
            # We can flatten i, j to get a matrix M_flat of size (n*m) x rank.
            # M_flat[i*m+j, r] = U[i, r] * V[j, r].
            
            M_flat = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    # U index: i*m + j ? No, U is (n*m, rank).
                    # In schoolbook, U[i*m+j, col] = 1.
                    # So U row index is i*m + j.
                    # V row index is j*p + k.
                    # Wait, V depends on j and k.
                    # In the sum for T[i, j, k], V is indexed by j and k.
                    # But V is a matrix (m*p, rank).
                    # So V[j*p + k, r].
                    # But k is fixed in the loop for W_k.
                    # So for a fixed k, V is a vector of length m (for j=0..m-1) for each r.
                    # Specifically, V_k[j, r] = V[j*p + k, r].
                    
                    # So M_flat[i*m+j, r] = U[i*m+j, r] * V[j*p+k, r].
                    # This depends on k!
                    # So we need to solve for W_k for each k.
                    pass # This logic is inside the k loop below

            valid = True
            W_solution = np.zeros((p * n, rank))
            
            for k in range(p):
                # Build M_k of shape (n*m, rank)
                M_k = np.zeros((n * m, rank))
                for i in range(n):
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        M_k[u_idx, :] = U_init[u_idx, :] * V_init[v_idx, :]
                
                # Target vector T_k of shape (n*m,)
                T_k = T_target[:, :, k].flatten()
                
                # Solve M_k * W_k = T_k
                try:
                    W_k, residuals, rank_k, s = np.linalg.lstsq(M_k, T_k, rcond=None)
                except Exception:
                    valid = False
                    break
                
                # Store W_k in W_solution
                # W_solution index: k*n + i ?
                # The output factor follows W[k*n+i, r].
                # So for fixed k, W[k*n + i, r] = W_k[i, r].
                for i in range(n):
                    W_solution[k * n + i, :] = W_k[i, :]
            
            if not valid:
                continue
                
            # Check if W_solution entries are close to {0, 0.5, 1}
            # We allow -0.5, -1 too?
            # The problem says integer/half-integer.
            # Let's check distance to nearest half-integer.
            W_rounded = _round_to_half_integer(W_solution)
            
            # Check if rounding changed the values significantly
            if not np.allclose(W_solution, W_rounded, atol=1e-6):
                continue
            
            # Verify the full decomposition
            if _verify_decomposition(U_init, V_init, W_rounded, n, m, p):
                print(f"Found valid rank {rank} decomposition!", file=sys.stderr)
                return U_init, V_init, W_rounded

    return None

def _schoolbook(n, m, p):
    """Generate the schoolbook decomposition."""
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
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # 1. Generate Schoolbook Decomposition (Rank 60)
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = U_school.shape[1]
    
    # Verify schoolbook (should always pass)
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        # Fallback to manual list construction if numpy fails (unlikely)
        pass

    # 2. Attempt to find a lower rank decomposition
    # We try ranks from 59 down to 50 (arbitrary limit, can be adjusted)
    # This is a heuristic search.
    if time.time() < deadline - 20:
        result = _search_lower_rank(n, m, p, best_rank - 1, 50, deadline - 10)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = best_U.shape[1]
    
    # Convert to lists for JSON serialization
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
