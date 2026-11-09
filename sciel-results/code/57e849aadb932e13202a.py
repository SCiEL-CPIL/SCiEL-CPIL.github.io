import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication M_{2,3,7}
    if no lower rank decomposition is found.
    The rank is n*m*p = 42.
    This implementation uses vectorized numpy operations to generate the factor matrices.
    Includes a deadline guard as per directive.
    """
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_sb = n * m * p
    
    # Generate the multi-dimensional grid of indices
    i_grid, j_grid, k_grid = np.indices((n, m, p))
    
    # Flatten the grids
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    k_flat = k_grid.flatten()
    
    # Initialize U, V, W for schoolbook
    U = np.zeros((n * m, rank_sb), dtype=np.int64)
    V = np.zeros((m * p, rank_sb), dtype=np.int64)
    W = np.zeros((p * n, rank_sb), dtype=np.int64)
    
    # Column indices
    col_indices = np.arange(rank_sb)
    
    # Row indices
    u_rows = i_flat * m + j_flat
    v_rows = j_flat * p + k_flat
    w_rows = k_flat * n + i_flat
    
    # Populate
    U[u_rows, col_indices] = 1
    V[v_rows, col_indices] = 1
    W[w_rows, col_indices] = 1
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank_sb

    def verify_decomposition(U, V, W):
        # Check if U, V, W exactly decompose the tensor
        # Tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # We need T[i, j, k] = 1 for all i, j, k.
        
        # Vectorized verification for speed
        # Reshape U, V, W to 3D
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        # We want T[i,j,k] = sum_r U_3d[i,j,r] * V_3d[j,k,r] * W_3d[k,i,r]
        # This is a bit complex to einsum directly without intermediates.
        # Let's compute M_ik[j, r] = U_3d[i,j,r] * V_3d[j,k,r]? No, k is in V.
        
        # Let's stick to the loop but make it slightly faster by pre-computing V*W products?
        # Or just use the loop, it's only 42 iterations.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integer(arr, tol=1e-6):
        # Snap entries to nearest half-integer if close
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    # Search for lower ranks
    # We'll try random U, V with entries in {-1, 0, 1} and solve for W.
    
    # Pre-calculate indices for speed
    indices_i = list(range(n))
    indices_k = list(range(p))
    indices_j = list(range(m))
    
    # Strategy:
    # 1. Try ranks from rank_sb - 1 down to 30.
    # 2. For each rank, try multiple random initializations of U and V.
    # 3. Solve for W using least squares for each (i, k) pair.
    # 4. Snap W to half-integers.
    # 5. Verify the full decomposition.
    
    # To improve chances of finding a solution, we can also try to optimize
    # the non-convex objective ||T(U,V,W) - 1||^2 using gradient-free methods
    # if random search fails, but given the time limit, we stick to the
    # algebraic approach which is more likely to find exact integer/half-integer
    # solutions if they exist for small rank reductions.
    
    # We will iterate ranks from rank_sb - 1 down to 35 (conservative to ensure time)
    # If time permits, we can go lower.
    
    min_rank_to_try = 35
    if time.time() < deadline - 60:
        min_rank_to_try = 30
        
    for rank in range(rank_sb - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
            
        max_trials = 50
        # Increase trials if we have lots of time left
        remaining_time = deadline - time.time()
        if remaining_time > 60:
            max_trials = 100
        elif remaining_time > 30:
            max_trials = 50
        else:
            max_trials = 10

        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            # Generate random U, V
            # Using a mix of sparse and denser matrices, now including half-integers
            if trial % 4 == 0:
                # Dense half-integers in [-1, 1]
                U = np.random.randint(-2, 3, size=(n * m, rank)) * 0.5
                V = np.random.randint(-2, 3, size=(m * p, rank)) * 0.5
            elif trial % 4 == 1:
                # Sparse half-integers in [-1, 1]
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                # Fill ~50% with random half-integers
                mask_u = np.random.rand(n * m, rank) < 0.5
                mask_v = np.random.rand(m * p, rank) < 0.5
                U[mask_u] = np.random.randint(-2, 3, size=(np.sum(mask_u),)) * 0.5
                V[mask_v] = np.random.randint(-2, 3, size=(np.sum(mask_v),)) * 0.5
            elif trial % 4 == 2:
                # Dense integers
                U = np.random.randint(-2, 3, size=(n * m, rank))
                V = np.random.randint(-2, 3, size=(m * p, rank))
            else:
                # Structured: few non-zeros per column, half-integers
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                for r in range(rank):
                    idx_u = np.random.choice(n*m, size=2, replace=False)
                    idx_v = np.random.choice(m*p, size=2, replace=False)
                    U[idx_u, r] = np.random.choice([-1, -0.5, 0.5, 1])
                    V[idx_v, r] = np.random.choice([-1, -0.5, 0.5, 1])
            
            W = np.zeros((p * n, rank))
            valid = True
            
            for i in indices_i:
                for k in indices_k:
                    # Build A_ik
                    # A_ik[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    # Vectorized construction
                    u_block = U[i * m : (i + 1) * m, :] # m x rank
                    # V rows are j*p+k. For fixed k, j varies 0..m-1.
                    # Rows are k, p+k, 2p+k ... (m-1)p+k.
                    v_block = V[k : m * p : p, :] # m x rank
                    
                    A_ik = u_block * v_block # m x rank
                    
                    # Solve A_ik @ w = ones
                    try:
                        w, residuals, rank_A, s = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                        # Check if it's a solution
                        if not np.allclose(A_ik @ w, np.ones(m), atol=1e-9):
                            valid = False
                            break
                        W[k * n + i, :] = w
                    except:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Snap W to half-integers
            W_snapped = snap_to_half_integer(W)
            if W_snapped is None:
                continue
                
            # Verify the whole decomposition with snapped W
            if verify_decomposition(U, V, W_snapped):
                best_U, best_V, best_W = U, V, W_snapped
                best_rank = rank
                break # Found a valid one for this rank, move to lower rank

    # Convert to lists of lists for return
    U_ret = best_U.tolist()
    V_ret = best_V.tolist()
    W_ret = best_W.tolist()

    return {
        "U": U_ret,
        "V": V_ret,
        "W": W_ret
    }
# EVOLVE-BLOCK-END
