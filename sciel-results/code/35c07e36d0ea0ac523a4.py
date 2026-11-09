import numpy as np
import time

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix by a 3 by 7 matrix.
    Falls back to schoolbook rank 42 if no lower rank is found within the time limit.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_sb = n * m * p  # 42
    
    # 1. Generate Schoolbook Decomposition
    U_sb = np.zeros((n * m, rank_sb), dtype=np.int64)
    V_sb = np.zeros((m * p, rank_sb), dtype=np.int64)
    W_sb = np.zeros((p * n, rank_sb), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb

    def verify_decomposition(U, V, W):
        """
        Checks if sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1 for all i,j,k.
        """
        # Reshape for easier access
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R)
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        
        R = U.shape[1]
        
        for i in range(n):
            for k in range(p):
                # W_3d[k, i, :] is the vector for W[k*n+i, :]
                w_vec = W_3d[k, i, :] # shape (R,)
                for j in range(m):
                    # V_3d[j, k, :] is the vector for V[j*p+k, :]
                    v_vec = V_3d[j, k, :] # shape (R,)
                    # U_3d[i, j, :] is the vector for U[i*m+j, :]
                    u_vec = U_3d[i, j, :] # shape (R,)
                    
                    # Compute dot product sum_r u_vec[r] * v_vec[r] * w_vec[r]
                    # Element-wise product first
                    prod = u_vec * v_vec * w_vec
                    val = np.sum(prod)
                    
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_integer(arr, tol=1e-6):
        snapped = np.round(arr * 2) / 2
        if np.allclose(arr, snapped, atol=tol):
            return snapped
        return None

    # 2. Search for lower ranks
    # We prioritize ranks 41, 40, 39...
    # We use a mix of random initialization and structured initialization.
    
    min_rank_to_try = 35
    if time.time() < deadline - 30:
        min_rank_to_try = 30
        
    for rank in range(rank_sb - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
            
        # Allocate more trials for higher ranks (closer to schoolbook)
        # as they are more likely to exist if any do.
        remaining_time = deadline - time.time()
        if rank >= rank_sb - 3:
            max_trials = 200
        elif rank >= rank_sb - 10:
            max_trials = 100
        else:
            max_trials = 50
            
        # Adjust trials based on remaining time
        if remaining_time < 10:
            max_trials = 10

        for trial in range(max_trials):
            if time.time() > deadline:
                break
                
            # Generate random U, V
            # Use integers and half-integers
            # Strategy: Sparse matrices are more likely to yield integer W solutions
            if trial % 3 == 0:
                # Sparse integers
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                # Fill ~30% of entries with random integers {-1, 1}
                mask_u = np.random.rand(n * m, rank) < 0.3
                mask_v = np.random.rand(m * p, rank) < 0.3
                U[mask_u] = np.random.choice([-1, 1], size=np.sum(mask_u))
                V[mask_v] = np.random.choice([-1, 1], size=np.sum(mask_v))
            elif trial % 3 == 1:
                # Dense integers
                U = np.random.randint(-1, 2, size=(n * m, rank))
                V = np.random.randint(-1, 2, size=(m * p, rank))
            else:
                # Structured: Ensure each column of U and V has at least one non-zero
                U = np.zeros((n * m, rank))
                V = np.zeros((m * p, rank))
                for r in range(rank):
                    # Pick random rows
                    idx_u = np.random.choice(n*m, size=1)
                    idx_v = np.random.choice(m*p, size=1)
                    U[idx_u, r] = np.random.choice([-1, 1])
                    V[idx_v, r] = np.random.choice([-1, 1])
                    # Add a second non-zero to increase connectivity
                    if np.random.rand() > 0.5:
                        idx_u2 = np.random.choice(n*m, size=1)
                        idx_v2 = np.random.choice(m*p, size=1)
                        U[idx_u2, r] = np.random.choice([-1, 1])
                        V[idx_v2, r] = np.random.choice([-1, 1])
            
            W = np.zeros((p * n, rank))
            valid = True
            
            # Solve for W
            # For each i, k, we have a linear system A_ik @ w = 1
            # A_ik[j, r] = U[i*m+j, r] * V[j*p+k, r]
            
            for i in range(n):
                for k in range(p):
                    # Construct A_ik (m x rank)
                    # Rows of A_ik correspond to j
                    # Col r of A_ik is U[i*m+j, r] * V[j*p+k, r] for j=0..m-1
                    
                    A_ik = np.zeros((m, rank))
                    for j in range(m):
                        A_ik[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A_ik @ w = ones(m)
                    # Use lstsq
                    try:
                        w, residuals, rank_A, s = np.linalg.lstsq(A_ik, np.ones(m), rcond=None)
                        # Check residual
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
                
            # Verify
            if verify_decomposition(U, V, W_snapped):
                best_U, best_V, best_W = U, V, W_snapped
                best_rank = rank
                break # Found valid one for this rank, try lower

    # Convert to lists
    U_ret = best_U.tolist()
    V_ret = best_V.tolist()
    W_ret = best_W.tolist()

    return {
        "U": U_ret,
        "V": V_ret,
        "W": W_ret
    }
# EVOLVE-BLOCK-END
