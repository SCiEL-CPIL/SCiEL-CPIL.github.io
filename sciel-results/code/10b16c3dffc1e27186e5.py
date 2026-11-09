import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _is_halfint(val, tol=1e-9):
    """Check if a value is an integer or half-integer."""
    v = val * 2
    return np.isclose(v, np.round(v), atol=tol)

def _clean_halfint(val, tol=1e-9):
    """Convert a value to the nearest half-integer."""
    v = val * 2
    return np.round(v) / 2.0

def _solve_subproblem(n, m, p, max_rank, min_rank, time_limit, seed_offset=0):
    """
    Finds a decomposition for n x m x p.
    Returns (U, V, W) as numpy arrays or None.
    """
    start_time = time.time()
    np.random.seed(seed_offset)
    
    # Try ranks from max_rank down to min_rank
    for r in range(max_rank, min_rank - 1, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Number of attempts per rank
        # For small problems, we can try many times
        max_attempts = 5000
        
        for _ in range(max_attempts):
            if time.time() - start_time > time_limit:
                break
            
            # Generate random U and V with entries in {-1, -0.5, 0, 0.5, 1}
            vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            
            U = np.random.choice(vals, size=(n*m, r))
            V = np.random.choice(vals, size=(m*p, r))
            
            num_eqs = n * m * p
            num_vars = p * n * r
            
            A = np.zeros((num_eqs, num_vars))
            b = np.ones(num_eqs)
            
            # Fill A
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r_idx in range(r):
                            u_val = U[i * m + j, r_idx]
                            v_val = V[j * p + k, r_idx]
                            if u_val == 0 or v_val == 0:
                                continue
                            var_idx = (k * n + i) * r + r_idx
                            A[eq_idx, var_idx] = u_val * v_val
            
            try:
                w, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            w_cleaned = np.array([_clean_halfint(val) for val in w])
            W = w_cleaned.reshape((p * n, r))
            
            # Verify
            valid = True
            # Use vectorized check for speed
            # Reshape U, V, W to 3D
            U3 = U.reshape(n, m, r)
            V3 = V.reshape(m, p, r)
            W3 = W.reshape(p, n, r)
            
            # Compute tensor product sum
            # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
            # We can do this by iterating r or using einsum
            # einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            try:
                T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                if not np.allclose(T, 1.0, atol=1e-9):
                    valid = False
            except Exception:
                valid = False
            
            if valid:
                return U, V, W

    return None

def construct():
    """
    Constructs the rank-minimal exact tensor decomposition for matrix multiplication.
    Uses the decomposition 7 = 3 + 4 to split the problem into 2x3x3 and 2x3x4.
    """
    n, m, p = DIMENSIONS
    p1, p2 = 3, 4
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Search for 2x3x3
    # Schoolbook rank for 2x3x3 is 18.
    # We try to find a lower rank.
    # Let's allocate 50% of time to each sub-problem, or adaptively.
    # Start with reasonable max ranks.
    max_rank_1 = 17
    min_rank_1 = 5
    max_rank_2 = 23
    min_rank_2 = 5
    
    # Time allocation
    time_for_1 = 50.0
    time_for_2 = 50.0
    
    # Solve 2x3x3
    res1 = _solve_subproblem(n, m, p1, max_rank_1, min_rank_1, time_for_1, seed_offset=42)
    
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    
    if res1 is None:
        # Fallback to schoolbook for part 1
        r1 = n * m * p1
        U1 = np.zeros((n * m, r1))
        V1 = np.zeros((m * p1, r1))
        W1 = np.zeros((p1 * n, r1))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p1):
                    U1[i * m + j, col] = 1
                    V1[j * p1 + k, col] = 1
                    W1[k * n + i, col] = 1
                    col += 1
    else:
        U1, V1, W1 = res1
        r1 = U1.shape[1]
        
    # Solve 2x3x4
    # Adjust time limit based on remaining time
    time_for_2 = max(0, remaining - 5.0)
    res2 = _solve_subproblem(n, m, p2, max_rank_2, min_rank_2, time_for_2, seed_offset=123)
    
    if res2 is None:
        # Fallback to schoolbook for part 2
        r2 = n * m * p2
        U2 = np.zeros((n * m, r2))
        V2 = np.zeros((m * p2, r2))
        W2 = np.zeros((p2 * n, r2))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p2):
                    U2[i * m + j, col] = 1
                    V2[j * p2 + k, col] = 1
                    W2[k * n + i, col] = 1
                    col += 1
    else:
        U2, V2, W2 = res2
        r2 = U2.shape[1]
        
    # Combine
    total_rank = r1 + r2
    
    U_total = np.zeros((n * m, total_rank))
    V_total = np.zeros((m * p, total_rank))
    W_total = np.zeros((p * n, total_rank))
    
    # Fill U
    U_total[:, :r1] = U1
    U_total[:, r1:] = U2
    
    # Fill V
    # V1 is for k in 0..p1-1
    # V2 is for k in p1..p-1 (mapped to 0..p2-1 in local coords)
    V_total[:m*p1, :r1] = V1
    # Map V2 to correct rows
    # V2 has shape (m, p2). We need to place it in V_total rows m*p1 to m*p
    # V2[j, k_local] corresponds to V_total[j, p1 + k_local]
    # So V_total[m*p1 + j*p2 + k_local, r1 + r_local] = V2[j, k_local]
    # Reshape V2 to (m, p2, r2)
    V2_3d = V2.reshape(m, p2, r2)
    # Expand to (m, p, r2) with zeros
    V2_full = np.zeros((m, p, r2))
    V2_full[:, p1:, :] = V2_3d
    # Flatten to (m*p, r2)
    V2_flat = V2_full.reshape(m * p, r2)
    V_total[:, r1:] = V2_flat
    
    # Fill W
    # W1 is for k in 0..p1-1
    # W2 is for k in p1..p-1
    # W1 shape (p1, n, r1) -> (p1*n, r1)
    # W2 shape (p2, n, r2) -> (p2*n, r2)
    W1_3d = W1.reshape(p1, n, r1)
    W2_3d = W2.reshape(p2, n, r2)
    
    # We need W_total shape (p*n, total_rank)
    # Rows 0..p1*n-1 correspond to k=0..p1-1
    # Rows p1*n..p*n-1 correspond to k=p1..p-1
    
    # W1 is already in the right row order (k*n+i)
    W_total[:p1*n, :r1] = W1
    
    # W2 needs to be placed in rows p1*n to p*n
    # W2_3d[k_local, i, r_local]
    # Target row: (p1 + k_local)*n + i
    # So we just need to reshape W2 to (p2*n, r2) which is already the case if W2 is (p2*n, r2)
    # Wait, _solve_subproblem returns W as (p*n, r).
    # So W2 is (p2*n, r2).
    # The row index in W2 is k_local*n + i.
    # The row index in W_total should be (p1 + k_local)*n + i = p1*n + k_local*n + i.
    # So we just place W2 in the bottom block.
    W_total[p1*n:, r1:] = W2
    
    # Convert to lists
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
# EVOLVE-BLOCK-END
