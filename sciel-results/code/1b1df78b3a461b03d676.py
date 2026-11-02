"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Returns dict with U, V, W matrices.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    def verify_decomposition(U, V, W):
        n, m, p = DIMENSIONS
        r = U.shape[1]
        # Vectorized verification for speed
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # We need sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] == 1
        # This is a tensor contraction.
        # We can check each (i,j,k) tuple.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # Sum over l
                    s = np.dot(U[u_idx, :], np.multiply(V[v_idx, :], W[w_idx, :]))
                    if not np.isclose(s, 1.0, atol=1e-9):
                        return False
        return True

    def get_schoolbook():
        rank = n * m * p
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    np.random.seed(42)
    
    # Prioritize ranks likely to be optimal or near-optimal
    # Known: R(3,3,3)=23. R(3,3,4) is likely 24 or 26.
    # Search order: 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35
    ranks_to_try = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    # Define value sets for random initialization
    # Small integers are more likely to yield exact solutions
    vals_int = [-2, -1, 0, 1, 2]
    vals_half = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
    
    for r in ranks_to_try:
        if time.time() - start_time > deadline:
            break
            
        # Allocate more time to lower ranks
        if r <= 26:
            max_attempts = 1000
        elif r <= 30:
            max_attempts = 500
        else:
            max_attempts = 200
            
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline:
                break
                
            # Generate random U, V
            # Use integer values primarily, as they are more common in exact decompositions
            if attempt % 3 == 0:
                U_rand = np.random.choice(vals_int, size=(n * m, r))
                V_rand = np.random.choice(vals_int, size=(m * p, r))
            elif attempt % 3 == 1:
                U_rand = np.random.choice(vals_half, size=(n * m, r))
                V_rand = np.random.choice(vals_half, size=(m * p, r))
            else:
                # Sparse initialization
                U_rand = np.zeros((n * m, r))
                V_rand = np.zeros((m * p, r))
                # Ensure some non-zeros
                for _ in range(r):
                    U_rand[np.random.randint(n*m), np.random.randint(r)] = np.random.choice([-1, 1, 2])
                    V_rand[np.random.randint(m*p), np.random.randint(r)] = np.random.choice([-1, 1, 2])
            
            # Build A_mat efficiently
            # A_mat[eq_idx, w_idx] = U[i*m+j, l] * V[j*p+k, l]
            # eq_idx = i*m*p + j*p + k
            # w_idx = (k*n+i)*r + l
            
            A_mat = np.zeros((n * m * p, p * n * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    u_row = U_rand[i * m + j, :]
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        v_row = V_rand[j * p + k, :]
                        prod = u_row * v_row
                        w_base = (k * n + i) * r
                        A_mat[eq_idx, w_base:w_base+r] = prod
            
            # Solve for W
            try:
                W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_rand = W_flat.reshape(p * n, r)
            except np.linalg.LinAlgError:
                continue
            
            # Check if W_rand is close to half-integers
            w_doubled = W_rand * 2
            w_rounded = np.round(w_doubled)
            
            # Check closeness
            if not np.allclose(w_doubled, w_rounded, atol=1e-6):
                continue
            
            # Snap to half-integers
            W_snapped = w_rounded / 2.0
            
            # Verify exactness with snapped values
            if verify_decomposition(U_rand, V_rand, W_snapped):
                return {"U": U_rand.tolist(), "V": V_rand.tolist(), "W": W_snapped.tolist()}
            
    # Fallback to schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
