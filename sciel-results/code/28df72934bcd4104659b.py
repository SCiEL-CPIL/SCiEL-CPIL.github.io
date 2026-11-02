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
    
    def is_valid_half_integer(val):
        # Check if val is close to a multiple of 0.5
        doubled = val * 2
        rounded = np.round(doubled)
        return np.isclose(doubled, rounded, atol=1e-9)

    def snap_to_half_integer(val):
        return np.round(val * 2) / 2.0

    def verify_decomposition(U, V, W):
        n, m, p = DIMENSIONS
        # U shape (n*m, r), V shape (m*p, r), W shape (p*n, r)
        r = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for l in range(r):
                        s += U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
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

    # Try to find a lower rank decomposition using randomized search with rounding
    # We search for U, V with small integer/half-integer values, solve for W,
    # check if W is close to integer/half-integer, snap, and verify.
    
    np.random.seed(42)
    
    # Ranks to try, from low to high
    ranks_to_try = list(range(24, 36))
    
    for r in ranks_to_try:
        if time.time() - start_time > deadline:
            break
            
        # Number of attempts for this rank
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline:
                break
                
            # Generate random U, V with entries from {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
            # This keeps the numbers small and increases the chance that W will be "nice"
            vals = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
            U_rand = np.random.choice(vals, size=(n * m, r))
            V_rand = np.random.choice(vals, size=(m * p, r))
            
            # Set up linear system for W
            # A_mat shape: (n*m*p, p*n*r)
            # b_vec shape: (n*m*p,)
            A_mat = np.zeros((n * m * p, p * n * r))
            b_vec = np.ones(n * m * p)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for l in range(r):
                            w_idx = (k * n + i) * r + l
                            A_mat[eq_idx, w_idx] = U_rand[i * m + j, l] * V_rand[j * p + k, l]
            
            # Solve for W
            try:
                W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                W_rand = W_flat.reshape(p * n, r)
            except np.linalg.LinAlgError:
                continue
            
            # Check if W_rand is close to half-integers
            # Also check U and V are already half-integers (they are by construction)
            w_doubled = W_rand * 2
            w_rounded = np.round(w_doubled)
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
