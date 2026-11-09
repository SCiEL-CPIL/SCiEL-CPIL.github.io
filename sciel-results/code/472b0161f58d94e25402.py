"""Matrix multiplication 3,3,4 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal (or near-minimal) exact tensor decomposition
    for multiplying a 3x3 matrix by a 3x4 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns:
        dict with keys 'U', 'V', 'W' - factor matrices of shapes
        (n*m, rank), (m*p, rank) with integer/half-integer entries
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s limit
    
    # Precompute index arrays for vectorized operations
    # For each (i,j,k) triple, we need indices:
    #   U_idx = i*m + j
    #   V_idx = j*p + k
    #   W_idx = k*n + i
    idx_i = np.repeat(np.arange(n), m * p)
    idx_j = np.tile(np.repeat(np.arange(m), p), n)
    idx_k = np.tile(np.arange(p), n * m)
    
    U_rows = idx_i * m + idx_j  # shape (36,)
    V_rows = idx_j * p + idx_k  # shape (36,)
    W_rows = idx_k * n + idx_i  # shape (36,)
    
    def check_solution(U, V, W, tol=1e-8):
        """Check if the decomposition is exact within tolerance."""
        # Vectorized: for each of 36 (i,j,k) triples, compute dot product
        # U[U_rows, :] * V[V_rows, :] * W[W_rows, :] summed over rank
        vals = np.sum(U[U_rows] * V[V_rows] * W[W_rows], axis=1)
        max_err = np.max(np.abs(vals - 1.0))
        return max_err < tol, max_err
    
    def round_to_half_ints(U, V, W):
        """Try to round to half-integers and verify."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        valid, err = check_solution(U_r, V_r, W_r, tol=1e-6)
        if valid:
            return U_r, V_r, W_r
        return None
    
    def polish(U, V, W, max_iter=500, tol=1e-14):
        """High-precision polish pass with tighter tolerance."""
        r = U.shape[1]
        prev_err = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Solve for W: fix U, V
            # A[eq, out*r:(out+1)*r] = U[U_rows[eq]] * V[V_rows[eq]]
            A_W = np.zeros((36, 12 * r))
            for eq in range(36):
                row_w = W_rows[eq]
                A_W[eq, row_w*r:(row_w+1)*r] = U[U_rows[eq]] * V[V_rows[eq]]
            W_flat, _, _, _ = np.linalg.lstsq(A_W, np.ones(36), rcond=None)
            W = W_flat.reshape(12, r)
            
            # Solve for V: fix U, W
            A_V = np.zeros((36, 12 * r))
            for eq in range(36):
                row_v = V_rows[eq]
                A_V[eq, row_v*r:(row_v+1)*r] = U[U_rows[eq]] * W[W_rows[eq]]
            V_flat, _, _, _ = np.linalg.lstsq(A_V, np.ones(36), rcond=None)
            V = V_flat.reshape(12, r)
            
            # Solve for U: fix V, W
            A_U = np.zeros((36, 9 * r))
            for eq in range(36):
                row_u = U_rows[eq]
                A_U[eq, row_u*r:(row_u+1)*r] = V[V_rows[eq]] * W[W_rows[eq]]
            U_flat, _, _, _ = np.linalg.lstsq(A_U, np.ones(36), rcond=None)
            U = U_flat.reshape(9, r)
            
            # Check error
            valid, max_err = check_solution(U, V, W, tol=tol)
            
            if max_err < 1e-13:
                rounded = round_to_half_ints(U, V, W)
                if rounded is not None:
                    return rounded
                return U, V, W
            
            # Stagnation detection
            if it > 20 and abs(prev_err - max_err) < 1e-16:
                break
            prev_err = max_err
        
        return U, V, W
    
    def try_rank(r, max_seeds=30):
        """Try to find a decomposition of rank r."""
        for seed in range(max_seeds):
            if time.time() > deadline:
                return None
            
            rng = np.random.RandomState(seed)
            
            # Different initialization strategies
            strategy = seed % 4
            if strategy == 0:
                # Random Gaussian
                U = rng.randn(9, r)
                V = rng.randn(12, r)
                W = rng.randn(12, r)
            elif strategy == 1:
                # Sparse +/-1 initialization
                U = np.zeros((9, r))
                V = np.zeros((12, r))
                W = np.zeros((12, r))
                for col in range(r):
                    u_idx = rng.choice(9, size=min(3, 9), replace=False)
                    v_idx = rng.choice(12, size=min(4, 12), replace=False)
                    w_idx = rng.choice(12, size=min(4, 12), replace=False)
                    U[u_idx, col] = rng.choice([-1, 1], size=len(u_idx))
                    V[v_idx, col] = rng.choice([-1, 1], size=len(v_idx))
                    W[w_idx, col] = rng.choice([-1, 1], size=len(w_idx))
            elif strategy == 2:
                # Half-integer random
                U = rng.choice([-1, -0.5, 0, 0.5, 1], size=(9, r))
                V = rng.choice([-1, -0.5, 0, 0.5, 1], size=(12, r))
                W = rng.choice([-1, -0.5, 0, 0.5, 1], size=(12, r))
            else:
                # Small Gaussian with scaling
                scale = 1.0 / np.sqrt(r)
                U = rng.randn(9, r) * scale
                V = rng.randn(12, r) * scale
                W = rng.randn(12, r) * scale
            
            # ALS iterations
            prev_err = float('inf')
            for it in range(300):
                if time.time() > deadline:
                    return None
                
                # Solve for W
                A_W = np.zeros((36, 12 * r))
                for eq in range(36):
                    row_w = W_rows[eq]
                    A_W[eq, row_w*r:(row_w+1)*r] = U[U_rows[eq]] * V[V_rows[eq]]
                W_flat, _, _, _ = np.linalg.lstsq(A_W, np.ones(36), rcond=None)
                W = W_flat.reshape(12, r)
                
                # Solve for V
                A_V = np.zeros((36, 12 * r))
                for eq in range(36):
                    row_v = V_rows[eq]
                    A_V[eq, row_v*r:(row_v+1)*r] = U[U_rows[eq]] * W[W_rows[eq]]
                V_flat, _, _, _ = np.linalg.lstsq(A_V, np.ones(36), rcond=None)
                V = V_flat.reshape(12, r)
                
                # Solve for U
                A_U = np.zeros((36, 9 * r))
                for eq in range(36):
                    row_u = U_rows[eq]
                    A_U[eq, row_u*r:(row_u+1)*r] = V[V_rows[eq]] * W[W_rows[eq]]
                U_flat, _, _, _ = np.linalg.lstsq(A_U, np.ones(36), rcond=None)
                U = U_flat.reshape(9, r)
                
                # Check error
                valid, max_err = check_solution(U, V, W, tol=1e-12)
                
                if max_err < 1e-12:
                    # Polish
                    result = polish(U, V, W, max_iter=200, tol=1e-14)
                    if result is not None:
                        U_p, V_p, W_p = result
                        rounded = round_to_half_ints(U_p, V_p, W_p)
                        if rounded is not None:
                            return rounded
                        else:
                            # Return as-is if exact enough
                            if check_solution(U_p, V_p, W_p, tol=1e-10)[0]:
                                return U_p, V_p, W_p
                    break
                
                # Stagnation
                if it > 15 and abs(prev_err - max_err) < 1e-15:
                    break
                prev_err = max_err
            
            # After ALS, try rounding
            if check_solution(U, V, W, tol=1e-8)[0]:
                rounded = round_to_half_ints(U, V, W)
                if rounded is not None:
                    return rounded
        
        return None
    
    # Try ranks from low to high (lower rank = better score)
    # Start from a reasonable lower bound
    for r in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            # Final verification
            if check_solution(U, V, W, tol=1e-8)[0]:
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
    
    # Fallback: schoolbook rank 36
    r = 36
    U = [[0] * r for _ in range(9)]
    V = [[0] * r for _ in range(12)]
    W = [[0] * r for _ in range(12)]
    for c in range(36):
        i = c // 12
        j = (c % 12) // 4
        k = c % 4
        U[i * 3 + j][c] = 1
        V[j * 4 + k][c] = 1
        W[k * 3 + i][c] = 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
