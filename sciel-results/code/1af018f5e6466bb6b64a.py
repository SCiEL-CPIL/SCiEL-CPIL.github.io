"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    dim_U = n * m
    dim_V = m * p
    dim_W = p * n
    num_eqs = n * m * p
    
    def get_schoolbook():
        rank = num_eqs
        U = np.zeros((dim_U, rank), dtype=int)
        V = np.zeros((dim_V, rank), dtype=int)
        W = np.zeros((dim_W, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Precompute index arrays
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    
    U_rows = i_idx * m + j_idx
    V_rows = j_idx * p + k_idx
    W_rows = k_idx * n + i_idx
    
    def verify(U, V, W):
        U_sub = U[U_rows, :]
        V_sub = V[V_rows, :]
        W_sub = W[W_rows, :]
        prod = U_sub * V_sub * W_sub
        sums = np.sum(prod, axis=1)
        return np.allclose(sums, 1.0, atol=1e-9)
    
    def is_half_integer(mat):
        scaled = mat * 2
        rounded = np.round(scaled)
        return np.allclose(scaled, rounded, atol=1e-9)
    
    # Search for rank r
    # We search from 83 down to 70
    # Allocate more time to higher ranks (closer to 84)
    ranks_to_try = list(range(83, 79, -1)) + list(range(79, 69, -1))
    
    for r in ranks_to_try:
        if time.time() > deadline - 10:
            break
        
        # Number of tries
        if r > 79:
            num_tries = 300
        else:
            num_tries = 100
            
        for trial in range(num_tries):
            if time.time() > deadline - 2:
                break
                
            # Generate random U', V' with integer entries
            # Try different ranges
            if trial % 3 == 0:
                vals = [-1, 0, 1]
            elif trial % 3 == 1:
                vals = [-2, -1, 0, 1, 2]
            else:
                vals = [-1, 1] # Sparse
            
            U_prime = np.random.choice(vals, size=(dim_U, r))
            V_prime = np.random.choice(vals, size=(dim_V, r))
            
            # Build system A W' = b
            U_sub = U_prime[U_rows, :]
            V_sub = V_prime[V_rows, :]
            coeff = U_sub * V_sub # (num_eqs, r)
            
            A = np.zeros((num_eqs, dim_W * r))
            row_indices = np.arange(num_eqs)[:, None]
            col_indices = W_rows[:, None] * r + np.arange(r)[None, :]
            A[row_indices, col_indices] = coeff
            
            b = np.ones(num_eqs) * 8
            
            # Solve A W_flat = b
            try:
                W_flat, residuals, rank_sol, s = np.linalg.lstsq(A, b, rcond=None)
            except Exception:
                continue
                
            if len(residuals) > 0 and residuals[0] > 1e-6:
                continue
                
            W_prime = W_flat.reshape((dim_W, r))
            
            # Check if W_prime is close to integers
            W_rounded = np.round(W_prime)
            if np.max(np.abs(W_prime - W_rounded)) < 1e-6:
                # Check exactness with integers
                U_test = U_prime.astype(float) / 2.0
                V_test = V_prime.astype(float) / 2.0
                W_test = W_rounded.astype(float) / 2.0
                
                if verify(U_test, V_test, W_test):
                    if is_half_integer(U_test) and is_half_integer(V_test) and is_half_integer(W_test):
                        return {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}
                        
    return get_schoolbook()
# EVOLVE-BLOCK-END
