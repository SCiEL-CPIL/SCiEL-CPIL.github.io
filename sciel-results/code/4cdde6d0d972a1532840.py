import numpy as np
import time

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    target_rank = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep 30s buffer
    
    # Schoolbook solution as fallback
    def schoolbook():
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
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_result = schoolbook()
    
    # Try to find a lower rank decomposition
    # We search for R from target_rank - 1 down to target_rank - 10
    # If found, return it.
    
    # Precompute indices for the linear system
    # Equation for (i, j, k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let u_idx = i*m+j, v_idx = j*p+k, w_idx = k*n+i
    # Coefficient for W[w_idx, r] is U[u_idx, r] * V[v_idx, r]
    
    for R in range(target_rank - 1, max(target_rank - 15, 0), -1):
        if time.time() > deadline:
            break
            
        # Try a few random initializations
        for attempt in range(10):
            if time.time() > deadline:
                break
                
            # Random U, V
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            
            # Construct linear system A W_flat = b
            # W_flat has size (p*n)*R
            # A has size (n*m*p) x (p*n*R)
            # This might be too large to construct explicitly if R is large?
            # n*m*p = 96. p*n*R = 24*R.
            # 96 x (24*R) matrix. For R=95, 96 x 2280. Small.
            
            A = np.zeros((n * m * p, p * n * R))
            b = np.ones(n * m * p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        
                        # Row eq_idx of A
                        # For each r in 0..R-1:
                        # A[eq_idx, w_idx*R + r] = U[u_idx, r] * V[v_idx, r]
                        
                        # Vectorized
                        coeffs = U[u_idx, :] * V[v_idx, :]
                        start_col = w_idx * R
                        end_col = start_col + R
                        A[eq_idx, start_col:end_col] = coeffs
                        
                        eq_idx += 1
            
            # Solve A W_flat = b
            # Use lstsq
            try:
                W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(p * n, R)
                
                # Verify
                # Calculate sum_r U[u_idx, r] V[v_idx, r] W[w_idx, r]
                # This is element-wise product of (U * V) and W?
                # Let's check a few samples
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :])
                            if abs(val - 1.0) > 1e-5:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    
            except Exception:
                continue
                
    return best_result
# EVOLVE-BLOCK-END
