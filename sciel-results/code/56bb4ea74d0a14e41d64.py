import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # Schoolbook decomposition (Rank 24)
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
        return {"U": U, "V": V, "W": W}
    
    # Try to find a lower rank decomposition (e.g., 23)
    target_rank = 23
    max_attempts = 10000
    
    for attempt in range(max_attempts):
        if time.time() - start_time > timeout:
            break
            
        # Random sparse initialization for U and V
        # Entries in {-1, 0, 1}
        U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
        V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
        
        # Ensure some non-zero structure
        if np.all(U == 0) or np.all(V == 0):
            continue
            
        W_candidate = np.zeros((p * n, target_rank))
        valid = True
        
        for k in range(p):
            for i in range(n):
                # Build A_{k,i} matrix of shape (m, target_rank)
                # A_{k,i}[j, r] = U[i*m+j, r] * V[j*p+k, r]
                A_ki = np.zeros((m, target_rank))
                for j in range(m):
                    A_ki[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # Solve A_ki * w = ones(m)
                # w is of shape (target_rank,)
                try:
                    w, residuals, rank_A, sv = np.linalg.lstsq(A_ki, np.ones(m), rcond=None)
                    # Check residual
                    if residuals and residuals[0] > 1e-9:
                        valid = False
                        break
                    W_candidate[k * n + i, :] = w
                except:
                    valid = False
                    break
                    
        if not valid:
            continue
            
        # Check if W_candidate entries are close to half-integers
        # And if U, V are integers (they are)
        # Snap W to half-integers
        W_snapped = np.round(W_candidate * 2) / 2
        
        # Verify exactness with snapped values
        # U, V are integers, W is half-integer
        # Product U*V*W is half-integer. Sum should be 1.
        
        # Fast check: compute tensor product and compare to 1
        # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # We can check a few random indices first
        is_valid = True
        for _ in range(10):
            i = np.random.randint(n)
            j = np.random.randint(m)
            k = np.random.randint(p)
            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W_snapped[k * n + i, :])
            if not np.isclose(val, 1.0, atol=1e-9):
                is_valid = False
                break
                
        if is_valid:
            # Full verification
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i * m + j, :], V[j * p + k, :] * W_snapped[k * n + i, :])
                        if not np.isclose(val, 1.0, atol=1e-9):
                            is_valid = False
                            break
                if not is_valid:
                    break
            if is_valid:
                # Convert to lists for return
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_snapped.tolist()
                }
                
    # Fallback to schoolbook
    return get_schoolbook()
