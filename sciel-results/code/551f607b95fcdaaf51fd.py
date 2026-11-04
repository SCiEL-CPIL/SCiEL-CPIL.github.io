import numpy as np
import time
import random

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    def check_decomposition(U, V, W, R):
        # U: (n, m, R), V: (m, p, R), W: (p, n, R)
        T = np.einsum('ijr,lkr,ikir->ijkl', U, V, W)
        target = np.zeros((n, m, m, p), dtype=int)
        for i in range(n):
            for k in range(p):
                target[i, :, :, k] = np.eye(m, dtype=int)
        return np.array_equal(T, target)

    def try_find_decomposition(R, time_limit=10):
        start_time = time.time()
        
        for attempt in range(50000):
            if time.time() - start_time > time_limit:
                return None
            
            # Random sparse U, V
            U_flat = np.array([random.choice([-1, 0, 0, 0, 1]) for _ in range(n * m * R)])
            V_flat = np.array([random.choice([-1, 0, 0, 0, 1]) for _ in range(m * p * R)])
            
            U = U_flat.reshape(n, m, R)
            V = V_flat.reshape(m, p, R)
            
            W_flat = np.zeros(p * n * R, dtype=int)
            success = True
            
            for i in range(n):
                for k in range(p):
                    # Construct A (m*m x R) and b (m*m)
                    A = np.zeros((m * m, R), dtype=float)
                    b = np.zeros(m * m, dtype=float)
                    
                    for j in range(m):
                        for l in range(m):
                            # A[j*m+l, r] = U[i, j, r] * V[l, k, r]
                            A[j * m + l, :] = U[i, j, :] * V[l, k, :]
                            if j == l:
                                b[j * m + l] = 1.0
                                
                    # Solve A w = b
                    # Use least squares
                    try:
                        w, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        success = False
                        break
                        
                    # Check if residual is close to 0
                    if residuals is not None and residuals[0] > 1e-9:
                        success = False
                        break
                        
                    # Check if w can be integer
                    # Round w to nearest integer
                    w_int = np.rint(w)
                    # Check if A w_int == b
                    if not np.allclose(A @ w_int, b, atol=1e-9):
                        # Maybe half-integers?
                        w_half = np.rint(w * 2) / 2
                        if not np.allclose(A @ w_half, b, atol=1e-9):
                            success = False
                            break
                        else:
                            # Store half-integers? 
                            # Task allows half-integers.
                            # But we need to store in W_flat.
                            # If we use half-integers, we might mix types.
                            # Let's stick to integers for simplicity first.
                            # If half-integers work, we can scale everything by 2?
                            # No, U and V are integers.
                            # If W has half-integers, the product U*V*W will have half-integers.
                            # The sum must be integer (0 or 1).
                            # So half-integers are allowed.
                            # We need to store them.
                            # The return type is lists of numbers.
                            # Let's try to find integer solutions first.
                            pass
                            
                    if success:
                        # Store w in W_flat
                        # Index for W is k*n + i, so W[k, i, :]
                        # W_flat index: (k*n + i) * R + r
                        base_idx = (k * n + i) * R
                        W_flat[base_idx : base_idx + R] = w_int if np.allclose(A @ w_int, b, atol=1e-9) else w_half

            if success:
                W = W_flat.reshape(p, n, R)
                # Final verification
                if check_decomposition(U, V, W, R):
                    return {
                        "U": U_flat.reshape(n * m, R).tolist(),
                        "V": V_flat.reshape(m * p, R).tolist(),
                        "W": W_flat.reshape(p * n, R).tolist()
                    }

        return None

    # Try rank 11
    result = try_find_decomposition(11, time_limit=30)
    if result:
        return result
        
    # Try rank 12
    result = try_find_decomposition(12, time_limit=30)
    if result:
        return result

    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
# EVOLVE-BLOCK-END
