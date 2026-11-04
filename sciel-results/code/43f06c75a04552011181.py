import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Precompute target tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    TARGET_TENSOR = np.zeros((n*m, m*p, p*n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                TARGET_TENSOR[i*m+j, j*p+k, k*n+i] = 1
                
    # Scale target to allow integer solutions for half-integer U, V, W
    # If U, V, W are half-integers, 2U, 2V, 2W are integers.
    # Sum (2U)(2V)(2W) = 8 * Sum U V W = 8 * T
    SCALED_TARGET = 8 * TARGET_TENSOR

    start_time = time.time()
    # Strict deadline to avoid timeout. Evaluator kills at 120s.
    deadline = start_time + 100.0 

    def compute_error(U, V, W):
        """Compute squared error between decomposition and scaled target."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx - SCALED_TARGET
        return np.sum(diff**2)

    def try_rank_discrete(r):
        """Try to find a decomposition of rank r using fast discrete local search."""
        if time.time() > deadline:
            return None
        
        # Reduced search effort to fit in time budget
        num_restarts = 20
        steps_per_restart = 2000
        initial_temp = 5.0
        cooling_rate = 0.99
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Initialize with sparse random integers
            U = np.zeros((n*m, r), dtype=int)
            V = np.zeros((m*p, r), dtype=int)
            W = np.zeros((p*n, r), dtype=int)
            
            density = 0.3
            for _ in range(int(n*m*r*density)):
                i, j = np.random.randint(n*m), np.random.randint(r)
                U[i, j] = np.random.choice([-2, -1, 1, 2])
            for _ in range(int(m*p*r*density)):
                i, j = np.random.randint(m*p), np.random.randint(r)
                V[i, j] = np.random.choice([-2, -1, 1, 2])
            for _ in range(int(p*n*r*density)):
                i, j = np.random.randint(p*n), np.random.randint(r)
                W[i, j] = np.random.choice([-2, -1, 1, 2])
                
            current_error = compute_error(U, V, W)
            
            if current_error == 0:
                return {"U": (U / 2.0).tolist(), "V": (V / 2.0).tolist(), "W": (W / 2.0).tolist()}
                
            temp = initial_temp
            
            for step in range(steps_per_restart):
                if time.time() > deadline - 5:
                    return None
                    
                # Pick a random matrix and entry to perturb
                mat_idx = np.random.randint(3)
                if mat_idx == 0:
                    mat = U
                    rows, cols = n*m, r
                elif mat_idx == 1:
                    mat = V
                    rows, cols = m*p, r
                else:
                    mat = W
                    rows, cols = p*n, r
                    
                i = np.random.randint(rows)
                j = np.random.randint(cols)
                
                old_val = mat[i, j]
                new_val = old_val + np.random.choice([-2, -1, 1, 2])
                new_val = np.clip(new_val, -10, 10)
                
                if new_val == old_val:
                    continue
                    
                mat[i, j] = new_val
                
                new_error = compute_error(U, V, W)
                delta = new_error - current_error
                
                if delta <= 0 or np.random.rand() < np.exp(-delta / temp):
                    current_error = new_error
                    if current_error == 0:
                        return {"U": (U / 2.0).tolist(), "V": (V / 2.0).tolist(), "W": (W / 2.0).tolist()}
                else:
                    mat[i, j] = old_val
                    
                temp *= cooling_rate
                
        return None

    # Schoolbook fallback (Always valid)
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Try to find a better solution quickly
    # We try ranks slightly below schoolbook to see if we can improve
    # If we fail, we return schoolbook.
    # To save time, we only try a few ranks with limited steps.
    for r in [32, 30, 28]:
        if time.time() > deadline - 10:
            break
        result = try_rank_discrete(r)
        if result is not None:
            return result

    # Return schoolbook if no better solution found
    return {"U": U_sb, "V": V_sb, "W": W_sb}
