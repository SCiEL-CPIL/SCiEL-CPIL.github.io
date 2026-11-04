import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 36

    # Precompute target tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    TARGET_TENSOR = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                TARGET_TENSOR[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    def compute_error(U, V, W):
        """Compute squared error between decomposition and target."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        diff = T_approx - TARGET_TENSOR
        return np.sum(diff**2)

    def check_exact(U, V, W, tol=1e-9):
        """Check if decomposition is exact within tolerance."""
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(T_approx, TARGET_TENSOR, atol=tol)

    def try_rank_continuous(r, num_restarts=10, steps_per_restart=500):
        """Try to find a decomposition of rank r using continuous local search."""
        if time.time() > deadline:
            return None
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Initialize with small random values
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            current_error = compute_error(U, V, W)
            
            if current_error < 1e-12:
                return U, V, W
            
            # Simple gradient-free local search
            step_size = 0.05
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
                new_val = old_val + np.random.randn() * step_size
                
                mat[i, j] = new_val
                new_error = compute_error(U, V, W)
                
                if new_error < current_error:
                    current_error = new_error
                    if current_error < 1e-12:
                        return U, V, W
                else:
                    mat[i, j] = old_val
                    
                # Occasionally reduce step size
                if step % 50 == 0:
                    step_size *= 0.95
                    
        return None

    def try_rank_discrete(r, num_restarts=15, steps_per_restart=1500):
        """Try to find a decomposition of rank r using discrete local search."""
        if time.time() > deadline:
            return None
        
        # Scale target to allow integer solutions for half-integer U, V, W
        # If U, V, W are half-integers, 2U, 2V, 2W are integers.
        # Sum (2U)(2V)(2W) = 8 * Sum U V W = 8 * T
        SCALED_TARGET = 8 * TARGET_TENSOR
        
        def compute_scaled_error(U, V, W):
            T_approx = np.einsum('ar,br,cr->abc', U, V, W)
            diff = T_approx - SCALED_TARGET
            return np.sum(diff**2)
        
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
                
            current_error = compute_scaled_error(U, V, W)
            
            if current_error == 0:
                return (U / 2.0), (V / 2.0), (W / 2.0)
                
            temp = 5.0
            cooling_rate = 0.99
            
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
                
                new_error = compute_scaled_error(U, V, W)
                delta = new_error - current_error
                
                if delta <= 0 or np.random.rand() < np.exp(-delta / temp):
                    current_error = new_error
                    if current_error == 0:
                        return (U / 2.0), (V / 2.0), (W / 2.0)
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

    # Try to find a better solution
    # First try continuous search for slightly lower ranks
    for r in [35, 34, 33]:
        if time.time() > deadline - 15:
            break
        result = try_rank_continuous(r, num_restarts=8, steps_per_restart=300)
        if result is not None:
            U, V, W = result
            if check_exact(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Then try discrete search
    for r in [35, 34, 33, 32]:
        if time.time() > deadline - 10:
            break
        result = try_rank_discrete(r, num_restarts=10, steps_per_restart=1000)
        if result is not None:
            U, V, W = result
            if check_exact(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Return schoolbook if no better solution found
    return {"U": U_sb, "V": V_sb, "W": W_sb}
