import numpy as np
import time
import random

def construct():
    n, m, p = 2, 3, 4
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i, j, r] * V[j, k, r] * W[k, i, r] == 1
    
    def check_decomposition(U, V, W, R):
        """
        U: (n, m, R)
        V: (m, p, R)
        W: (p, n, R)
        Returns error sum of squares.
        """
        # Reshape for einsum
        # U[i, j, r], V[j, k, r], W[k, i, r] -> T[i, j, k]
        # Note: W is indexed by k, i in the formula sum_r ... W[k, i, r]
        # In the standard interface, W is (p*n, R).
        # W_flat[k*n + i, r] corresponds to W_tensor[k, i, r].
        
        # We will work with flattened arrays for the interface, 
        # but internal logic uses 3D for clarity if needed.
        # Actually, let's just use the flattened logic for speed and simplicity in the loop.
        
        # U_flat: (n*m, R) -> U[i*m+j, r]
        # V_flat: (m*p, R) -> V[j*p+k, r]
        # W_flat: (p*n, R) -> W[k*n+i, r]
        
        total_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for r in range(R):
                        val += U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
                    # Target is 1
                    diff = val - 1
                    total_err += diff * diff
        return total_err

    def try_find_decomposition(R, max_iter=50000):
        # Initialize with small random integers
        # Using integers to avoid float issues initially
        U = np.random.randint(-2, 3, size=(n*m, R))
        V = np.random.randint(-2, 3, size=(m*p, R))
        W = np.random.randint(-2, 3, size=(p*n, R))
        
        current_err = check_decomposition(U, V, W, R)
        if current_err == 0:
            return U, V, W
            
        # Local search
        # We will perturb one element at a time
        # Precompute indices for faster access?
        # Given small sizes, direct access is fine.
        
        for it in range(max_iter):
            # Check time
            if time.time() > deadline:
                return None, None, None
                
            # Pick random element to perturb
            # Choose which matrix: U, V, or W
            mat_idx = random.randint(0, 2)
            if mat_idx == 0:
                row = random.randint(0, n*m - 1)
                col = random.randint(0, R - 1)
                delta = random.choice([-1, 1])
                old_val = U[row, col]
                U[row, col] = old_val + delta
            elif mat_idx == 1:
                row = random.randint(0, m*p - 1)
                col = random.randint(0, R - 1)
                delta = random.choice([-1, 1])
                old_val = V[row, col]
                V[row, col] = old_val + delta
            else:
                row = random.randint(0, p*n - 1)
                col = random.randint(0, R - 1)
                delta = random.choice([-1, 1])
                old_val = W[row, col]
                W[row, col] = old_val + delta
                
            new_err = check_decomposition(U, V, W, R)
            
            # Accept if better, or with small probability if worse (simulated annealing lite)
            if new_err <= current_err or (new_err > current_err and random.random() < 0.05):
                current_err = new_err
                if current_err == 0:
                    return U, V, W
            else:
                # Revert
                if mat_idx == 0:
                    U[row, col] = old_val
                elif mat_idx == 1:
                    V[row, col] = old_val
                else:
                    W[row, col] = old_val
                    
        return None, None, None

    # Try ranks from 20 upwards
    # Known lower bound is 12, but 20 is a strong candidate for 2x3x4
    # If 20 fails, try 21, 22, 23
    for R in range(20, 25):
        if time.time() > deadline:
            break
        U, V, W = try_find_decomposition(R, max_iter=20000)
        if U is not None:
            # Convert to lists of integers/floats as required
            # The problem allows integer/half-integer. Integers are fine.
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook rank 24
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
