"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 3x4 by 4x5 matrix multiplication.
    
    Strategy:
    1. Attempt a short randomized local search for a rank 59 decomposition.
    2. If not found, return the schoolbook decomposition (rank 60).
    """
    n, m, p = DIMENSIONS
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def try_search_rank59(time_limit=10.0):
        """
        Attempts to find a rank 59 decomposition using local search.
        Returns None if not found.
        """
        target_rank = 59
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        # Generate the target tensor T (flattened)
        # T[a,b,c] = 1 if A[a] and B[b] contribute to C[c]
        # a = i*m + j, b = j*p + k, c = k*n + i
        T = np.zeros((size_u, size_v, size_w), dtype=np.int8)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1

        # Initialize U, V, W with random values
        # We use a mix of integers and half-integers
        # Start with a random initialization close to schoolbook?
        # Or just random?
        
        U = np.zeros((size_u, target_rank), dtype=np.float64)
        V = np.zeros((size_v, target_rank), dtype=np.float64)
        W = np.zeros((size_w, target_rank), dtype=np.float64)
        
        # Initialize with schoolbook-like structure but with noise?
        # Actually, let's just start with random small integers
        for r in range(target_rank):
            U[:, r] = np.random.choice([-1, 0, 1], size=size_u)
            V[:, r] = np.random.choice([-1, 0, 1], size=size_v)
            W[:, r] = np.random.choice([-1, 0, 1], size=size_w)
            
        start_time = time.time()
        best_error = float('inf')
        best_U = U.copy()
        best_V = V.copy()
        best_W = W.copy()
        
        def calc_error(U, V, W):
            # Compute sum_r U[:,r] V[:,r] W[:,r]
            # This is a tensor product sum
            # Efficient way: 
            # Result[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
            # We can do this by matrix multiplications
            # Temp[a,b] = sum_r U[a,r] V[b,r]
            # Result[a,b,c] = sum_r Temp[a,b] W[c,r] ? No.
            # Result[a,b,c] = sum_r (U[a,r] * V[b,r]) * W[c,r]
            
            # Let's just loop for simplicity in the search step, 
            # but it's slow. 
            # Better: 
            # AV = U @ V.T  (size_u x size_v)
            # This gives sum_r U[a,r] V[b,r]
            # But we need to multiply by W[c,r] and sum over r.
            # This is not a simple matrix product.
            
            # We can compute it as:
            # For each r, add U[:,r] * V[:,r] * W[:,r] to the tensor.
            # This is O(R * size_u * size_v * size_w) which is huge.
            
            # Alternative: 
            # The error is || T - Sum(U_r V_r W_r) ||^2
            # We can compute the current tensor approximation.
            
            approx = np.zeros((size_u, size_v, size_w), dtype=np.float64)
            for r in range(target_rank):
                u = U[:, r]
                v = V[:, r]
                w = W[:, r]
                # Outer product of u and v, then scale by w?
                # approx[a,b,c] += u[a] * v[b] * w[c]
                # This is u[:,None,None] * v[None,:,None] * w[None,None,:]
                approx += np.einsum('a,b,c->abc', u, v, w)
                
            error = np.sum((T - approx) ** 2)
            return error

        # Local search
        steps = 0
        while time.time() - start_time < time_limit:
            steps += 1
            # Pick a random variable to perturb
            mat = random.choice(['U', 'V', 'W'])
            r = random.randint(0, target_rank - 1)
            if mat == 'U':
                idx = random.randint(0, size_u - 1)
                old_val = U[idx, r]
                delta = random.choice([-0.5, 0.5, -1, 1])
                new_val = old_val + delta
                # Snap to half-integer
                new_val = round(new_val * 2) / 2
                U[idx, r] = new_val
                error = calc_error(U, V, W)
                if error < best_error:
                    best_error = error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if best_error == 0:
                        return {
                            "U": best_U.tolist(),
                            "V": best_V.tolist(),
                            "W": best_W.tolist()
                        }
                else:
                    # Revert if not improved (greedy)
                    # Or accept with probability (simulated annealing)
                    # For simplicity, revert
                    U[idx, r] = old_val
            elif mat == 'V':
                idx = random.randint(0, size_v - 1)
                old_val = V[idx, r]
                delta = random.choice([-0.5, 0.5, -1, 1])
                new_val = old_val + delta
                new_val = round(new_val * 2) / 2
                V[idx, r] = new_val
                error = calc_error(U, V, W)
                if error < best_error:
                    best_error = error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if best_error == 0:
                        return {
                            "U": best_U.tolist(),
                            "V": best_V.tolist(),
                            "W": best_W.tolist()
                        }
                else:
                    V[idx, r] = old_val
            else: # W
                idx = random.randint(0, size_w - 1)
                old_val = W[idx, r]
                delta = random.choice([-0.5, 0.5, -1, 1])
                new_val = old_val + delta
                new_val = round(new_val * 2) / 2
                W[idx, r] = new_val
                error = calc_error(U, V, W)
                if error < best_error:
                    best_error = error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if best_error == 0:
                        return {
                            "U": best_U.tolist(),
                            "V": best_V.tolist(),
                            "W": best_W.tolist()
                        }
                else:
                    W[idx, r] = old_val

        return None

    # Try search
    result = try_search_rank59(5.0)
    if result:
        return result
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
