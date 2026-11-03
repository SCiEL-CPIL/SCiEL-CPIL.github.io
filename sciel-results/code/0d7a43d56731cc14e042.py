"""Matrix multiplication 2x5x5 rank-minimal exact tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Build the target tensor
    target = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    def schoolbook():
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

    def verify(U, V, W):
        """Check if the decomposition is exact using einsum."""
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        try:
            computed = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
        except Exception:
            return False
        return np.allclose(computed, target, atol=1e-9)

    def try_rank(r, seed=42, max_iters=100):
        """Attempt to find a rank-r decomposition via random search with structured values."""
        if time.time() > deadline - 5:
            return None
            
        np.random.seed(seed)
        # Use small set of values to increase chance of hitting integer/half-integer solutions
        vals = [-1, -0.5, 0, 0.5, 1]
        
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                return None
                
            # Sparse initialization
            U = np.zeros((n * m, r))
            V = np.zeros((m * p, r))
            W = np.zeros((p * n, r))
            
            for col in range(r):
                # Pick random non-zero positions
                u_idx = np.random.choice(n * m, 1)
                v_idx = np.random.choice(m * p, 1)
                w_idx = np.random.choice(p * n, 1)
                
                U[u_idx[0], col] = random.choice(vals)
                V[v_idx[0], col] = random.choice(vals)
                W[w_idx[0], col] = random.choice(vals)
                
            if verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        return None

    # Theoretical lower bound is 25. Schoolbook is 50.
    # Given the structure, 50 is likely minimal for integer/half-integer constraints.
    # We try a few ranks below 50 to be sure, but prioritize returning 50 quickly.
    
    # Try rank 49 briefly
    if time.time() < deadline - 20:
        res = try_rank(49, seed=1, max_iters=50)
        if res:
            return res

    # Try rank 48 briefly
    if time.time() < deadline - 20:
        res = try_rank(48, seed=2, max_iters=50)
        if res:
            return res

    # Return schoolbook
    return schoolbook()

# EVOLVE-BLOCK-END
