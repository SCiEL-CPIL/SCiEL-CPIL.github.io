"""Search for a low-rank exact tensor decomposition of 2x3 by 3x7 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]
DEADLINE = 120.0
MARGIN = 5.0

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    # Shapes: U is (n*m, R), V is (m*p, R), W is (p*n, R)
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # Condition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    
    # 1. Try to find rank 39 via random search with integer/half-integer constraints
    # This is a hard search, so we limit the time.
    # A simple randomized local search might work if initialized well, but we need to be careful.
    # Given the time limit, a full exhaustive or even robust gradient search for rank 39 is risky.
    # However, the schoolbook rank is 42. We can try to see if we can shave off 3 ranks.
    
    # Let's try a few quick random attempts to find a rank < 42 solution.
    # If found, return it. Otherwise, return the schoolbook rank 42.
    
    def verify(U, V, W):
        # U, V, W are numpy arrays of shape (nm, R), (mp, R), (pn, R)
        # Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # This is a tensor contraction.
        # We can do this by reshaping and multiplying.
        # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
        # Let's just loop for simplicity since R is small.
        R = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(R):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-6:
                        return False
        return True

    def random_decomposition(R, seed=None):
        if seed is not None:
            np.random.seed(seed)
        # Initialize with small random integers or half-integers
        # A common trick is to start with random integers in {-1, 0, 1}
        U = np.random.randint(-1, 2, (n*m, R))
        V = np.random.randint(-1, 2, (m*p, R))
        W = np.random.randint(-1, 2, (p*n, R))
        return U, V, W

    # We'll try to find a rank 41 or 40 decomposition quickly.
    # If we can't find one in a few seconds, we fall back to 42.
    # Note: It is known that rank(2,3,7) <= 39, but finding it requires a specific construction or heavy search.
    # Given the constraints, a reliable quick search for 39 is unlikely.
    # However, maybe we can find 41 or 40?
    
    # Let's try a simple local search starting from random integer matrices.
    # We will try to minimize the error ||T - T_target||^2 where T_target is 1 everywhere.
    
    def local_search(U, V, W, max_iters=100):
        # Gradient descent on the error
        # Error E = sum_{i,j,k} (sum_r U_{ij,r} V_{jk,r} W_{ki,r} - 1)^2
        # We can't easily take gradients with integer constraints.
        # Instead, we'll do a simple "perturb and keep if better" search.
        best_error = float('inf')
        for _ in range(max_iters):
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            # Pick a random matrix, row, and column to perturb
            mat_idx = np.random.randint(0, 3)
            if mat_idx == 0:
                r, c = np.random.randint(0, n*m), np.random.randint(0, R)
                old_val = U[r, c]
                U[r, c] += np.random.choice([-1, 1])
            elif mat_idx == 1:
                r, c = np.random.randint(0, m*p), np.random.randint(0, R)
                old_val = V[r, c]
                V[r, c] += np.random.choice([-1, 1])
            else:
                r, c = np.random.randint(0, p*n), np.random.randint(0, R)
                old_val = W[r, c]
                W[r, c] += np.random.choice([-1, 1])
            
            # Calculate error
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for rr in range(R):
                            val += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                        err += (val - 1.0)**2
            if err < best_error:
                best_error = err
                if err < 1e-6:
                    return U, V, W, True
            else:
                # Revert
                if mat_idx == 0:
                    U[r, c] = old_val
                elif mat_idx == 1:
                    V[r, c] = old_val
                else:
                    W[r, c] = old_val
        return U, V, W, False

    # Try a few ranks
    for R in [41, 40, 39]:
        attempts = 5
        for a in range(attempts):
            if time.time() - start_time > DEADLINE - MARGIN:
                break
            U, V, W = random_decomposition(R, seed=42+a)
            # Quick check if it's already a solution (unlikely)
            if verify(U, V, W):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            # Local search
            U, V, W, found = local_search(U, V, W, max_iters=50)
            if found:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        if time.time() - start_time > DEADLINE - MARGIN:
            break

    # 2. Fallback: Schoolbook decomposition of rank 42
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
