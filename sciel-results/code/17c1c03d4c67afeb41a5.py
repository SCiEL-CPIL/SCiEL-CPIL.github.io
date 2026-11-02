import numpy as np
import time
import random

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (Rank 30)
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U_school[u_idx][col] = 1
                V_school[v_idx][col] = 1
                W_school[w_idx][col] = 1
                col += 1
                
    # Precompute target tensor values
    # Target(u, v, w) is 1 if indices are consistent, else 0
    # u = i*m + j, v = j'*p + k, w = k'*n + i'
    # Consistent if i==i', j==j', k==k'
    target = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                target[u, v, w] = 1

    # Try to find a lower rank decomposition (Rank 29)
    target_rank = rank_school - 1
    deadline = time.time() + 110  # Safety margin
    
    # Define a function to check if a decomposition is valid
    def is_valid(U, V, W, R):
        # U, V, W are numpy arrays of shape (dim, R)
        # Compute tensor contraction
        # T[u, v, w] = sum_r U[u, r] * V[v, r] * W[w, r]
        # We can do this efficiently using einsum or loops
        # Since R is small (29), loops are fine
        T = np.zeros((n * m, m * p, p * n))
        for r in range(R):
            T += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        
        # Check against target
        # Allow small floating point error, then round
        if np.allclose(T, target, atol=1e-6):
            # Try to round to integers or half-integers
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify exactness with rounded values
            T_exact = np.zeros((n * m, m * p, p * n))
            for r in range(R):
                T_exact += np.outer(np.outer(U_rounded[:, r], V_rounded[:, r]), W_rounded[:, r])
            
            if np.allclose(T_exact, target, atol=1e-9):
                return True, U_rounded, V_rounded, W_rounded
        return False, None, None, None

    # Random search for Rank 29
    # Use random initializations with small integers/half-integers
    # and a simple local optimization (e.g., gradient descent on squared error)
    # Since it's non-convex, random restarts are key.
    
    best_result = None
    
    # We will do a few random restarts
    # Each restart:
    # 1. Initialize U, V, W with random values from {-1, -0.5, 0, 0.5, 1}
    # 2. Run a few steps of a simple optimization (e.g., coordinate descent or gradient)
    # 3. Check validity
    
    # To make it faster, we can use a "greedy" approach or just random perturbations
    # Given the time limit, we'll do a bounded number of restarts
    
    max_restarts = 50
    steps_per_restart = 100
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize
        U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n * m, target_rank))
        V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m * p, target_rank))
        W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p * n, target_rank))
        
        # Simple local search: perturb one variable at a time
        # Objective: minimize sum(T - target)^2
        # This is expensive to compute from scratch each time.
        # We can use a gradient-free method like Nelder-Mead, but it's slow in Python.
        # Instead, we'll do a few random perturbations and check.
        
        # Actually, a better strategy for small dimensions is to use a solver if available,
        # but we can't rely on external solvers.
        # Let's try a few random perturbations.
        
        current_error = None
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Compute current error
            T = np.zeros((n * m, m * p, p * n))
            for r in range(target_rank):
                T += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
            error = np.sum((T - target) ** 2)
            
            if current_error is None or error < current_error:
                current_error = error
                if error < 1e-6:
                    valid, U_r, V_r, W_r = is_valid(U, V, W, target_rank)
                    if valid:
                        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # Perturb a random element
            if random.random() < 0.5:
                # Perturb U
                i, j = np.random.randint(0, n * m), np.random.randint(0, target_rank)
                U[i, j] += random.choice([-0.5, 0.5])
            elif random.random() < 0.5:
                # Perturb V
                i, j = np.random.randint(0, m * p), np.random.randint(0, target_rank)
                V[i, j] += random.choice([-0.5, 0.5])
            else:
                # Perturb W
                i, j = np.random.randint(0, p * n), np.random.randint(0, target_rank)
                W[i, j] += random.choice([-0.5, 0.5])
                
            # Keep values bounded
            U = np.clip(U, -10, 10)
            V = np.clip(V, -10, 10)
            W = np.clip(W, -10, 10)

    # If no better rank found, return schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
