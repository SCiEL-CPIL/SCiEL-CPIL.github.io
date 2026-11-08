"""Matrix multiplication 2x5x6 tensor decomposition - structured approach."""
import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = 2, 5, 6
    trivial_rank = n * m * p  # 60
    
    # The tensor T has dimensions (n*m, m*p, p*n) = (10, 30, 12)
    # T[a,b,c] = 1 if indices are consistent (j=j', k=k', i=i'), 0 otherwise
    # where a=i*m+j, b=j'*p+k, c=k'*n+i'
    
    # Trivial decomposition: one rank-1 term per (i,j,k) triple
    rank = trivial_rank
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    
    # Verify: for consistent (a,b,c), sum_r U[a,r]*V[b,r]*W[c,r] should be 1
    # For inconsistent (a,b,c), it should be 0
    # The trivial solution satisfies this by construction.
    
    # Attempt to find a lower-rank decomposition using numerical optimization
    # Strategy: try ranks from 59 down, using random initialization + refinement
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try a few candidate ranks with limited time budget
    for test_rank in range(59, 50, -1):  # Try ranks 59 down to 51
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for trial in range(3):
            if time.time() > deadline - 10:
                break
            
            # Initialize with the trivial solution's first test_rank columns
            # This gives a valid starting point (though not minimal)
            U_try = U[:, :test_rank].copy()
            V_try = V[:, :test_rank].copy()
            W_try = W[:, :test_rank].copy()
            
            # Add small random perturbation
            U_try += 0.01 * np.random.randn(n*m, test_rank)
            V_try += 0.01 * np.random.randn(m*p, test_rank)
            W_try += 0.01 * np.random.randn(p*n, test_rank)
            
            # Simple iterative refinement using alternating minimization
            # For each mode, solve a linear least squares problem
            converged = False
            for iteration in range(10):
                if time.time() > deadline - 10:
                    break
                
                # Solve for U: for each a, minimize ||U[a,:] * (V^T ⊙ W^T) - T[a,:,:]||^2
                # This is expensive. Use a simpler approach.
                
                # Instead, check if the current decomposition is close to exact
                # by sampling a few (a,b,c) triples
                
                # Check consistent triples
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.sum(U_try[a, :] * V_try[b, :] * W_try[c, :])
                            error += (val - 1.0) ** 2
                
                # Check a few inconsistent triples
                for _ in range(10):
                    a = np.random.randint(0, n*m)
                    b = np.random.randint(0, m*p)
                    c = np.random.randint(0, p*n)
                    # Check if consistent
                    i_a = a // m
                    j_a = a % m
                    j_b = b // p
                    k_b = b % p
                    k_c = c // n
                    i_c = c % n
                    if j_a == j_b and k_b == k_c and i_a == i_c:
                        continue  # Skip consistent triples
                    val = np.sum(U_try[a, :] * V_try[b, :] * W_try[c, :])
                    error += val ** 2
                
                if error < 1e-6:
                    converged = True
                    best_U, best_V, best_W, best_rank = U_try, V_try, W_try, test_rank
                    break
                
                # Simple update: project to reduce error
                # This is a heuristic, not guaranteed to converge
                U_try *= 0.99
                V_try *= 0.99
                W_try *= 0.99
            
            if converged:
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-START
# Entry point is construct(), returns dict with 'U', 'V', 'W' keys
# EVOLVE-BLOCK-END
