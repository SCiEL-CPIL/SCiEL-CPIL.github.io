import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Build the target tensor T (flattened)
    # Matrix A is n x m. Index (i,j) -> a = i*m + j
    # Matrix B is m x p. Index (j,k) -> b = j*p + k
    # Matrix C is n x p. Index (i,k) -> c = k*n + i (Trace convention)
    
    # Non-zero entries of T
    nonzero_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero_entries.append((a, b, c))
    
    num_nonzero = len(nonzero_entries)
    
    # We seek U, V, W with integer entries such that:
    # For each (a,b,c) in nonzero: sum_r U[a,r]*V[b,r]*W[c,r] = 8
    # For all other (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 0
    
    # Strategy: Fix U and W, then solve for V linearly
    # Or use a constructive approach based on known decompositions
    
    # For 2x3x4, known rank is 19. Let's try to construct it directly.
    # We'll use the fact that this is a sub-tensor of 3x3x3 multiplication.
    
    # Alternative: Use random search with exact rational verification
    # Try ranks from 19 upward
    
    def verify_decomposition(U, V, W, rank):
        """Verify that U, V, W (as integer arrays) satisfy the decomposition."""
        for (a, b, c) in nonzero_entries:
            val = 0
            for r in range(rank):
                val += U[a, r] * V[b, r] * W[c, r]
            if val != 8:
                return False
        
        # Check that zero entries remain zero (sample a few for speed)
        all_indices = list(product(range(n*m), range(m*p), range(p*n)))
        nonzero_set = set(nonzero_entries)
        for (a, b, c) in all_indices:
            if (a, b, c) not in nonzero_set:
                val = 0
                for r in range(rank):
                    val += U[a, r] * V[b, r] * W[c, r]
                if val != 0:
                    return False
        return True
    
    # Try rank 19 first (known optimal)
    for target_rank in [19, 20, 21, 22, 23, 24]:
        if time.time() > deadline - 15:
            break
        
        # Use random initialization and local search
        np.random.seed(42 + target_rank)
        
        for attempt in range(50):
            if time.time() > deadline - 10:
                break
            
            # Initialize with small random integers in range [-2, 2]
            U = np.random.randint(-2, 3, size=(n*m, target_rank))
            V = np.random.randint(-2, 3, size=(m*p, target_rank))
            W = np.random.randint(-2, 3, size=(p*n, target_rank))
            
            # Local search: try to fix errors by adjusting entries
            for iteration in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Check current error
                errors = {}
                for (a, b, c) in nonzero_entries:
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(target_rank))
                    errors[(a, b, c)] = val - 8
                
                # If all errors are zero, we're done
                if all(e == 0 for e in errors.values()):
                    if verify_decomposition(U, V, W, target_rank):
                        U_out = (U / 2.0).tolist()
                        V_out = (V / 2.0).tolist()
                        W_out = (W / 2.0).tolist()
                        return {"U": U_out, "V": V_out, "W": W_out}
                
                # Find the worst error
                if errors:
                    worst_key = max(errors, key=lambda k: abs(errors[k]))
                    worst_err = errors[worst_key]
                    if worst_err == 0:
                        break
                    
                    a, b, c = worst_key
                    # Try adjusting U[a, r], V[b, r], or W[c, r] for some r
                    for r in range(target_rank):
                        # Try changing U[a, r] by +/-1
                        for delta in [-1, 1]:
                            U[a, r] += delta
                            # Check if this helps
                            new_val = sum(U[a, r2] * V[b, r2] * W[c, r2] for r2 in range(target_rank))
                            if abs(new_val - 8) < abs(worst_err):
                                pass  # Keep the change
                            else:
                                U[a, r] -= delta  # Revert
                        # Try changing V[b, r] by +/-1
                        for delta in [-1, 1]:
                            V[b, r] += delta
                            new_val = sum(U[a, r2] * V[b, r2] * W[c, r2] for r2 in range(target_rank))
                            if abs(new_val - 8) < abs(worst_err):
                                pass
                            else:
                                V[b, r] -= delta
                        # Try changing W[c, r] by +/-1
                        for delta in [-1, 1]:
                            W[c, r] += delta
                            new_val = sum(U[a, r2] * V[b, r2] * W[c, r2] for r2 in range(target_rank))
                            if abs(new_val - 8) < abs(worst_err):
                                pass
                            else:
                                W[c, r] -= delta
    
    # Fallback: Return naive decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx, (i, j, k) in enumerate(product(range(n), range(m), range(p))):
        U[i * m + j][idx] = 1.0
        V[j * p + k][idx] = 1.0
        W[k * n + i][idx] = 1.0
        
    return {"U": U, "V": V, "W": W}
