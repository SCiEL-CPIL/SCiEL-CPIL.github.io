import time
import numpy as np
import itertools

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0])
    
    # Build the multiplication tensor from the decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Compare with the standard multiplication tensor where T[i,j,k] = 1 if the product uses A[i,j]*B[j,k]
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i,j,k), compute the sum
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The standard tensor has T[i,j,k] = 1 (each product term appears once)
                val = 0.0
                for r in range(rank):
                    val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Start with schoolbook decomposition
    rank = nm * mp  # Wait, that's not right. Let me reconsider.
    # Schoolbook rank for n x m times m x p is n*m*p
    rank = n * m * p
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    # Try to reduce rank by removing columns
    current_rank = rank
    while current_rank > 1 and time.time() < deadline:
        # Try removing each column and check validity
        removed = False
        for col_idx in range(current_rank):
            if time.time() >= deadline:
                break
            
            # Temporarily remove column col_idx
            U_test = np.delete(U, col_idx, axis=1)
            V_test = np.delete(V, col_idx, axis=1)
            W_test = np.delete(W, col_idx, axis=1)
            
            if _validate(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
                # Successfully removed this column
                U = U_test
                V = V_test
                W = W_test
                current_rank -= 1
                removed = True
                break
        
        if not removed:
            break
    
    # Try more aggressive reduction: try removing multiple columns at once
    # First, let's try random linear combinations to see if we can merge terms
    if time.time() < deadline - 30:
        # Try to find a lower rank by searching over random projections
        np.random.seed(42)
        
        for trial in range(50):
            if time.time() >= deadline - 10:
                break
            
            target_rank = current_rank - 1
            if target_rank < 1:
                break
            
            # Try to project the current decomposition to a lower rank
            # This is heuristic - try random linear combinations
            
            # For now, just try removing one more column if possible
            for col_idx in range(current_rank):
                if time.time() >= deadline - 5:
                    break
                U_test = np.delete(U, col_idx, axis=1)
                V_test = np.delete(V, col_idx, axis=1)
                W_test = np.delete(W, col_idx, axis=1)
                
                if _validate(U_test.tolist(), V_test.tolist(), W_test.tolist(), n, m, p):
                    U = U_test
                    V = V_test
                    W = W_test
                    current_rank -= 1
                    break
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
