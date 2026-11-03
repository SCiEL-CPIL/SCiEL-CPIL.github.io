"""Tensor decomposition for 3x3 by 3x5 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
import random
import itertools

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the multiplication tensor."""
    # For each (i, j, k), check sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(U.shape[1]):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    return True

def _try_reduce_rank(U, V, W, n, m, p, max_iterations=1000):
    """
    Try to reduce the rank by finding linear dependencies.
    We apply random linear combinations to the columns and check if any become zero.
    """
    current_U = U.copy()
    current_V = V.copy()
    current_W = W.copy()
    current_rank = U.shape[1]
    
    for iteration in range(max_iterations):
        # Pick two random columns to combine
        if current_rank < 2:
            break
        c1, c2 = random.sample(range(current_rank), 2)
        
        # Try random combination: new_col = a * col1 + b * col2
        # We want to find a, b such that at least one factor has a zero in some row
        # that would allow us to eliminate a column
        
        # Strategy: try to make one of the factors have a zero pattern
        # that allows elimination
        
        # Try several random combinations
        found_reduction = False
        for trial in range(50):
            a = random.uniform(-2, 2)
            b = random.uniform(-2, 2)
            
            # Create new column c1 as a combination
            new_U_col = a * current_U[:, c1] + b * current_U[:, c2]
            new_V_col = a * current_V[:, c1] + b * current_V[:, c2]
            new_W_col = a * current_W[:, c1] + b * current_W[:, c2]
            
            # Check if any factor has all zeros (which would make this term useless)
            if np.allclose(new_U_col, 0) or np.allclose(new_V_col, 0) or np.allclose(new_W_col, 0):
                # This column becomes zero, we can remove it
                # But we need to verify the decomposition still works
                # Actually, if a column is zero in any factor, it contributes nothing
                # So we can just remove it
                
                # Remove column c1 (keep c2)
                mask = [i for i in range(current_rank) if i != c1]
                new_U = current_U[:, mask]
                new_V = current_V[:, mask]
                new_W = current_W[:, mask]
                
                if _verify_decomposition(new_U, new_V, new_W, n, m, p):
                    current_U = new_U
                    current_V = new_V
                    current_W = new_W
                    current_rank -= 1
                    found_reduction = True
                    break
        
        if not found_reduction:
            # Try a different approach: look for linear dependencies
            # Check if any column is a linear combination of others
            # This is more complex, so let's try a simpler heuristic
            
            # Try to find if two columns can be combined to create zeros
            # in specific positions that allow elimination
            
            # For now, try random small perturbations
            continue
    
    return current_U, current_V, current_W

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
    
    # Try to reduce the rank
    random.seed(42)
    U, V, W = _try_reduce_rank(U, V, W, n, m, p, max_iterations=2000)
    
    final_rank = U.shape[1]
    
    # Convert to lists
    U_list = [[float(U[i, j]) for j in range(final_rank)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(final_rank)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(final_rank)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
