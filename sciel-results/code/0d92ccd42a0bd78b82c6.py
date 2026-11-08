"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if isinstance(U[0], list) else U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m + j]
                v_val = V_arr[j*p + k]
                w_val = W_arr[k*n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook decomposition
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m + j, col] = 1
                V0[j*p + k, col] = 1
                W0[k*n + i, col] = 1
                col += 1
    
    def to_dict(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best = to_dict(U0, V0, W0)
    
    # Try to reduce rank by setting some columns to zero and checking
    # This is a greedy approach: try removing one column at a time
    
    current_U = U0.copy()
    current_V = V0.copy()
    current_W = W0.copy()
    current_rank = rank0
    
    # Try to remove columns
    for attempt in range(10000):
        if time.time() > deadline:
            break
        
        if current_rank <= 0:
            break
        
        # Try removing a random column
        col_idx = random.randint(0, current_rank - 1)
        
        # Check if we can remove this column by adjusting others
        # For now, just try zeroing it out and see if we can fix
        # This is complex, so let's try a simpler approach:
        # Try random perturbations
        
        # Alternative: try to find if any column is redundant
        # Check if column col_idx can be expressed as a combination of others
        # This requires solving a linear system, which is expensive
        
        # Simpler: try small random modifications
        if random.random() < 0.5:
            # Try setting a random entry to 0
            mat = random.choice([current_U, current_V, current_W])
            row = random.randint(0, mat.shape[0] - 1)
            col = random.randint(0, mat.shape[1] - 1)
            old_val = mat[row, col]
            mat[row, col] = 0
            
            # Check if still valid (this won't work since we're modifying in-place)
            # Need to check with the current state
            if _verify(current_U.tolist(), current_V.tolist(), current_W.tolist(), n, m, p):
                # Valid! Keep the change
                pass
            else:
                # Revert
                mat[row, col] = old_val
    
    # Convert best to dict
    if _verify(current_U.tolist(), current_V.tolist(), current_W.tolist(), n, m, p):
        best = to_dict(current_U, current_V, current_W)
    
    return best
# EVOLVE-BLOCK-END
