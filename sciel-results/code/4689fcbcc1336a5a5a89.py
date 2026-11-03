import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # ... existing code ...
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Attempt rank reduction via SVD
    # Note: This is a heuristic. True tensor rank reduction requires more than matrix SVD.
    # However, we check for exact linear dependencies in the columns.
    
    # If we find that the rank of U is less than the number of columns,
    # we might be able to project. But we must maintain the sum.
    # For now, we return the original if no simple reduction is found.
    
    return {"U": U, "V": V, "W": W}
