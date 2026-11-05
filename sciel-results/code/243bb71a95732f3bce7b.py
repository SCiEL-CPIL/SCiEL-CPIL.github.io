import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    # --- Part 1: Strassen 4x4x4 ---
    # Rank 49
    U1 = np.zeros((16, 49))
    V1 = np.zeros((28, 49))
    W1 = np.zeros((28, 49))
    
    # ... (Strassen logic as above, filling U1, V1, W1) ...
    # Note: V1 rows 0..15 correspond to B cols 0..3.
    # V1 rows 16..27 correspond to B cols 4..6 (should be 0).
    # W1 rows 0..15 correspond to C cols 0..3.
    # W1 rows 16..27 correspond to C cols 4..6 (should be 0).
    
    # ... implementation ...
    
    U_parts.append(U1)
    V_parts.append(V1)
    W_parts.append(W1)
    
    # --- Part 2: Schoolbook 4x4x3 ---
    # Rank 48
    U2 = np.zeros((16, 48))
    V2 = np.zeros((28, 48))
    W2 = np.zeros((28, 48))
    
    # ... (Schoolbook logic) ...
    # V2 rows for B cols 4..6 are filled.
    # W2 rows for C cols 4..6 are filled.
    
    U_parts.append(U2)
    V_parts.append(V2)
    W_parts.append(W2)
    
    U = np.concatenate(U_parts, axis=1)
    V = np.concatenate(V_parts, axis=1)
    W = np.concatenate(W_parts, axis=1)
    
    return {"U": U, "V": V, "W": W}
