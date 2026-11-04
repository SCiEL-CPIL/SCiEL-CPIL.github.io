<<<<<<< SEARCH
    rank = len(U)
    # Should be 7 + 18 = 25
    
    return {"U": U, "V": V, "W": W}
=======
    rank = len(U)
    # Should be 7 + 18 = 25
    
    import numpy as np
    
    # Transpose to match required shapes:
    # U: (n*m, rank) -> (6, rank)
    # V: (m*p, rank) -> (15, rank)
    # W: (p*n, rank) -> (10, rank)
    
    U_mat = np.array(U, dtype=float).T
    V_mat = np.array(V, dtype=float).T
    W_mat = np.array(W, dtype=float).T
    
    return {"U": U_mat, "V": V_mat, "W": W_mat}
>>>>>>> REPLACE
