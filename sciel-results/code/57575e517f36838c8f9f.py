<<<<<<< SEARCH
    rank = len(U)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
    rank = len(U)
    
    # Convert lists of vectors to 2D arrays
    # U is list of rank vectors of length n*m -> need shape (n*m, rank)
    # V is list of rank vectors of length m*p -> need shape (m*p, rank)
    # W is list of rank vectors of length p*n -> need shape (p*n, rank)
    
    U_arr = np.array(U, dtype=float).T
    V_arr = np.array(V, dtype=float).T
    W_arr = np.array(W, dtype=float).T
    
    return {"U": U_arr, "V": V_arr, "W": W_arr}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
