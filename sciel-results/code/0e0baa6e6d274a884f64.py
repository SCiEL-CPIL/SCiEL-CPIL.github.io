<<<<<<< SEARCH
    # Convert to numpy arrays
    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U, "V": V, "W": W}
=======
    # Convert to numpy arrays
    rank = len(U_cols)
    if rank == 0:
        return {"U": np.zeros((n*m, 1)), "V": np.zeros((m*p, 1)), "W": np.zeros((p*n, 1))}
        
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for r in range(rank):
        U[:, r] = U_cols[r]
        V[:, r] = V_cols[r]
        W[:, r] = W_cols[r]
        
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
