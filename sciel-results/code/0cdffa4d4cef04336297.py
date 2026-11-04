<<<<<<< SEARCH
    # Using numpy arrays for efficiency
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
=======
    # Using numpy arrays for efficiency
    # Note: Shapes must be (n*m, rank), (m*p, rank), (p*n, rank)
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
>>>>>>> REPLACE
<<<<<<< SEARCH
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1.0
                V[j * p + k, column] = 1.0
                W[k * n + i, column] = 1.0
                column += 1
                
                # Check time, though this loop is very fast
                if time.time() > deadline:
                    # If we hit the deadline (unlikely for this small size), 
                    # we return what we have. Since we build column by column,
                    # we might have a partial decomposition if interrupted,
                    # but for n=3, m=5, p=5, it finishes in microseconds.
                    pass

    return {"U": U, "V": V, "W": W}
=======
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: row i*m+j (correct for n*m rows)
                U[i * m + j, column] = 1.0
                # V index: row j*p+k (correct for m*p rows)
                V[j * p + k, column] = 1.0
                # W index: row k*n+i (correct for p*n rows, following W[k*n+i,r] convention)
                W[k * n + i, column] = 1.0
                column += 1
                
                # Check time, though this loop is very fast
                if time.time() > deadline:
                    # If we hit the deadline (unlikely for this small size), 
                    # we return what we have. Since we build column by column,
                    # we might have a partial decomposition if interrupted,
                    # but for n=3, m=5, p=5, it finishes in microseconds.
                    pass

    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
