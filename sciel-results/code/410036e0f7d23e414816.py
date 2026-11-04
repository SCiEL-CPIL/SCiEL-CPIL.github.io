def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Recursive/Structured Ansatz:
    # The tensor T_{n,m,p} is the direct sum of p copies of T_{n,m,1}.
    # We construct each T_{n,m,1} block independently and place it in the
    # corresponding rank slice.
    
    for k in range(p):
        # Base case: T_{n,m,1} has rank n*m
        # Indices for the k-th block
        start_rank = k * n * m
        end_rank = start_rank + n * m
        
        # For each (i, j) pair in the n x m grid
        idx = 0
        for i in range(n):
            for j in range(m):
                r = start_rank + idx
                
                # U maps A_{ij} to the r-th rank component
                U[i * m + j, r] = 1
                
                # V maps B_{jk} to the r-th rank component
                # Note: B is m x p, so B_{j, k} is at row j, col k
                V[j * p + k, r] = 1
                
                # W maps the product to C_{ik}
                # Note: C is n x p, but W is p x n?
                # Task says: W[k*n+i, r].
                # This implies W's rows are indexed by (k, i) -> k*n + i.
                # This corresponds to the output element C_{i, k}.
                W[k * n + i, r] = 1
                
                idx += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
