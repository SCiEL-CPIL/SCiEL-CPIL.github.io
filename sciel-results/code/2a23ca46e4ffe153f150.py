<<<<<<< SEARCH
def _find_low_rank_3x4x4(max_rank, time_limit):
    """
    Attempt to find a low-rank decomposition for 3x4x4 matrix multiplication.
    Returns U, V, W if found, else None.
    """
    start_time = time.time()
    n, m, p = 3, 4, 4
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target ranks to try, from low to high
    target_ranks = [18, 20, 24, 30, 36, 40, 48]
    
    # Filter target ranks <= max_rank
    target_ranks = [r for r in target_ranks if r <= max_rank]
    
    if not target_ranks:
        return None
        
    for rank in target_ranks:
        if time.time() - start_time > time_limit:
            break
            
        # Try to find a decomposition of this rank
        # Use a simple random initialization and refinement
        # This is a heuristic search
        
        # We can use the fact that the tensor is sparse.
        # For now, we just check if we can construct one.
        # Since deriving exact integer/half-integer solutions is hard,
        # we will use a fallback strategy:
        # If we cannot find a specific low rank, we return None.
        
        # However, we know that rank 48 is always possible (schoolbook).
        # Can we do better?
        # Let's try to construct a rank 36 decomposition if possible.
        # For now, this function is a placeholder for a search algorithm.
        # If we don't have a specific algorithm, we return None.
        
        # To be safe and improve on 96, we can try to use a block structure.
        # If we assume R(3x4x4) <= 36, then total rank 72.
        # If we assume R(3x4x4) <= 24, then total rank 48.
        
        # Let's try to generate a valid decomposition using a known method if available.
        # Since I don't have a specific low-rank generator, I will return None 
        # unless rank == 48 (which is trivial).
        
        if rank == 48:
             U = np.zeros((dim_u, rank))
             V = np.zeros((dim_v, rank))
             W = np.zeros((dim_w, rank))
             col = 0
             for i in range(n):
                 for j in range(m):
                     for k in range(p):
                         U[i * m + j, col] = 1
                         V[j * p + k, col] = 1
                         W[k * n + i, col] = 1
                         col += 1
             if _verify_decomposition(U, V, W, n, m, p):
                 return U, V, W
                 
    return None
=======
def _find_low_rank_3x4x4(max_rank, time_limit):
    """
    Attempt to find a low-rank decomposition for 3x4x4 matrix multiplication.
    Returns U, V, W if found, else None.
    Currently only supports schoolbook rank 48.
    """
    n, m, p = 3, 4, 4
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # We only have a known construction for rank 48 (schoolbook)
    # If max_rank is at least 48, we can return it.
    if max_rank >= 48:
        rank = 48
        U = np.zeros((dim_u, rank))
        V = np.zeros((dim_v, rank))
        W = np.zeros((dim_w, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        if _verify_decomposition(U, V, W, n, m, p):
            return U, V, W
            
    return None
>>>>>>> REPLACE

