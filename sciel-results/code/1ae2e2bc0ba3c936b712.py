<<<<<<< SEARCH
    U, V, W = get_rank14_factors()
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return naive_decomp()
=======
    # Try to find a rank-12 decomposition
    # This is a heuristic search
    import random
    import time
    
    def try_rank12_search():
        start_time = time.time()
        deadline = start_time + 100  # 100s budget
        
        # Initialize random matrices
        for _ in range(1000):
            if time.time() > deadline:
                break
            
            U = [[random.randint(-1, 1) for _ in range(12)] for _ in range(nm)]
            V = [[random.randint(-1, 1) for _ in range(12)] for _ in range(mp)]
            W = [[random.randint(-1, 1) for _ in range(12)] for _ in range(pn)]
            
            if verify(U, V, W):
                return U, V, W
        
        return None

    U12, V12, W12 = try_rank12_search()
    if U12 is not None:
        return {"U": U12, "V": V12, "W": W12}

    U, V, W = get_rank14_factors()
    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return naive_decomp()
>>>>>>> REPLACE
