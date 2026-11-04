<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin under 120s cap

    n, m, p = DIMENSIONS

    # Use Strassen-based decomposition for 2x3*3x10: rank 55
    U, V, W, rank = _build_strassen(n, m, p)

    # Validate
    if not _validate(n, m, p, U, V, W):
        # Fall back to trivial if validation fails
        U, V, W, rank = _build_trivial(n, m, p)

    return {"U": U, "V": V, "W": W}
=======
def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin under 120s cap

    n, m, p = DIMENSIONS

    # Try multiple approaches and pick the best one
    best_rank = float('inf')
    best_result = None
    
    # Approach 1: Strassen decomposition (rank 55)
    try:
        U, V, W, rank = _build_strassen(n, m, p)
        if _validate(n, m, p, U, V, W) and rank < best_rank:
            best_rank = rank
            best_result = (U, V, W)
    except Exception as e:
        print(f"Strassen approach failed: {e}")
    
    # Approach 2: Vandermonde-based optimization (exploratory)
    try:
        U, V, W, rank = _optimize_strassen_with_vandermonde(n, m, p)
        if _validate(n, m, p, U, V, W) and rank < best_rank:
            best_rank = rank
            best_result = (U, V, W)
    except Exception as e:
        print(f"Vandermonde optimization failed: {e}")
    
    # Approach 3: Trivial decomposition (fallback)
    if best_result is None:
        U, V, W, rank = _build_trivial(n, m, p)
        best_result = (U, V, W)
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE

