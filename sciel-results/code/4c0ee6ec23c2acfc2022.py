<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    T = _build_target()
    T_float = T.astype(np.float64)
    
    # Baseline: Rank 18
    U18, V18, W18 = _get_rank18_solution()
    best_result = {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
    
    # Try constructive rank 16 (will likely fail, but quick to check)
    if time.time() < deadline - 120:
        U16, V16, W16 = _try_constructive_rank16()
        if _check_decomposition_float(U16, V16, W16, T_float):
            if _check_decomposition_exact(U16, V16, W16, T):
                best_result = {"U": U16.tolist(), "V": V16.tolist(), "W": W16.tolist()}
                return best_result
    
    # Try to find Rank 17 with structured initialization - give it most of the time
    search_deadline_17 = min(deadline - 15, start_time + 125)
    result = _try_als_rank(17, T, search_deadline_17, n_seeds=100, max_iters=60)
    if result is not None:
        U_l, V_l, W_l = result
        best_result = {"U": U_l, "V": V_l, "W": W_l}
    
    # If time permits, try Rank 16 with ALS
    if time.time() < deadline - 5:
        result = _try_als_rank(16, T, deadline - 2, n_seeds=10, max_iters=30)
        if result is not None:
            U_l, V_l, W_l = result
            best_result = {"U": U_l, "V": V_l, "W": W_l}
    
    return best_result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    T = _build_target()
    T_float = T.astype(np.float64)
    
    # Baseline: Rank 18
    U18, V18, W18 = _get_rank18_solution()
    best_result = {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
    
    # Try to find Rank 17 with structured initialization - give it most of the time
    search_deadline_17 = deadline - 10
    result = _try_als_rank(17, T, search_deadline_17, n_seeds=200, max_iters=80)
    if result is not None:
        U_l, V_l, W_l = result
        best_result = {"U": U_l, "V": V_l, "W": W_l}
    
    return best_result
>>>>>>> REPLACE
