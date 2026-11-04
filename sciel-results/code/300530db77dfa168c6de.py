<<<<<<< SEARCH
def construct():
    """
    Rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Uses 8 independent 2x2 Strassen blocks (rank 7 each) as baseline,
    then attempts rank reduction via linear algebra.
    Output factor follows W[k*n+i, r] convention.
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit
    
    # Build the baseline rank-56 decomposition
    U, V, W = _build_strassen56()
    
    # Validate the baseline
    if not _validate(U, V, W):
        # Should not happen, but return anyway
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    U_reduced, V_reduced, W_reduced, reduced_rank = _try_reduce_rank(
        U, V, W, 55, deadline
    )
    
    if reduced_rank < len(U[0]):
        # Validate the reduced version
        if _validate(U_reduced, V_reduced, W_reduced):
            return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    
    # Return the baseline
    return {"U": U, "V": V, "W": W}
=======
def construct():
    """
    Rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.
    Uses 8 independent 2x2 Strassen blocks (rank 7 each) as baseline,
    then attempts rank reduction via linear algebra.
    Output factor follows W[k*n+i, r] convention.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Build the baseline rank-56 decomposition
    U, V, W = _build_strassen56()
    
    # Validate the baseline
    if not _validate(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    U_reduced, V_reduced, W_reduced, reduced_rank = _try_reduce_rank(
        U, V, W, 55, deadline
    )
    
    if reduced_rank < len(U[0]):
        # Validate the reduced version
        if _validate(U_reduced, V_reduced, W_reduced):
            return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    
    # Return the baseline
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
