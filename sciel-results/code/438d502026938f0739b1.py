<<<<<<< SEARCH
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if time.time() > _DEADLINE - _MARGIN:
        # Safety: should never trigger for this O(1) construction
        pass
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    if time.time() > _DEADLINE - _MARGIN:
        # Safety: should never trigger for this O(1) construction
        pass
    
    return {"U": np.array(U), "V": np.array(V), "W": np.array(W)}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
