<<<<<<< SEARCH
def construct():
    # Start with schoolbook decomposition
    U, V, W, rank = _schoolbook()
    
    # Verify schoolbook is correct
    if not _verify(U, V, W, rank):
        U, V, W, rank = _schoolbook()
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to find a reduced-rank decomposition
    # Try ranks from 49 down to 30
    for target_rank in range(best_rank - 1, 25, -1):
        if not _deadline_ok():
            break
        
        result = _try_structured_search(target_rank)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
        
        if not _deadline_ok():
            break
        
        result = _try_block_structured(target_rank)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
=======
def construct():
    # Try rank-10 construction first
    U, V, W, rank = _construct_rank10()
    if _verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    U, V, W, rank = _schoolbook()
    
    # Verify schoolbook is correct
    if not _verify(U, V, W, rank):
        U, V, W, rank = _schoolbook()
    
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try to find a reduced-rank decomposition
    # Try ranks from 49 down to 30
    for target_rank in range(best_rank - 1, 25, -1):
        if not _deadline_ok():
            break
        
        result = _try_structured_search(target_rank)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
        
        if not _deadline_ok():
            break
        
        result = _try_block_structured(target_rank)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
>>>>>>> REPLACE
