<<<<<<< SEARCH
def construct():
    # The schoolbook decomposition is the most reliable known solution for this size
    # within the constraints of this script. Complex searches for rank reduction
    # are unlikely to succeed in the given time frame and have been removed
    # to ensure immediate and stable return.
    return _schoolbook()
=======
def construct():
    """
    Try to find a lower-rank decomposition than schoolbook (rank 48).
    
    Start with schoolbook as the guaranteed fallback.
    Then try structured searches for lower ranks.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    deadline = _START_TIME + DEADLINE_SECONDS
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = schoolbook_rank
    
    # Try to find lower ranks
    # Based on known results, the matmul rank for 3x4x4 might be lower than 48.
    # Let's try ranks from 47 down to some reasonable minimum.
    
    # For efficiency, try a few promising ranks first
    ranks_to_try = [47, 46, 45, 44, 43, 42, 41, 40]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 30:  # Leave 30s safety margin
            break
        
        result = _try_lower_rank_structured(target_rank)
        if result is not None:
            best = result
            best_rank = target_rank
            # If we found a lower rank, continue trying even lower
            # But if we're running low on time, stop
            if time.time() > deadline - 60:
                break
    
    return best
>>>>>>> REPLACE
