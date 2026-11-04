<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    baseline = _schoolbook(n, m, p)
    best_result = baseline
    best_rank = n * m * p
    
    # Try to find lower rank
    # Start from 49 down to 25
    for rank in range(49, 24, -1):
        if time.time() > deadline:
            break
        
        result = _try_rank_local_search(n, m, p, rank, time_limit=5.0, seed=rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
            
    return best_result
=======
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    baseline = _schoolbook(n, m, p)
    best_result = baseline
    best_rank = n * m * p
    
    # Try to find lower rank
    # Start from 49 down to 25
    for rank in range(49, 24, -1):
        if time.time() > deadline:
            break
        
        # Allocate time based on rank
        remaining = deadline - time.time()
        if remaining < 3:
            break
        
        # Give more time to higher ranks (closer to baseline)
        time_for_rank = min(8.0, remaining * 0.3)
        
        result = _try_rank_local_search(n, m, p, rank, time_limit=time_for_rank, seed=rank)
        if result is not None:
            best_result = result
            best_rank = rank
            break
            
    return best_result
>>>>>>> REPLACE
