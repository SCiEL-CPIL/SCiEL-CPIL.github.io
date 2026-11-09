<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_S_int = 0
    best_count = 0
    
    while time.time() < deadline:
        S_int, count = _greedy_initial()
        S_int, count = _local_search(S_int, count, time_limit=5.0)
        
        if count > best_count:
            best_count = count
            best_S_int = S_int
            
    # Convert back to list of triples
    result = []
    for i in range(N_TRIPLES):
        if best_S_int & (1 << i):
            result.append(list(ALL_TRIPLES[i]))
            
    return result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    best_S_int = 0
    best_count = 0
    
    while time.time() < deadline:
        # Check if we have enough time for another iteration
        remaining = deadline - time.time()
        if remaining < 8.0:  # Need at least ~8s for greedy + local search
            break
            
        S_int, count = _greedy_initial()
        
        # Scale local search time based on remaining budget
        ls_time = min(6.0, remaining * 0.5)
        S_int, count = _local_search(S_int, count, time_limit=ls_time)
        
        if count > best_count:
            best_count = count
            best_S_int = S_int
            
    # Convert back to list of triples
    result = []
    for i in range(N_TRIPLES):
        if best_S_int & (1 << i):
            result.append(list(ALL_TRIPLES[i]))
            
    return result
>>>>>>> REPLACE
