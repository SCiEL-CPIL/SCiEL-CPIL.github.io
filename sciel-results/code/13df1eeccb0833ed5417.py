<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = 115.0 # Safety margin before 120s limit
    
    # Try from largest n down
    # n=47 requires min degree 8.
    # n=46 requires min degree 7.
    
    # Allocate time budget for each n
    # We want to spend most time on the largest n that is likely to succeed.
    # If 47 fails quickly, we try 46.
    
    time_budgets = {
        47: 30.0,
        46: 40.0,
        45: 20.0,
        44: 10.0,
        43: 5.0,
        42: 5.0,
        41: 5.0,
        40: 5.0,
        39: 2.0
    }
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
            
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 2:
            break
            
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        # Cap the time for this n
        t_limit = min(time_budgets.get(n, 5.0), remaining_time - 1)
        
        if t_limit <= 0:
            continue
            
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result is not None:
            return result
            
    # Fallback: return empty graph on 39 vertices (min degree 0 >= 39-39=0)
    # This is always valid.
    return {'n': 39, 'edges': []}
=======
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    time_budgets = {
        47: 50.0,
        46: 60.0,
        45: 20.0,
        44: 10.0,
        43: 5.0,
        42: 3.0,
        41: 2.0,
        40: 1.0,
        39: 1.0
    }
    
    for n in range(47, 38, -1):
        if time.time() - start_time > deadline:
            break
            
        remaining_time = deadline - (time.time() - start_time)
        if remaining_time < 3:
            break
            
        min_deg_req = n - 39
        if min_deg_req < 1:
            min_deg_req = 1
            
        t_limit = min(time_budgets.get(n, 3.0), remaining_time - 2)
        
        if t_limit <= 1:
            continue
            
        result = _construct_c4_free(n, min_deg_req, t_limit)
        if result is not None:
            return result
            
    return {'n': 39, 'edges': []}
>>>>>>> REPLACE
