<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Leave 20s margin
    
    # Start with schoolbook
    schoolbook = _construct_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = schoolbook["rank"]
    
    # Try lower ranks using Newton's method
    # The minimal rank for 4x4x5 MM is unknown, but we know it's <= 80
    # Let's try ranks from 79 down to some lower bound
    # 
    # For matrix multiplication, the tensor rank is at least max(n*m, m*p, p*n) / something
    # A lower bound is max(n*m, m*p, p*n) but that's not tight.
    # Actually, a simple lower bound is: rank >= max(n*m, m*p, p*n) is not true.
    # 
    # Let's just try a range of ranks.
    
    # Try ranks: we'll try from 79 down to 60
    for rank in range(79, 59, -1):
        if time.time() > deadline:
            break
        
        result = _try_newton(rank, n, m, p, max_iter=50, tol=1e-10, deadline=deadline)
        if result is not None:
            # Verify
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-8):
                best_result = result
                best_rank = rank
                break  # Found a valid lower rank, stop
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
=======
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # Leave 30s margin for safety
    
    # Start with schoolbook
    schoolbook = _construct_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = schoolbook["rank"]
    
    # Try lower ranks using scipy optimization
    # For 4x4x5 MM, try ranks from 79 down to 50
    # Allocate time based on how many ranks we can try
    ranks_to_try = list(range(79, 49, -1))
    
    for rank in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Time budget for this rank
        time_elapsed = time.time() - start_time
        time_remaining = deadline - time_elapsed
        max_time_for_rank = min(15, max(5, time_remaining / max(1, len(ranks_to_try))))
        
        result = _try_optimize(rank, n, m, p, max_time=max_time_for_rank, deadline=deadline)
        if result is not None:
            # Verify
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-8):
                best_result = result
                best_rank = rank
                break  # Found a valid lower rank, stop
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
>>>>>>> REPLACE

