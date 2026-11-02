def construct():
    deadline = time.time() + 135  # Safety margin before 150s kill
    
    # First, get the known rank-11 construction
    result11 = _rank11_decomposition()
    if _verify(result11["U"], result11["V"], result11["W"], n, m, p):
        best = result11
        best_rank = 11
    else:
        best = _schoolbook()
        best_rank = 12
    
    # Try Kronecker product lifting approach for rank 10
    if best_rank > 10 and time.time() < deadline - 60:
        result10_k = _try_rank10_kronecker(deadline)
        if result10_k is not None:
            best = result10_k
            best_rank = 10
    
    # Try linear projection approach for rank 10
    if best_rank > 10 and time.time() < deadline - 40:
        result10_p = _try_rank10_projection(deadline)
        if result10_p is not None:
            best = result10_p
            best_rank = 10
    
    # Try structured rank-10 search
    if best_rank > 10 and time.time() < deadline - 20:
        result10_s = _try_rank10_structured(deadline)
        if result10_s is not None:
            best = result10_s
            best_rank = 10
    
    return best
