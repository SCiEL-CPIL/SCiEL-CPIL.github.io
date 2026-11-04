def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the rank 44 decomposition
    result_44 = _strassen_44()
    ok, err = _verify_fast(result_44["U"], result_44["V"], result_44["W"], n, m, p)
    
    best_result = result_44
    best_rank = 44
    
    if ok:
        # Try to reduce rank
        reduced = _try_reduce_rank(
            np.array(result_44["U"]), 
            np.array(result_44["V"]), 
            np.array(result_44["W"]), 
            n, m, p
        )
        if reduced:
            # Verify the reduced result
            ok_red, err_red = _verify_fast(reduced["U"], reduced["V"], reduced["W"], n, m, p)
            if ok_red:
                new_rank = len(reduced["U"][0])
                if new_rank < best_rank:
                    best_result = reduced
                    best_rank = new_rank
    
    # Fallback to schoolbook if 44 fails (should not happen)
    if not ok:
        return _schoolbook()
        
    return best_result
