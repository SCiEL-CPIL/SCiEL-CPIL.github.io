def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    # Initial seeds
    best_set = [0, 1, 3]
    best_score = _score_set(best_set)
    
    try:
        # Structured seeds
        structured_candidates = [
            [0, 1, 4, 9, 16, 25, 36, 49, 64, 81], # Squares
            [0, 1, 2, 4, 8, 16, 32, 64, 128], # Powers of 2
            [0, 1, 3, 7, 15, 31, 63, 127], # Mersenne-like
            [0, 1, 4, 10, 20, 35, 56, 84], # Triangular
            [0, 1, 3, 4, 9, 10, 12, 13], # Small Sidon-like
            [0, 1, 4, 10, 18, 28, 40, 54], # Quadratic-ish
            [0, 1, 2, 4, 8, 16, 32, 64, 128, 256], # Larger powers of 2
            [0, 1, 3, 9, 27, 81, 243], # Powers of 3
            [0, 1, 4, 6, 10, 15, 21, 28], # Triangular + 1?
            [0, 1, 3, 6, 10, 15, 21, 28], # Triangular
        ]
        
        for cand in structured_candidates:
            if time.time() > deadline - 5:
                break
            s = _score_set(cand)
            if s > best_score:
                best_score = s
                best_set = list(cand)
        
        # Cyclic difference set seeds
        for q in [2, 3, 4, 5, 7]:
            if time.time() > deadline - 20:
                break
            diff_set = _cyclic_difference_set_seed(q, deadline)
            s = _score_set(diff_set)
            if s > best_score:
                best_score = s
                best_set = list(diff_set)
        
        # Geometric search (inspired by continuous optimization approach)
        if time.time() < deadline - 15:
            geo_set, geo_score = _continuous_geometric_search(deadline)
            if geo_score > best_score and geo_set:
                best_score = geo_score
                best_set = geo_set
        
        # Single pass of Dynamic Local Search
        if time.time() < deadline - 10:
            local_score, local_set = _local_search_dynamic(best_set, deadline, 500, 30)
            if local_score > best_score:
                best_score = local_score
                best_set = local_set

        # Final Polish: High-precision local search on the best set
        if time.time() < deadline - 2:
            final_score, final_set = _final_polish(best_set, deadline)
            if final_score > best_score:
                best_score = final_score
                best_set = final_set

    except Exception:
        # Fallback to initial best or seed
        pass

    # Ensure valid return
    if not isinstance(best_set, list) or 0 not in best_set:
        return [0, 1, 3]
        
    return best_set
