<<<<<<< SEARCH
def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
    
    candidates = _construct_candidates()
    
    best_A = None
    best_obj = float('inf')
    
    rng = random.Random(42)
    
    # First pass: quick local search on each candidate
    for A in candidates:
        if time.time() - start_time > time_budget:
            break
        if not _is_valid(A):
            continue
        elapsed = time.time() - start_time
        remaining_for_this = max(1.0, time_budget - elapsed)
        refined_A, obj = _local_search(A, remaining_for_this, rng)
        if obj < best_obj:
            best_obj = obj
            best_A = refined_A
            
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Second pass: more intensive refinement with double moves
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 2:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining * 0.5, rng, use_double_move=True)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass
    
    # Final high-precision polish
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 1:
        try:
            rng2 = random.Random(123)
            refined_A, refined_obj = _polish_search(best_A, remaining, rng2)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass

    if not _is_valid(best_A):
        best_A = list(range(1, 21))
        
    return sorted(best_A)
=======
def construct():
    start_time = time.time()
    time_budget = TIME_LIMIT - 5
    
    candidates = _construct_candidates()
    
    best_A = None
    best_obj = float('inf')
    
    rng = random.Random(42)
    
    # First pass: quick local search on each candidate
    # Allocate time proportionally: more time for better initial candidates
    n_candidates = len(candidates)
    per_candidate_time = max(1.0, (time_budget - 10) / max(1, n_candidates))
    
    for A in candidates:
        if time.time() - start_time > time_budget:
            break
        if not _is_valid(A):
            continue
        elapsed = time.time() - start_time
        remaining_for_this = min(per_candidate_time, max(1.0, time_budget - elapsed))
        refined_A, obj = _local_search(A, remaining_for_this, rng)
        if obj < best_obj:
            best_obj = obj
            best_A = refined_A
            
    if best_A is None:
        best_A = list(range(1, 21))
        best_obj = _objective(best_A)
        
    # Second pass: more intensive refinement with double moves
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 3:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining * 0.4, rng, use_double_move=True)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass
    
    # Third pass: another round of single-move refinement
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 2:
        try:
            refined_A, refined_obj = _local_search(best_A, remaining * 0.5, rng)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass
    
    # Final high-precision polish
    remaining = TIME_LIMIT - 5 - (time.time() - start_time)
    if remaining > 1:
        try:
            rng2 = random.Random(123)
            refined_A, refined_obj = _polish_search(best_A, remaining, rng2)
            if refined_obj < best_obj:
                best_A = refined_A
                best_obj = refined_obj
        except Exception:
            pass

    if not _is_valid(best_A):
        best_A = list(range(1, 21))
        
    return sorted(best_A)
>>>>>>> REPLACE
