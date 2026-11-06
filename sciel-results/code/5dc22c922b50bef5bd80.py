<<<<<<< SEARCH
    # Evaluate all candidates using the heuristic
    best_cell = None
    best_heuristic_score = -1
    
    for idx, pattern_func in enumerate(candidates):
        if not time_ok():
            break
        
        cell = make_cell_from_pattern(pattern_func)
        
        if not check_symmetry(cell):
            continue
        
        score = approximate_gap(cell)
        
        if score > best_heuristic_score:
            best_heuristic_score = score
            best_cell = cell
=======
    # Evaluate all candidates using the fast TM gap estimate
    best_cell = None
    best_score = -1.0
    
    # First pass: use heuristic to pre-filter
    pre_filtered = []
    for idx, pattern_func in enumerate(candidates):
        if not time_ok():
            break
        
        cell = make_cell_from_pattern(pattern_func)
        
        if not check_symmetry(cell):
            continue
        
        heuristic_score = approximate_gap(cell)
        pre_filtered.append((heuristic_score, cell))
    
    # Sort by heuristic score and evaluate top candidates with fast TM estimate
    pre_filtered.sort(key=lambda x: x[0], reverse=True)
    
    # Evaluate top N candidates with the more expensive fast TM estimate
    n_eval = min(50, len(pre_filtered))
    for idx in range(n_eval):
        if not time_ok():
            break
        
        heuristic_score, cell = pre_filtered[idx]
        
        try:
            tm_score = fast_tm_gap_estimate(cell)
        except:
            tm_score = 0.0
        
        if tm_score > best_score:
            best_score = tm_score
            best_cell = cell
    
    # If no valid cell was found, use the best heuristic cell
    if best_cell is None and pre_filtered:
        best_cell = pre_filtered[0][1]
        best_score = pre_filtered[0][0]
>>>>>>> REPLACE

Now update the local search to use the fast TM estimate:

