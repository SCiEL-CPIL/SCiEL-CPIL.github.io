<<<<<<< SEARCH
def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Returns (centers, sides) maximizing sum of side lengths.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 580s safety margin
    
    # Generate multiple initial packings
    inits = [
        _init_grid_packing(),
        _init_heterogeneous_1(),
        _init_heterogeneous_2(),
        _init_heterogeneous_3(),
        _init_heterogeneous_4(),
        _init_heterogeneous_5(),
        _init_heterogeneous_6(),
        _init_heterogeneous_7(),
    ]
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for centers, sides in inits:
        if time.time() > deadline:
            break
        
        # First, resolve any overlaps
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        centers = _resolve_overlap(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        
        if not _is_valid(centers, sides):
            # Try harder to fix
            for _ in range(10):
                centers = _resolve_overlap(centers, sides)
                centers = _clamp_to_unit(centers, sides)
                if _is_valid(centers, sides):
                    break
        
        if not _is_valid(centers, sides):
            continue
        
        # Local optimization
        opt_centers, opt_sides = _local_optimize(centers, sides)
        
        # If time remaining, do a second pass with more rounds
        remaining = deadline - time.time()
        if remaining > 120:
            opt_centers2, opt_sides2 = _local_optimize(opt_centers, opt_sides, max_rounds=80)
            if _is_valid(opt_centers2, opt_sides2):
                obj2 = _objective(opt_centers2, opt_sides2)
                if obj2 > _objective(opt_centers, opt_sides):
                    opt_centers, opt_sides = opt_centers2, opt_sides2
        
        if _is_valid(opt_centers, opt_sides):
            obj = _objective(opt_centers, opt_sides)
            if obj > best_obj:
                best_obj = obj
                best_centers = opt_centers.copy()
                best_sides = opt_sides.copy()
        
        # Also try the original (pre-optimization) in case optimizer degraded
        orig_obj = _objective(centers, sides)
        if orig_obj > best_obj:
            best_obj = orig_obj
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # If no valid solution found, fall back to grid
    if best_centers is None:
        centers, sides = _init_grid_packing()
        best_centers = centers
        best_sides = sides
    else:
        # Final validation
        if not _is_valid(best_centers, best_sides):
            centers, sides = _init_grid_packing()
            best_centers = centers
            best_sides = sides
        
        # Final polish: one more round of growth on the best solution
        final_sides = best_sides.copy()
        for i in range(10):
            ns = _grow_square_max(best_centers, final_sides, i, tol=1e-7)
            if ns > final_sides[i] + 1e-9:
                final_sides[i] = ns
        if _is_valid(best_centers, final_sides):
            if _objective(best_centers, final_sides) > _objective(best_centers, best_sides):
                best_sides = final_sides
    
    return best_centers, best_sides
=======
def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Returns (centers, sides) maximizing sum of side lengths.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 580s safety margin
    
    # Generate multiple initial packings
    inits = [
        _init_grid_packing(),
        _init_heterogeneous_1(),
        _init_heterogeneous_2(),
        _init_heterogeneous_3(),
        _init_heterogeneous_4(),
        _init_heterogeneous_5(),
        _init_heterogeneous_6(),
        _init_heterogeneous_7(),
    ]
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for centers, sides in inits:
        if time.time() > deadline:
            break
        
        # First, resolve any overlaps
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        centers = _resolve_overlap(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        
        if not _is_valid(centers, sides):
            # Try harder to fix
            for _ in range(10):
                centers = _resolve_overlap(centers, sides)
                centers = _clamp_to_unit(centers, sides)
                if _is_valid(centers, sides):
                    break
        
        if not _is_valid(centers, sides):
            continue
        
        # Local optimization
        opt_centers, opt_sides = _local_optimize(centers, sides)
        
        # If time remaining, do a second pass with more rounds
        remaining = deadline - time.time()
        if remaining > 120:
            opt_centers2, opt_sides2 = _local_optimize(opt_centers, opt_sides, max_rounds=80)
            if _is_valid(opt_centers2, opt_sides2):
                obj2 = _objective(opt_centers2, opt_sides2)
                if obj2 > _objective(opt_centers, opt_sides):
                    opt_centers, opt_sides = opt_centers2, opt_sides2
        
        if _is_valid(opt_centers, opt_sides):
            obj = _objective(opt_centers, opt_sides)
            if obj > best_obj:
                best_obj = obj
                best_centers = opt_centers.copy()
                best_sides = opt_sides.copy()
        
        # Also try the original (pre-optimization) in case optimizer degraded
        orig_obj = _objective(centers, sides)
        if orig_obj > best_obj:
            best_obj = orig_obj
            best_centers = centers.copy()
            best_sides = sides.copy()
    
    # If no valid solution found, fall back to grid
    if best_centers is None:
        centers, sides = _init_grid_packing()
        best_centers = centers
        best_sides = sides
    else:
        # Final validation
        if not _is_valid(best_centers, best_sides):
            centers, sides = _init_grid_packing()
            best_centers = centers
            best_sides = sides
        
        # Final polish: simultaneous growth on the best solution
        final_sides = _grow_all_simultaneous(best_centers, best_sides, tol=1e-7, max_iters=100)
        if _is_valid(best_centers, final_sides):
            if _objective(best_centers, final_sides) > _objective(best_centers, best_sides):
                best_sides = final_sides
    
    return best_centers, best_sides
>>>>>>> REPLACE
