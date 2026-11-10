<<<<<<< SEARCH
def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Returns (centers, sides) maximizing sum of side lengths.
    """
    start_time = time.time()
    deadline = start_time + 570.0
    
    inits = [
        _init_grid_packing(),
        _init_heterogeneous_1(),
        _init_heterogeneous_2(),
        _init_heterogeneous_3(),
        _init_heterogeneous_4(),
        _init_heterogeneous_5(),
        _init_heterogeneous_6(),
    ]
    
    # Add random inits
    for s in [123, 456, 789, 1011, 2020, 3030]:
        inits.append(_init_random(seed=s))
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for idx, (centers, sides) in enumerate(inits):
        if time.time() > deadline:
            break
        
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        
        # Resolve overlaps
        centers = _resolve_overlaps(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        
        if not _is_valid(centers, sides):
            # Try harder to resolve
            centers = _force_based_separation(centers, sides, max_iters=1000)
            centers = _clamp_to_unit(centers, sides)
            if not _is_valid(centers, sides):
                continue
        
        # Local optimization with unique seed per init
        opt_centers, opt_sides = _local_optimize(centers, sides, seed=42 + idx * 100)
        
        if _is_valid(opt_centers, opt_sides):
            obj = _objective(opt_centers, opt_sides)
            if obj > best_obj:
                best_obj = obj
                best_centers = opt_centers.copy()
                best_sides = opt_sides.copy()
        
        # If we have a good result, try to improve it further
        if best_obj > 1.15 and time.time() < deadline - 60:
            # Additional shake-and-reground on best
            rng = np.random.default_rng(seed=999 + idx)
            for _ in range(5):
                result = _shake_and_regrow(best_centers, best_sides, rng=rng)
                if result is not None:
                    new_c, new_s = result
                    obj = _objective(new_c, new_s)
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        best_centers = new_c.copy()
                        best_sides = new_s.copy()
    
    if best_centers is None:
        centers, sides = _init_grid_packing()
        best_centers = centers
        best_sides = sides
    else:
        if not _is_valid(best_centers, best_sides):
            centers, sides = _init_grid_packing()
            best_centers = centers
            best_sides = sides
        
        # Final high-precision polish
        for _ in range(3):
            final_sides = best_sides.copy()
            for i in range(10):
                ns = _grow_square_max(best_centers, final_sides, i, tol=1e-9)
                if ns > final_sides[i] + 1e-12:
                    final_sides[i] = ns
            if _is_valid(best_centers, final_sides, tol=1e-7):
                if _objective(best_centers, final_sides) > _objective(best_centers, best_sides):
                    best_sides = final_sides.copy()
    
    return best_centers, best_sides
=======
def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Returns (centers, sides) maximizing sum of side lengths.
    """
    start_time = time.time()
    deadline = start_time + 570.0
    
    inits = [
        _init_grid_packing(),
        _init_heterogeneous_1(),
        _init_heterogeneous_2(),
        _init_heterogeneous_3(),
        _init_heterogeneous_4(),
        _init_heterogeneous_5(),
        _init_heterogeneous_6(),
    ]
    
    # Add random inits
    for s in [123, 456, 789, 1011, 2020, 3030, 4040, 5050, 6060, 7070]:
        inits.append(_init_random(seed=s))
    
    best_centers = None
    best_sides = None
    best_obj = -1.0
    
    for idx, (centers, sides) in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        centers = centers.copy()
        sides = sides.copy()
        centers = _clamp_to_unit(centers, sides)
        
        # Resolve overlaps
        centers = _resolve_overlaps(centers, sides)
        centers = _clamp_to_unit(centers, sides)
        
        if not _is_valid(centers, sides):
            # Try harder to resolve
            centers = _force_based_separation(centers, sides, max_iters=1000, lr=0.08)
            centers = _clamp_to_unit(centers, sides)
            if not _is_valid(centers, sides):
                continue
        
        # Local optimization with unique seed per init
        opt_centers, opt_sides = _local_optimize(centers, sides, seed=42 + idx * 100)
        
        if _is_valid(opt_centers, opt_sides):
            obj = _objective(opt_centers, opt_sides)
            if obj > best_obj:
                best_obj = obj
                best_centers = opt_centers.copy()
                best_sides = opt_sides.copy()
        
        # If we have a good result, try to improve it further
        if best_obj > 1.15 and time.time() < deadline - 60:
            # Additional shake-and-reground on best
            rng = np.random.default_rng(seed=999 + idx)
            for _ in range(8):
                result = _shake_and_regrow(best_centers, best_sides, rng=rng)
                if result is not None:
                    new_c, new_s = result
                    obj = _objective(new_c, new_s)
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        best_centers = new_c.copy()
                        best_sides = new_s.copy()
    
    if best_centers is None:
        centers, sides = _init_grid_packing()
        best_centers = centers
        best_sides = sides
    else:
        if not _is_valid(best_centers, best_sides):
            centers, sides = _init_grid_packing()
            best_centers = centers
            best_sides = sides
        
        # Final high-precision polish - more iterations
        for _ in range(5):
            final_sides = best_sides.copy()
            for i in range(10):
                ns = _grow_square_max(best_centers, final_sides, i, tol=1e-10)
                if ns > final_sides[i] + 1e-12:
                    final_sides[i] = ns
            if _is_valid(best_centers, final_sides, tol=1e-7):
                if _objective(best_centers, final_sides) > _objective(best_centers, best_sides):
                    best_sides = final_sides.copy()
        
        # One more round of force separation + regrow on best
        if time.time() < deadline - 10:
            sep_centers = _force_based_separation(best_centers.copy(), best_sides, tol, max_iters=500, lr=0.05)
            if _is_valid(sep_centers, best_sides, tol):
                sep_sides = _grow_all_iteratively(sep_centers.copy(), best_sides.copy(), tol)
                if _is_valid(sep_centers, sep_sides, tol):
                    obj = _objective(sep_centers, sep_sides)
                    if obj > _objective(best_centers, best_sides) + 1e-9:
                        best_centers = sep_centers.copy()
                        best_sides = sep_sides.copy()
    
    return best_centers, best_sides
>>>>>>> REPLACE
