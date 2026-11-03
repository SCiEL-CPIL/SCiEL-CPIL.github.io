<<<<<<< SEARCH
    # Try SLSQP polish if time permits
    if time.time() < _DEADLINE - 30.0:
        polished = _slsqp_polish(inner, max_iter=100, tol=1e-8, beta=50.0)
        if polished is not inner:
            pcx, pcy, pside, pangle = _min_enclosing_hex_side(polished)
            if pside < best_side - 1e-6:
                best_side = pside
                best_inner = polished
                best_center = (pcx, pcy)
                best_angle = pangle
=======
    # Try differential evolution global search if time permits
    if time.time() < _DEADLINE - 40.0:
        de_result = _differential_evolution_global(inner, max_iter=30, pop_size=20)
        if de_result is not None:
            dcx, dcy, dside, dangle = _min_enclosing_hex_side(de_result)
            if dside < best_side - 1e-6:
                best_side = dside
                best_inner = de_result
                best_center = (dcx, dcy)
                best_angle = dangle
    
    # Try SLSQP polish if time permits
    if time.time() < _DEADLINE - 30.0:
        polished = _slsqp_polish(best_inner, max_iter=100, tol=1e-8, beta=50.0)
        if polished is not best_inner:
            pcx, pcy, pside, pangle = _min_enclosing_hex_side(polished)
            if pside < best_side - 1e-6:
                best_side = pside
                best_inner = polished
                best_center = (pcx, pcy)
                best_angle = pangle
>>>>>>> REPLACE

