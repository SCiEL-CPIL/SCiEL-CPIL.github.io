<<<<<<< SEARCH
    # Phase 4: If time remains, do a final polish
    if time.time() < deadline - 5:
        best_p, best_cost = local_search(best_p.copy(), best_cost, deadline - 2)

    return best_p.tolist()
=======
    # Phase 4: If time remains, do a final polish with LAP and 2-opt
    if time.time() < deadline - 5:
        # Try LAP refinement one more time
        p_lap = lap_refine(best_p)
        cost_lap = compute_cost(p_lap)
        
        # Local search on the LAP result
        p_lap, cost_lap = local_search(p_lap, cost_lap, deadline - 2)
        
        if cost_lap < best_cost:
            best_p = p_lap
            best_cost = cost_lap
        else:
            # If LAP didn't help, polish the current best with 2-opt
            best_p, best_cost = local_search(best_p.copy(), best_cost, deadline - 2)

    return best_p.tolist()
>>>>>>> REPLACE
