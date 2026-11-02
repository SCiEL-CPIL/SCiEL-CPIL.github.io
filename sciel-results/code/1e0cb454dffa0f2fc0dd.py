<<<<<<< SEARCH
    # Step 5: Annealing refinement
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 8
        if remaining > 8:
            c_ann, r_ann = _anneal_refine(best_c, max_time=min(12.0, remaining))
            if r_ann > best_radius:
                best_radius = r_ann
                best_c = c_ann.copy()
    # Step 6: SLSQP polish with hard constraints
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            c_slsqp, r_slsqp = _slsqp_polish(best_c, max_time=min(15.0, remaining))
            if r_slsqp > best_radius:
                best_radius = r_slsqp
                best_c = c_slsqp.copy()
    # Step 7: Final high-precision polish
    if time.time() < deadline - 4:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            c_pol, r_pol = _polish(best_c, max_time=min(18.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
=======
    # Step 5: Annealing refinement
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 8
        if remaining > 8:
            c_ann, r_ann = _anneal_refine(best_c, max_time=min(12.0, remaining))
            if r_ann > best_radius:
                best_radius = r_ann
                best_c = c_ann.copy()
    # Step 5b: Rigid cluster annealing
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 8
        if remaining > 8:
            c_rca, r_rca = _rigid_cluster_anneal(best_c, max_time=min(18.0, remaining))
            if r_rca > best_radius:
                best_radius = r_rca
                best_c = c_rca.copy()
            # Re-optimize after rigid cluster annealing
            if time.time() < deadline - 12:
                max_opt_time = min(15.0, deadline - time.time() - 8)
                c_opt, r_opt = _local_optimize(best_c, max_time=max_opt_time)
                if r_opt > best_radius:
                    best_radius = r_opt
                    best_c = c_opt.copy()
    # Step 6: SLSQP polish with hard constraints
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            c_slsqp, r_slsqp = _slsqp_polish(best_c, max_time=min(15.0, remaining))
            if r_slsqp > best_radius:
                best_radius = r_slsqp
                best_c = c_slsqp.copy()
    # Step 7: Final high-precision polish
    if time.time() < deadline - 4:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            c_pol, r_pol = _polish(best_c, max_time=min(18.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
>>>>>>> REPLACE

