<<<<<<< SEARCH
    # Phase 8: High-precision final polish (includes SLSQP + epigraph)
    if time_left() > 8.0:
        bary_hp = _polish_high_precision(best_bary, maxiter=400)
        score_hp = _score_bary(bary_hp)
        if score_hp > best_score:
            best_score = score_hp
            best_bary = bary_hp.copy()
    
    # Phase 9: Final active-set + SLSQP + epigraph combo
    if time_left() > 5.0:
        bary_as = _active_set_refine(best_bary, maxiter=100)
        score_as = _score_bary(bary_as)
        if score_as > best_score:
            best_score = score_as
            best_bary = bary_as.copy()
        
        if time_left() > 4.0:
            bary_epi = _epigraph_refine(best_bary, maxiter=150)
            score_epi = _score_bary(bary_epi)
            if score_epi > best_score:
                best_score = score_epi
                best_bary = bary_epi.copy()
        
        if time_left() > 3.0:
            bary_slsqp = _slsqp_refine(best_bary, maxiter=100)
            score_slsqp = _score_bary(bary_slsqp)
            if score_slsqp > best_score:
                best_score = score_slsqp
                best_bary = bary_slsqp.copy()
=======
    # Phase 8: High-precision final polish
    if time_left() > 10.0:
        bary_hp = _polish_high_precision(best_bary, maxiter=500)
        score_hp = _score_bary(bary_hp)
        if score_hp > best_score:
            best_score = score_hp
            best_bary = bary_hp.copy()
    
    # Phase 9: Final active-set + SLSQP + epigraph combo
    if time_left() > 6.0:
        bary_as = _active_set_refine(best_bary, maxiter=150)
        score_as = _score_bary(bary_as)
        if score_as > best_score:
            best_score = score_as
            best_bary = bary_as.copy()
        
        if time_left() > 4.0:
            bary_epi = _epigraph_refine(best_bary, maxiter=200)
            score_epi = _score_bary(bary_epi)
            if score_epi > best_score:
                best_score = score_epi
                best_bary = bary_epi.copy()
        
        if time_left() > 3.0:
            bary_slsqp = _slsqp_refine(best_bary, maxiter=150)
            score_slsqp = _score_bary(bary_slsqp)
            if score_slsqp > best_score:
                best_score = score_slsqp
                best_bary = bary_slsqp.copy()
>>>>>>> REPLACE
