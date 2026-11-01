<<<<<<< SEARCH
def _polish(centers, start_time):
    """Progressively polish the solution with increasing beta, force-directed, and NM."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 2500.0]
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=80, gtol=1e-10)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
        
        # Add force-directed refinement after each beta stage
        if time.time() - start_time < _DEADLINE - 20:
            refined, refined_score = _force_directed_refine(current, start_time, max_iter=50, step_size=0.0003)
            if refined_score > best_score:
                best_score = refined_score
                current = refined
    
    # Nelder-Mead polish on the nonsmooth objective
    if time.time() - start_time < _DEADLINE - 12:
        nm_result = _nelder_mead_polish(current, maxiter=500)
        if np.all(np.isfinite(nm_result)):
            r = _compute_score(nm_result)
            if r > best_score:
                best_score = r
                current = nm_result
    
    # Final high-precision polish
    if time.time() - start_time < _DEADLINE - 5:
        final = _optimize_from_start(current, beta=5000.0, maxiter=100, gtol=1e-12)
        if np.all(np.isfinite(final)):
            r = _compute_score(final)
            if r > best_score:
                best_score = r
                current = final
    
    return current, best_score
=======
def _polish(centers, start_time):
    """Progressively polish the solution with increasing beta, force-directed, and NM."""
    current = centers.copy()
    best_score = _compute_score(current)
    
    betas = [50.0, 100.0, 200.0, 400.0, 800.0, 1500.0, 2500.0, 5000.0, 10000.0, 25000.0]
    for beta in betas:
        if time.time() - start_time > _DEADLINE - 15:
            break
        
        polished = _optimize_from_start(current, beta, maxiter=100, gtol=1e-11)
        if np.all(np.isfinite(polished)):
            r = _compute_score(polished)
            if r > best_score:
                best_score = r
                current = polished
        
        # Add force-directed refinement after each beta stage
        if time.time() - start_time < _DEADLINE - 15:
            refined, refined_score = _force_directed_refine(current, start_time, max_iter=60, step_size=0.0002)
            if refined_score > best_score:
                best_score = refined_score
                current = refined
    
    # Nelder-Mead polish on the nonsmooth objective
    if time.time() - start_time < _DEADLINE - 10:
        nm_result = _nelder_mead_polish(current, maxiter=800)
        if np.all(np.isfinite(nm_result)):
            r = _compute_score(nm_result)
            if r > best_score:
                best_score = r
                current = nm_result
    
    # Final ultra-high-precision polish with very high beta
    if time.time() - start_time < _DEADLINE - 5:
        final = _optimize_from_start(current, beta=100000.0, maxiter=200, gtol=1e-13)
        if np.all(np.isfinite(final)):
            r = _compute_score(final)
            if r > best_score:
                best_score = r
                current = final
        
        # One more polish pass
        if time.time() - start_time < _DEADLINE - 3:
            final2 = _optimize_from_start(current, beta=500000.0, maxiter=150, gtol=1e-14)
            if np.all(np.isfinite(final2)):
                r = _compute_score(final2)
                if r > best_score:
                    best_score = r
                    current = final2
    
    return current, best_score
>>>>>>> REPLACE

