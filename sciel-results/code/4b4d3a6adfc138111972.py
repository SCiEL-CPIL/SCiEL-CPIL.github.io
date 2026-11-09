<<<<<<< SEARCH
    # Final normalization
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        best_points = best_points / norms
=======
    # Phase 2: Smooth optimization from best points
    if best_points is not None and time.time() < deadline - 60.0:
        smooth_optimized = _smooth_optimize(best_points, time.time(), deadline, max_iters=1500)
        radius, _ = _covering_radius_and_worst_facets(smooth_optimized)
        if radius < best_radius:
            best_radius = radius
            best_points = smooth_optimized.copy()
    
    # Phase 3: Final polish
    if best_points is not None and time.time() < deadline - 10.0:
        polished = _polish(best_points, time.time(), deadline, max_iters=3000)
        radius, _ = _covering_radius_and_worst_facets(polished)
        if radius < best_radius:
            best_radius = radius
            best_points = polished.copy()
    
    # Final normalization
    if best_points is not None:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        best_points = best_points / norms
>>>>>>> REPLACE
