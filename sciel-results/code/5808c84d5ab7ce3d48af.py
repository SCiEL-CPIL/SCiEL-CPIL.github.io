<<<<<<< SEARCH
    # 4. Geometric boosts if MDS succeeded
    coords = _mds_coordinates(d)
    if coords is not None and time.time() < _DEADLINE - 5.0:
        try:
            # Spectral gap penalty using Fiedler vector
            spectral_penalty = _spectral_gap_penalty(d, coords)
            if spectral_penalty is not None:
                heuristic *= spectral_penalty
            
            # Convex Hull boost
            # Hull edges are almost certainly in the optimal tour.
            hull = scipy.spatial.ConvexHull(coords)
            hull_boost = 2.0 # Softer boost
            for simplex in hull.simplices:
                i, j = simplex
                heuristic[i, j] *= hull_boost
                heuristic[j, i] *= hull_boost
                
        except Exception:
            pass
=======
    # 4. Geometric boosts if MDS succeeded
    coords = _mds_coordinates(d)
    if coords is not None and time.time() < _DEADLINE - 10.0:
        try:
            # Spectral gap penalty using Fiedler vector
            spectral_penalty = _spectral_gap_penalty(d, coords)
            if spectral_penalty is not None:
                heuristic *= spectral_penalty
            
            # Convex Hull boost
            # Hull edges are almost certainly in the optimal tour.
            hull = scipy.spatial.ConvexHull(coords)
            hull_boost = 2.5 # Slightly stronger boost
            for simplex in hull.simplices:
                i, j = simplex
                heuristic[i, j] *= hull_boost
                heuristic[j, i] *= hull_boost
            
            # Crossing detection penalty
            if best_tour is not None:
                crossing_penalty = _detect_crossings(best_tour, d, coords)
                if crossing_penalty is not None:
                    heuristic *= crossing_penalty
                    
        except Exception:
            pass
>>>>>>> REPLACE

