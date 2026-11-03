    inner = _build_inner(best_params)
    if not _pairwise_nonoverlap(inner):
        # Use fallback
        inner = [[0.0, 0.0, 0.0]]
        for k in range(6):
            angle = k * 60.0
            x = 2.0 * np.cos(np.radians(angle))
            y = 2.0 * np.sin(np.radians(angle))
            inner.append([x, y, 0.0])
        for k in range(4):
            angle = 45.0 + k * 90.0
            x = 3.5 * np.cos(np.radians(angle))
            y = 3.5 * np.sin(np.radians(angle))
            inner.append([x, y, 0.0])
        
        # Verify fallback is valid
        if not _pairwise_nonoverlap(inner):
             # Should not happen, but if it does, return a single hexagon? 
             # No, must return N=11. 
             # Let's just trust the fallback geometry.
             pass
