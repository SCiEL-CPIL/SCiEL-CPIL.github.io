    # Calculate optimal center C
    M0 = (np.max(p_best @ n0) + np.min(p_best @ n0)) / 2.0
    M1 = (np.max(p_best @ n1) + np.min(p_best @ n1)) / 2.0
    M2 = (np.max(p_best @ n2) + np.min(p_best @ n2)) / 2.0
    
    # Use n0 and n1 to find C
    A = np.array([n0, n1]).T
    M = np.array([M0, M1])
    try:
        center = np.linalg.solve(A, M)
    except np.linalg.LinAlgError:
        center = np.mean(p_best, axis=0)
        
    # Verify and adjust S if needed
    max_dist = 0
    for p in p_best:
        d0 = abs((p - center) @ n0)
        d1 = abs((p - center) @ n1)
        d2 = abs((p - center) @ n2)
        max_dist = max(max_dist, max(d0, d1, d2))
        
    S_safe = 2 * max_dist / np.sqrt(3)
    S_final = max(S_best, S_safe)
