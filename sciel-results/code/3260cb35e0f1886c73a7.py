def _thue_morse(n):
    """Generate Thue-Morse sequence."""
    t = [1]
    while len(t) < n:
        t += [-x for x in t]
    return np.array(t[:n], dtype=np.float64)

def construct():
    best_s = None
    best_peak = float('inf')
    
    inits = []
    # 1. All ones
    inits.append(np.ones(N))
    # 2. Alternating
    inits.append(np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.float64))
    # 3. Thue-Morse
    inits.append(_thue_morse(N))
    
    # 4. Random restarts
    np.random.seed(42)
    
    # Process initial inits first
    for s_init in inits:
        if _time_left() < 10.0:
            break
        s_opt, peak = _local_search(s_init.copy())
        if peak < best_peak:
            best_peak = peak
            best_s = s_opt.copy()
            
    # Then random restarts
    while _time_left() > 10.0:
        s = np.random.choice([1, -1], size=N).astype(np.float64)
        s_opt, peak = _local_search(s)
        if peak < best_peak:
            best_peak = peak
            best_s = s_opt.copy()
            
    if best_s is None:
        best_s = np.ones(N)
        
    return best_s
