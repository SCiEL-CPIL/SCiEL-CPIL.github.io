<<<<<<< SEARCH
    # Initial guess: try several heuristics
    # Heuristic 1: Keich-like construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps_bits = np.array(list(bin(idx)[2:]), dtype=int)
        eps_bits = np.pad(eps_bits, (pos - len(eps_bits), 0), 'constant')
        keich[idx] = np.sum((1 - j_terms) * eps_bits * (2.0 ** -j_terms) / pos)
    
    # Heuristic 2: Uniform spread
    uniform = np.linspace(0, 0.5, N)
    
    # Heuristic 3: Random restarts
    rng_init = np.random.default_rng(42)
    
    # Start with best of initial guesses
    best_v = float('inf')
    best_p = None
    
    candidates = [
        np.concatenate([keich, keich]),
        np.concatenate([uniform, uniform]),
    ]
    
    for _ in range(5):
        rand_x = rng_init.uniform(0, 0.5, N)
        rand_y = rng_init.uniform(0, 0.5, N)
        candidates.append(np.concatenate([rand_x, rand_y]))
    
    for p0 in candidates:
        v = objective(p0)
        if v < best_v:
            best_v = v
            best_p = p0.copy()
=======
    # Initial guess: try several heuristics
    # Heuristic 1: Keich-like construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps_bits = np.array(list(bin(idx)[2:]), dtype=int)
        eps_bits = np.pad(eps_bits, (pos - len(eps_bits), 0), 'constant')
        keich[idx] = np.sum((1 - j_terms) * eps_bits * (2.0 ** -j_terms) / pos)
    
    # Heuristic 2: Uniform spread
    uniform = np.linspace(0, 0.5, N)
    
    # Heuristic 3: Random restarts
    rng_init = np.random.default_rng(42)
    
    # Start with best of initial guesses
    best_v = float('inf')
    best_p = None
    
    candidates = [
        np.concatenate([keich, keich]),
        np.concatenate([uniform, uniform]),
    ]
    
    for _ in range(10):
        rand_x = rng_init.uniform(-0.1, 0.6, N)
        rand_y = rng_init.uniform(-0.1, 0.6, N)
        candidates.append(np.concatenate([rand_x, rand_y]))
    
    for p0 in candidates:
        v = objective(p0)
        if v < best_v:
            best_v = v
            best_p = p0.copy()
>>>>>>> REPLACE

