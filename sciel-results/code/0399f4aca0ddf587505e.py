<<<<<<< SEARCH
    # Phase 1: Generate diverse +/-1 initializations
    t_gen = deadline - 30  # leave 30s for refinement
    sources = [
        lambda: _try_random(t_gen, rng, 150),
        lambda: _try_block_patterns(t_gen, rng),
        lambda: _try_fourier(t_gen, rng),
        lambda: _try_gaussian_smoothed(t_gen, rng),
        lambda: _try_m_sequences(t_gen, rng),
    ]
    for src in sources:
        if time.time() > t_gen - 5:
            break
        x, v = src()
        consider(x, v)
=======
    # Phase 1: Parametric search and diverse initializations
    t_gen = deadline - 30  # leave 30s for refinement
    
    # Run parametric DE first as it's a structural change
    if time.time() < t_gen:
        x, v = _try_parametric_de(t_gen - 5, rng)
        consider(x, v)
        
    sources = [
        lambda: _try_random(t_gen, rng, 150),
        lambda: _try_block_patterns(t_gen, rng),
        lambda: _try_fourier(t_gen, rng),
        lambda: _try_gaussian_smoothed(t_gen, rng),
        lambda: _try_m_sequences(t_gen, rng),
    ]
    for src in sources:
        if time.time() > t_gen - 5:
            break
        x, v = src()
        consider(x, v)
>>>>>>> REPLACE
