<<<<<<< SEARCH
def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Generate diverse initializations
    i = np.arange(N)
    initializations = []
    
    # Cosine
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    initializations.append(_fix_sum(h_cos))
    
    # Sine
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    initializations.append(_fix_sum(h_sin))
    
    # Alternating
    h_alt = np.where(i % 2 == 0, 0.7, 0.3)
    initializations.append(_fix_sum(h_alt))
    
    # Block patterns
    for block_size in [5, 10, 20]:
        h_block = np.zeros(N)
        for start in range(0, N, block_size * 2):
            h_block[start:start+block_size] = 0.8
            h_block[start+block_size:start+2*block_size] = 0.2
        initializations.append(_fix_sum(h_block))
    
    # Random
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        initializations.append(_fix_sum(h_rand))
    
    # 2. Run PGD from each initialization
    for h_init in initializations:
        if time.time() > _DEADLINE - 30:
            break
        h_pgd = _pgd_optimize(h_init, beta=50.0, max_iter=3000)
        _update_best(h_pgd)
    
    # 3. SLSQP polish on best
    if best_h is not None and time.time() < _DEADLINE - 20:
        h_slsqp = _run_slsqp(best_h, beta=50.0, maxiter=200, ftol=1e-10, gtol=1e-8)
        _update_best(h_slsqp)
    
    # 4. Higher beta PGD polish
    if best_h is not None and time.time() < _DEADLINE - 15:
        h_pgd_high = _pgd_optimize(best_h, beta=100.0, max_iter=2000)
        _update_best(h_pgd_high)
    
    # 5. SLSQP with higher beta
    if best_h is not None and time.time() < _DEADLINE - 10:
        h_slsqp_high = _run_slsqp(best_h, beta=100.0, maxiter=300, ftol=1e-12, gtol=1e-10)
        _update_best(h_slsqp_high)
    
    # 6. Final ultra-precise polish
    if best_h is not None and time.time() < _DEADLINE - 5:
        h_final = _polish(best_h, beta=200.0, maxiter=500, ftol=1e-14, gtol=1e-12)
        _update_best(h_final)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
=======
def _coordinate_descent_polish(h, beta, max_sweeps=200, tol=1e-12):
    """
    Coordinate descent on the active set (indices where 0 < h_i < 1)
    to shave off residual error that smooth-surrogate methods miss.
    """
    h = _fix_sum(h.copy())
    cs = _overlap_values(h)
    active_mask = (h > 1e-10) & (h < 1.0 - 1e-10)
    active_idx = np.where(active_mask)[0]
    
    if len(active_idx) < 2:
        return h
    
    best_val = _score(h)
    best_h = h.copy()
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 3:
            break
        improved = False
        for idx in active_idx:
            if time.time() > _DEADLINE - 3:
                break
            # Try small perturbations
            for delta in [1e-4, -1e-4, 5e-5, -5e-5, 1e-5, -1e-5]:
                h_trial = h.copy()
                h_trial[idx] += delta
                h_trial = _fix_sum(h_trial)
                if np.any(h_trial < 0) or np.any(h_trial > 1):
                    continue
                s = _score(h_trial)
                if s < best_val - 1e-14:
                    h = h_trial
                    best_val = s
                    best_h = h.copy()
                    improved = True
                    break
        if not improved:
            break
    
    return best_h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Generate diverse initializations
    i = np.arange(N)
    initializations = []
    
    # Cosine
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    initializations.append(_fix_sum(h_cos))
    
    # Sine
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    initializations.append(_fix_sum(h_sin))
    
    # Alternating
    h_alt = np.where(i % 2 == 0, 0.7, 0.3)
    initializations.append(_fix_sum(h_alt))
    
    # Block patterns
    for block_size in [5, 10, 20]:
        h_block = np.zeros(N)
        for start in range(0, N, block_size * 2):
            h_block[start:start+block_size] = 0.8
            h_block[start+block_size:start+2*block_size] = 0.2
        initializations.append(_fix_sum(h_block))
    
    # More structured patterns
    for freq in [2, 3, 4]:
        h_struct = 0.5 + 0.4 * np.cos(2.0 * np.pi * freq * i / N)
        initializations.append(_fix_sum(h_struct))
    
    # Random
    for _ in range(8):
        h_rand = np.random.uniform(0, 1, N)
        initializations.append(_fix_sum(h_rand))
    
    # 2. Run Nesterov PGD from each initialization
    for h_init in initializations:
        if time.time() > _DEADLINE - 25:
            break
        h_pgd = _pgd_optimize(h_init, beta=50.0, max_iter=4000)
        _update_best(h_pgd)
    
    # 3. SLSQP polish on best
    if best_h is not None and time.time() < _DEADLINE - 18:
        h_slsqp = _run_slsqp(best_h, beta=50.0, maxiter=200, ftol=1e-10, gtol=1e-8)
        _update_best(h_slsqp)
    
    # 4. Higher beta PGD polish
    if best_h is not None and time.time() < _DEADLINE - 12:
        h_pgd_high = _pgd_optimize(best_h, beta=100.0, max_iter=3000)
        _update_best(h_pgd_high)
    
    # 5. SLSQP with higher beta
    if best_h is not None and time.time() < _DEADLINE - 8:
        h_slsqp_high = _run_slsqp(best_h, beta=100.0, maxiter=300, ftol=1e-12, gtol=1e-10)
        _update_best(h_slsqp_high)
    
    # 6. Coordinate descent polish on active set
    if best_h is not None and time.time() < _DEADLINE - 4:
        h_cd = _coordinate_descent_polish(best_h, beta=200.0, max_sweeps=100)
        _update_best(h_cd)
    
    # 7. Final ultra-precise SLSQP
    if best_h is not None and time.time() < _DEADLINE - 2:
        h_final = _polish(best_h, beta=200.0, maxiter=500, ftol=1e-14, gtol=1e-12)
        _update_best(h_final)
    
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
>>>>>>> REPLACE
