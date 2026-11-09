<<<<<<< SEARCH
def construct():
    deadline = time.time() + 570.0
    
    # Candidate 1: Roots of L_12^(-1/2)
    try:
        roots_lag = scipy.special.roots_genlaguerre(12, -0.5)[0]
        z_lag = [float(r) for r in roots_lag]
        # Ensure positive and distinct
        z_lag = sorted([z for z in z_lag if z > 0])
        if len(z_lag) < 12:
             # Fallback
             z_lag = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    except Exception:
        z_lag = None

    # Candidate 2: Bessel zeros squared
    z_bessel = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    best_score = float('inf')
    best_z = z_bessel
    
    # Try Bessel first as it is known to be valid and fast to generate
    if time.time() < deadline - 120:
        score_b = _compute_score_exact(z_bessel)
        if score_b is not None:
            best_score = score_b
            best_z = z_bessel
            
    # Try Laguerre roots if time permits
    if z_lag is not None and time.time() < deadline - 120:
        score_l = _compute_score_exact(z_lag)
        if score_l is not None and score_l < best_score:
            best_score = score_l
            best_z = z_lag
            
    return best_z
=======
def construct():
    deadline = time.time() + 570.0
    
    # Candidate 1: Roots of (L_13^(-1/2))'(x) = roots of L_12^(1/2)(x)
    # This is the primary candidate per the directive
    try:
        z_deriv = _laguerre_derivative_roots(13, -0.5)
        if len(z_deriv) < 12:
            raise ValueError("Not enough roots")
    except Exception:
        z_deriv = None

    # Candidate 2: Bessel zeros squared (fallback)
    z_bessel = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    
    # Candidate 3: Roots of L_12^(-1/2)(x) (original)
    try:
        z_lag = sorted([float(r) for r in scipy.special.roots_genlaguerre(12, -0.5)[0] if r > 0])
        if len(z_lag) < 12:
            z_lag = None
    except Exception:
        z_lag = None
    
    best_score = float('inf')
    best_z = z_bessel  # Safe fallback
    
    # Try derivative roots first (primary candidate)
    if z_deriv is not None and time.time() < deadline - 120:
        score_d = _compute_score_exact(z_deriv)
        if score_d is not None:
            best_score = score_d
            best_z = z_deriv
            
    # Try Laguerre roots if time permits
    if z_lag is not None and time.time() < deadline - 120:
        score_l = _compute_score_exact(z_lag)
        if score_l is not None and score_l < best_score:
            best_score = score_l
            best_z = z_lag
            
    # Try Bessel if time permits and we haven't found a good score
    if time.time() < deadline - 120:
        score_b = _compute_score_exact(z_bessel)
        if score_b is not None and score_b < best_score:
            best_score = score_b
            best_z = z_bessel
            
    return best_z
>>>>>>> REPLACE
