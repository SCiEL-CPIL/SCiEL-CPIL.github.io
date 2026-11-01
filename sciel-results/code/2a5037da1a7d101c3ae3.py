def _bound(params):
    a0, a4, a8 = params[0], params[1], params[2]
    a12 = -(a0 + 3.0*a4 + 105.0*a8) / 945.0
    
    coeffs = np.array([a0, a4, a8, a12])
    p = coeffs @ _H
    
    # Construct moment matrices M0, M2, N0, N2
    # M0[i,j] = I[i+j]
    # M2[i,j] = I[1+i+j]
    # N0[i,j] = 2*pi * I[i+j]
    # N2[i,j] = 2*pi * I[1+i+j]
    # We can precompute these matrices.
    # But p changes, so we just do p @ M @ p
    
    m0 = p @ _M0 @ p
    m2 = p @ _M2 @ p
    
    q = p * _signs
    n0 = q @ _N0 @ q
    n2 = q @ _N2 @ q
    
    if m0 <= 1e-30 or n0 <= 1e-30 or not np.isfinite(m0) or not np.isfinite(n0):
        return 1e18
    
    vf = m2 / m0
    vF = n2 / n0
    
    if vf <= 0 or vF <= 0:
        return 1e18
    
    return np.sqrt(vf * vF)
