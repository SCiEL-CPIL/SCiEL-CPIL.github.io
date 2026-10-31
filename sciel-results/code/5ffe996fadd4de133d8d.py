def _stationary_equations(a):
    p = _get_p(a)
    q = p * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
    
    Mx0 = p @ K0 @ p
    Mx1 = p @ K1 @ p
    Mxi0 = q @ K0 @ q
    Mxi1 = q @ K1 @ q
    
    if Mx0 <= 1e-15 or Mxi0 <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    vx = Mx1 / Mx0
    vxi = Mxi1 / Mxi0
    
    if vx <= 1e-15 or vxi <= 1e-15:
        return np.array([1e18, 1e18, 1e18])
        
    Sx0 = K0 @ p
    Sx1 = K1 @ p
    Sxi0 = K0 @ q
    Sxi1 = K1 @ q
    
    dMx0 = 2.0 * A_mat.T @ Sx0
    dMx1 = 2.0 * A_mat.T @ Sx1
    dMxi0 = 2.0 * A_mat.T @ Sxi0
    dMxi1 = 2.0 * A_mat.T @ Sxi1
    
    dvx = (dMx1 * Mx0 - Mx1 * dMx0) / (Mx0 * Mx0)
    dvxi = (dMxi1 * Mxi0 - Mxi1 * dMxi0) / (Mxi0 * Mxi0)
    
    grad_F = vxi * dvx + vx * dvxi
    
    return grad_F
