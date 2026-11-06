def _spectral_gradient(pts, base_pts, coeffs, grad_basis):
    """Compute gradient of energy with respect to spectral coefficients."""
    energy, grad = _energy_and_grad(pts)
    
    # grad is the gradient of energy w.r.t Cartesian coordinates of pts.
    # dE/dc_lm = sum_i grad_i . d(pts_i)/dc_lm
    # d(pts_i)/dc_lm = grad_basis[(l, m)][i] (since it's already tangent)
    
    coeff_grad = {}
    for (l, m) in coeffs:
        # Dot product of grad (N,3) and grad_basis[(l,m)] (N,3)
        coeff_grad[(l, m)] = np.sum(grad * grad_basis[(l, m)])
        
    return energy, coeff_grad
