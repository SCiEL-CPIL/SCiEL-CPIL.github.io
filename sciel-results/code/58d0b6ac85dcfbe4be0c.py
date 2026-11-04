def _upper_bound_C4(c0, c4, c8):
    """
    Compute the upper bound on A(f) * A(Fourier(f)) for the test function
    g(x) = (c0*H0(x) + c4*H4(x) + c8*H8(x) + c12*H12(x)) * exp(-x^2/2)
    where c12 is chosen so g(0) = 0.
    
    Note: The original code omitted the Gaussian factor, leading to incorrect norms.
    We use the analytic property that the Fourier transform of H_n(x)exp(-x^2/2)
    is i^n sqrt(2pi) H_n(xi)exp(-xi^2/2).
    """
    c0, c4, c8, c12 = _compute_coefficients(c0, c4, c8)
    
    # Grid for integration
    # The function decays as exp(-x^2/2), so [-10, 10] is sufficient for high precision
    x = np.linspace(-10, 10, 4096)
    
    H0 = _hermite_polynomial(0, x)
    H4 = _hermite_polynomial(4, x)
    H8 = _hermite_polynomial(8, x)
    H12 = _hermite_polynomial(12, x)
    
    # Gaussian factor
    gauss = np.exp(-x**2 / 2.0)
    
    # g(x) in spatial domain
    g = (c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12) * gauss
    
    # Check if g is essentially zero
    if np.max(np.abs(g)) < 1e-15:
        return 1e10
    
    # Compute L1 and L2 norms of g
    dx = x[1] - x[0]
    L1_g = np.trapz(np.abs(g), x)
    L2_g = np.sqrt(np.trapz(g**2, x))
    
    if L2_g < 1e-15:
        return 1e10
    
    A_g = L1_g / L2_g
    
    # Fourier transform in frequency domain
    # Using convention F(xi) = integral g(x) exp(-i x xi) dx
    # Identity: F[H_n(x) exp(-x^2/2)] = i^n sqrt(2pi) H_n(xi) exp(-xi^2/2)
    # i^0 = 1, i^4 = 1, i^8 = 1, i^12 = -1
    # So F(g)(xi) = sqrt(2pi) * (c0*H0(xi) + c4*H4(xi) + c8*H8(xi) - c12*H12(xi)) * exp(-xi^2/2)
    
    # The grid for xi is the same as x due to symmetry of the basis
    H0_f = _hermite_polynomial(0, x)
    H4_f = _hermite_polynomial(4, x)
    H8_f = _hermite_polynomial(8, x)
    H12_f = _hermite_polynomial(12, x)
    
    # Note: H_n(x) is the same polynomial evaluated at x. Since the grid is symmetric, 
    # we can reuse the values if we just change the coefficients.
    # But careful: H_n(-x) = (-1)^n H_n(x). The grid is symmetric, so the set of values is the same.
    
    g_f = (c0 * H0_f + c4 * H4_f + c8 * H8_f - c12 * H12_f) * gauss * np.sqrt(2 * np.pi)
    
    # Compute L1 and L2 norms of g_f
    L1_gf = np.trapz(np.abs(g_f), x)
    L2_gf = np.sqrt(np.trapz(g_f**2, x))
    
    if L2_gf < 1e-15:
        return 1e10
    
    A_gf = L1_gf / L2_gf
    
    bound = A_g * A_gf
    
    return bound
