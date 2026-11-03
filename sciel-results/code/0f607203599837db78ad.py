<<<<<<< SEARCH
def _laguerre_roots(m):
    """Get the m positive roots of the Laguerre polynomial L_m(x)."""
    # Use numpy's polynomial roots for Laguerre
    # L_m(x) = sum_{k=0}^{m} (-1)^k * C(m,k) * x^k / k!
    # Coefficients from highest degree
    coeffs = np.zeros(m + 1)
    for k in range(m + 1):
        coeffs[m - k] = ((-1) ** k) * np.math.comb(m, k) / np.math.factorial(k)
    roots = np.roots(coeffs)
    # Take real positive roots
    roots = roots[np.abs(roots.imag) < 1e-10].real
    roots = roots[roots > 0]
    roots = np.sort(roots)
    return roots
=======
def _laguerre_roots(m):
    """Get the m positive roots of the Laguerre polynomial L_m(x)."""
    # Use numpy's polynomial roots for Laguerre
    # L_m(x) = sum_{k=0}^{m} (-1)^k * C(m,k) * x^k / k!
    # Coefficients from highest degree
    coeffs = np.zeros(m + 1)
    for k in range(m + 1):
        coeffs[m - k] = ((-1) ** k) * math.comb(m, k) / math.factorial(k)
    roots = np.roots(coeffs)
    # Take real positive roots
    roots = roots[np.abs(roots.imag) < 1e-10].real
    roots = roots[roots > 0]
    roots = np.sort(roots)
    return roots
>>>>>>> REPLACE
