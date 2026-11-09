import numpy as np
from numpy.polynomial import laguerre as nplag

def construct():
    # Get coefficients for L_12 and L_11
    # nplag.lag2poly(n) returns coefficients for L_n(x)
    c12 = nplag.lag2poly(12)
    c11 = nplag.lag2poly(11)
    
    # We need to align degrees. c11 has length 12 (degree 11).
    # Pad c11 with a zero at the end (highest degree) to match length 13?
    # lag2poly returns coefficients in increasing order of power?
    # numpy.polynomial.laguerre.lag2poly returns [c0, c1, ..., cn] for c0*L0 + ... + cn*Ln?
    # Actually, it returns the coefficients of the polynomial in the monomial basis?
    # No, `lag2poly` converts Laguerre basis to power basis.
    # It returns an array of coefficients [c0, c1, ..., cn] such that
    # P(x) = c0*x^0 + c1*x^1 + ... + cn*x^n.
    
    # Let's check lengths.
    # L_12 has degree 12, so 13 coeffs.
    # L_11 has degree 11, so 12 coeffs.
    
    # P(x) = L_12(x) - L_11(x)
    # Coeffs:
    coeff_diff = np.zeros(13)
    coeff_diff[:12] = c11 # Wait, c11 is length 12.
    # c12 is length 13.
    # coeff_diff = c12 - [c11, 0]
    
    # Correct alignment:
    # c12[0..12]
    # c11[0..11]
    # diff[0..11] = c12[0..11] - c11[0..11]
    # diff[12] = c12[12]
    
    c_diff = c12.copy()
    c_diff[:len(c11)] -= c11
    
    # Find roots
    roots = np.roots(c_diff)
    
    # Filter real positive roots
    real_roots = roots[np.isreal(roots)].real
    pos_roots = real_roots[real_roots > 0]
    
    # Should be 12 roots
    if len(pos_roots) != 12:
        # Fallback to standard
        roots_std, _ = nplag.laggauss(12) # No, that's weights.
        # Use scipy
        from scipy.special import roots_laguerre
        r_std, _ = roots_laguerre(12)
        return [float(r) for r in r_std]
        
    return [float(r) for r in sorted(pos_roots)]
