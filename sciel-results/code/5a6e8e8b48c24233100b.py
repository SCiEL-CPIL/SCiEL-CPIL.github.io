import scipy.special

def construct():
    # The knowledge indicates that squares of Bessel J0 zeros are optimal.
    # J0 zeros correspond to the asymptotic distribution of L_n^(-1/2) zeros.
    # We need 12 distinct positive values.
    # scipy.special.jn_zeros(n, nt) returns the first nt zeros of J_n.
    # n=0 for J0.
    try:
        # Get 12 zeros of J0
        j0_zeros = scipy.special.jn_zeros(0, 12)
        # Square them to get the Laguerre-like locations
        z = [float(x**2) for x in j0_zeros]
        return z
    except Exception:
        # Fallback if scipy fails, though unlikely
        # Use a heuristic spacing based on the known density
        # z_i approx (pi/2 * (i + 0.25))^2 for large i, but let's just use a simple spread
        # The prompt says uniform is bad, but better than invalid.
        # However, scipy is available.
        return [4.0 * (i + 1) for i in range(12)]
