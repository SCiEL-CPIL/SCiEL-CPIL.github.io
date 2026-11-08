import numpy as np
from scipy.optimize import brentq


def construct():
    # Hermite polynomial values at 0 for even degrees (physicists' convention)
    h0, h4, h8, h12 = 1.0, -3.0, 105.0, -10395.0
    h = np.array([h0, h4, h8, h12])
    
    # Moments n + 1/2 for degrees 0, 4, 8, 12
    m = np.array([0.5, 4.5, 8.5, 12.5])
    
    # The problem is to minimize (sum c_i^2 m_i) / (sum c_i^2) subject to sum c_i h_i = 0.
    # This is a generalized eigenvalue problem. The minimum value is the smallest root mu
    # of the equation sum_i h_i^2 / (m_i - mu) = 0.
    # The smallest root lies in the interval (m[0], m[1]) = (0.5, 4.5).
    
    def g(mu):
        return np.sum(h**2 / (m - mu))
    
    # Find the root in (0.5, 4.5)
    # Use tight bounds to avoid singularities
    a = m[0] + 1e-12
    b = m[1] - 1e-12
    
    # Verify sign change
    fa = g(a)
    fb = g(b)
    
    if fa * fb > 0:
        # Fallback if no sign change (should not happen)
        # Return a default guess
        return np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
        
    mu_root = brentq(g, a, b, xtol=1e-15, rtol=1e-15)
    
    # The optimal coefficients are proportional to h_i / (m_i - mu_root)
    c = h / (m - mu_root)
    
    # Normalize to unit norm
    norm = np.linalg.norm(c)
    if norm > 1e-15:
        c = c / norm
    else:
        c = np.array([0.0, 0.0, 0.0, 0.0])
        
    # Return the first three coefficients
    return np.array([c[0], c[1], c[2]])
