<<<<<<< SEARCH
def _rational_potential_roots(a, b, c, d):
    """Find 12 positive critical points of V(x) = (ax^2+bx+c)/(x+d).
    V'(x) = (ax^2+bx+c)'(x+d) - (ax^2+bx+c)(x+d)' / (x+d)^2
          = (2ax+b)(x+d) - (ax^2+bx+c) / (x+d)^2
          = (2ax^2+2adx+bx+bd - ax^2-bx-c) / (x+d)^2
          = (ax^2 + (2ad)x + (bd-c)) / (x+d)^2
    So critical points are roots of ax^2 + 2ad*x + (bd-c) = 0 — only 2 roots!
    
    For 12 roots we need a more general rational potential. Let's use:
    V(x) = (x^4 + p*x^3 + q*x^2 + r*x + s) / (x + d)
    V'(x) = [ (4x^3+3px^2+2qx+r)(x+d) - (x^4+px^3+qx^2+rx+s) ] / (x+d)^2
    Numerator: 4x^4+4dx^3+3px^3+3dp*x^2+2qx^2+2dq*x+rx+rd - x^4-px^3-qx^2-rx-s
    = 3x^4 + (4d+3p)x^3 + (3dp+2q-q)x^2 + (2dq+r-r)x + (rd-s)
    = 3x^4 + (4d+3p)x^3 + (3dp+x)x^2 + 2dq*x + (rd-s)
    
    This gives 4 critical points, still not 12.
    
    For 12 critical points, we need a numerator of degree >= 12 in the derivative.
    V(x) = N(x)/(x+d) where N is degree k gives V'(x) numerator of degree k.
    For 12 roots, need k >= 12. But that's many parameters.
    
    Alternative: use V(x) = sum of 12 Gaussian-like bumps or a polynomial potential.
    Actually, let me use a different approach: parameterize roots via a smooth curve.
    
    Let me instead use the idea of representing roots as x_i = t_i / (1 - eps * u_i)
    where t_i are Laguerre roots and u_i are smooth perturbations.
    
    Actually, the simplest effective approach: use a 5-parameter rational ansatz
    V(x) = (a5*x^5 + a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0) / (x^2 + b1*x + b0)
    V'(x) numerator will be degree 6, giving up to 6 critical points. Still not 12.
    
    Let me take a completely different tack: use a parameterized family of 12 roots
    that smoothly deforms from Laguerre nodes. Parameterize as:
    x_i = x_i^0 * (1 + sum_k w_k * phi_k(i))
    where phi_k are orthogonal polynomials evaluated at normalized index i/n.
    """
    pass

def _parameterized_roots(params, base_roots):
    """Parameterize 12 roots as smooth deformation of base roots.
    params = [w1, w2, w3, w4, w5, w6] (6 params)
    x_i = base_i * (1 + w1*sin(pi*i/n) + w2*cos(2*pi*i/n) + w3*sin(2*pi*i/n) 
                   + w4*cos(3*pi*i/n) + w5*sin(3*pi*i/n) + w6*(i/(n-1)-0.5))
    with clipping to ensure positivity."""
    n = len(base_roots)
    idx = np.arange(n)
    norm = idx / (n - 1) if n > 1 else np.array([0.5])
    
    w = np.asarray(params)
    phase1 = np.sin(np.pi * norm)
    phase2 = np.cos(2*np.pi*norm)
    phase3 = np.sin(2*np.pi*norm)
    phase4 = np.cos(3*np.pi*norm)
    phase5 = np.sin(3*np.pi*norm)
    phase6 = norm - 0.5
    
    deform = 1.0 + 0.1*(w[0]*phase1 + w[1]*phase2 + w[2]*phase3 + w[3]*phase4 + w[4]*phase5 + w[5]*phase6)
    deform = np.maximum(deform, 0.1)  # ensure positive multiplier
    
    return base_roots * deform
=======
def _parameterized_roots(params, base_roots):
    """Parameterize 12 roots as smooth deformation of base roots.
    params = [w1, w2, w3, w4, w5, w6] (6 params)
    x_i = base_i * (1 + w1*sin(pi*i/n) + w2*cos(2*pi*i/n) + w3*sin(2*pi*i/n) 
                   + w4*cos(3*pi*i/n) + w5*sin(3*pi*i/n) + w6*(i/(n-1)-0.5))
    with clipping to ensure positivity."""
    n = len(base_roots)
    idx = np.arange(n)
    norm = idx / (n - 1) if n > 1 else np.array([0.5])
    
    w = np.asarray(params)
    phase1 = np.sin(np.pi * norm)
    phase2 = np.cos(2*np.pi*norm)
    phase3 = np.sin(2*np.pi*norm)
    phase4 = np.cos(3*np.pi*norm)
    phase5 = np.sin(3*np.pi*norm)
    phase6 = norm - 0.5
    
    deform = 1.0 + 0.1*(w[0]*phase1 + w[1]*phase2 + w[2]*phase3 + w[3]*phase4 + w[4]*phase5 + w[5]*phase6)
    deform = np.maximum(deform, 0.1)  # ensure positive multiplier
    
    return base_roots * deform
>>>>>>> REPLACE

