import numpy as np
from scipy.optimize import minimize
import time

def _hermite(n, x):
    """Compute Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h0 = np.ones_like(x)
    h1 = 2 * x
    for k in range(2, n + 1):
        h2 = 2 * x * h1 - 2 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1

def _compute_bound(a0, a4, a8):
    """Compute the uncertainty upper bound for given coefficients."""
    # H12 coefficient from g(0) = 0 constraint
    # H_n(0): H0=1, H4=12, H8=5040, H12=103950
    a12 = -(a0 + 12 * a4 + 5040 * a8) / 103950.0
    
    # Evaluate on a grid
    x = np.linspace(-10, 10, 10000)
    dx = x[1] - x[0]
    
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    g = (a0 * H0 + a4 * H4 + a8 * H8 + a12 * H12) * np.exp(-x**2 / 2)
    
    # Compute norms
    norm_g = np.sqrt(np.sum(g**2) * dx)
    
    if norm_g < 1e-15:
        return 1e10
    
    # For the uncertainty bound C4, we need to compute specific quantities
    # The bound typically involves ||g||_2, ||g'||_2, ||g''||_2, etc.
    # or relates to the Fourier transform properties
    
    # Compute derivative numerically
    dg = np.gradient(g, dx)
    d2g = np.gradient(dg, dx)
    
    # Also compute Fourier transform properties
    # For Hermite-Gaussians, the Fourier transform has specific properties
    # The uncertainty product A(f)*A(F) where A is some measure of spread
    
    # Based on the task context, C4 bound likely involves:
    # Computing the ratio that gives the upper bound on the uncertainty constant
    
    # Let's compute several candidate bounds and return the relevant one
    # The key quantity for Fourier uncertainty is often related to:
    # ||x*g||_2 * ||k*F(k)||_2 >= 1/4 (Heisenberg)
    # But C4 suggests a 4th order bound
    
    # For the specific formulation in the literature on this problem,
    # the bound involves computing integrals of |g|^2, |g'|^2, |g''|^2, |g'''|^2, |g''''|^2
    
    norm_g = np.sqrt(np.sum(g**2) * dx)
    norm_dg = np.sqrt(np.sum(dg**2) * dx)
    norm_d2g = np.sqrt(np.sum(d2g**2) * dx)
    
    # The C4 bound in the context of this problem likely refers to:
    # A bound on the product of norms that gives the uncertainty constant
    # Let's compute the standard Heisenberg-like bound and variants
    
    # Actually, for this specific problem (Goncalves et al. 2017 style),
    # the upper bound on C4 is computed as a specific ratio involving
    # the L2 norm and the norm of the 4th derivative or similar
    
    # Let me compute: bound = ||g''''||_2 / ||g||_2 type ratios
    d3g = np.gradient(d2g, dx)
    d4g = np.gradient(d3g, dx)
    norm_d4g = np.sqrt(np.sum(d4g**2) * dx)
    
    # The uncertainty constant bound might be:
    # C4_bound = (some combination of norms)
    
    # Based on the structure of the problem (Hermite polynomials up to degree 12),
    # the bound likely involves the ratio of higher-order derivative norms to lower-order norms
    
    # A common formulation: the bound on the uncertainty product is
    # related to ||g''||_2^2 / ||g||_2^2 or similar
    
    # Let me try: bound = ||g''||_2 / ||g||_2 * ||g||_2 / ||g'||_2 type products
    
    # Actually, re-reading the task: "reduce the upper bound on A(f)*A(Fourier(f))"
    # This is the Heisenberg uncertainty product. The "upper bound" C4 is what we minimize.
    
    # For a test function g, we compute:
    # Delta_x = sqrt(<x^2> - <x>^2) with respect to |g|^2
    # Delta_k = sqrt(<k^2> - <k>^2) with respect to |F(k)|^2
    # Then A(f)*A(F) >= Delta_x * Delta_k
    
    # But we want the UPPER BOUND that this test function provides on the universal constant
    
    # The way this works: for any f, A(f)*A(F) >= C. 
    # A test function g gives an upper bound C <= A(g)*A(F(g))
    # So we want to MINIMIZE A(g)*A(F(g)) to get the tightest upper bound
    
    # Compute the uncertainty product for g
    x_mean = np.sum(x * g**2) * dx / norm_g**2
    x2_mean = np.sum(x**2 * g**2) * dx / norm_g**2
    delta_x2 = x2_mean - x_mean**2
    
    # Fourier transform via FFT
    N = len(x)
    G = np.fft.fft(g) * dx
    k = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    
    G_norm = np.sqrt(np.sum(np.abs(G)**2) * dx / (2 * np.pi))
    
    if G_norm < 1e-15:
        return 1e10
    
    k_mean = np.sum(k * np.abs(G)**2) * dx / G_norm**2
    k2_mean = np.sum(k**2 * np.abs(G)**2) * dx / G_norm**2
    delta_k2 = k2_mean - k_mean**2
    
    if delta_x2 < 0 or delta_k2 < 0:
        return 1e10
    
    uncertainty_product = np.sqrt(delta_x2) * np.sqrt(delta_k2)
    
    return uncertainty_product

def construct():
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 210s limit
    
    # Baseline coefficients
    baseline = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_coeffs = baseline.copy()
    best_val = _compute_bound(*baseline)
    
    # Try optimization
    try:
        def objective(params):
            elapsed = time.time() - start_time
            if elapsed > deadline - 30:
                return best_val  # Return best known to stop optimization
            return _compute_bound(*params)
        
        # Multiple starting points
        starts = [
            baseline,
            np.array([-1.0, 0.05, 0.0003]),
            np.array([-1.1, 0.04, 0.0004]),
            np.array([-1.2, 0.06, 0.0002]),
            np.array([-0.9, 0.03, 0.0005]),
        ]
        
        for start in starts:
            elapsed = time.time() - start_time
            if elapsed > deadline - 60:
                break
            
            try:
                res = minimize(objective, start, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10,
                                    'adaptive': True})
                if res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
        
        # Also try COBYLA for potentially different landscape
        elapsed = time.time() - start_time
        if elapsed < deadline - 90:
            try:
                res = minimize(objective, baseline, method='COBYLA',
                             options={'maxiter': 3000, 'rhobeg': 0.1})
                if res.fun < best_val:
                    best_val = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
                
    except Exception:
        pass
    
    return best_coeffs
