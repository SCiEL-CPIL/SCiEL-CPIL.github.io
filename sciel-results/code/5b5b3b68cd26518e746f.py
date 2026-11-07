import numpy as np
from scipy.optimize import minimize
from sympy.polys.orthopolys import hermite_poly
import sympy as sp

# Precompute Hermite polynomials coefficients
def get_hermite_coeffs(n):
    x = sp.Symbol('x')
    Hn = hermite_poly(n, x)
    coeffs = sp.Poly(Hn, x).all_coeffs()
    return [float(c) for c in coeffs]

H0_coeffs = get_hermite_coeffs(0)
H4_coeffs = get_hermite_coeffs(4)
H8_coeffs = get_hermite_coeffs(8)
H12_coeffs = get_hermite_coeffs(12)

def pad_coeffs(coeffs, target_degree):
    padded = [0.0] * (target_degree - len(coeffs) + 1)
    padded += coeffs
    return np.array(padded)

H0_padded = pad_coeffs(H0_coeffs, 12)
H4_padded = pad_coeffs(H4_coeffs, 12)
H8_padded = pad_coeffs(H8_coeffs, 12)
H12_padded = pad_coeffs(H12_coeffs, 12)

def construct():
    c0_seed, c1_seed, c2_seed = -113 / 100, 1 / 25, 1 / 3240
    
    def compute_g_coeffs(c0, c1, c2):
        val_0 = c0 * H0_padded[-1] + c1 * H4_padded[-1] + c2 * H8_padded[-1]
        c3 = -val_0 / H12_padded[-1]
        g_coeffs = c0 * H0_padded + c1 * H4_padded + c2 * H8_padded + c3 * H12_padded
        return g_coeffs, c3
    
    def evaluate_value(c0, c1, c2):
        g_coeffs, c3 = compute_g_coeffs(c0, c1, c2)
        
        # Check leading coefficient sign
        lead_coeff = c3 * H12_padded[0]
        if lead_coeff <= 0:
            return None, None
        
        # Find roots of g
        try:
            roots = np.roots(g_coeffs)
        except Exception:
            return None, None
        
        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        positive_roots = real_roots[real_roots > 1e-10]
        
        if len(positive_roots) == 0:
            return 0.0, g_coeffs
        
        r_max = np.max(positive_roots)
        
        # Check non-negativity: g(x) >= 0 for all x >= 0
        # Dense grid check
        x_max = max(5.0, 3.0 * r_max)
        x_grid = np.linspace(0, x_max, 2000)
        g_vals = np.polyval(g_coeffs, x_grid)
        
        if np.any(g_vals < -1e-8):
            return None, None
        
        value = r_max ** 2 / (2 * np.pi)
        return value, g_coeffs
    
    # Try multiple starting points and local optimizations
    best_value = float('inf')
    best_coeffs = np.array([c0_seed, c1_seed, c2_seed])
    
    # Define objective for optimizer
    def objective(coeffs):
        c0, c1, c2 = coeffs
        value, _ = evaluate_value(c0, c1, c2)
        if value is None:
            return 1e10
        return value
    
    # Start with seed
    result = minimize(objective, np.array([c0_seed, c1_seed, c2_seed]), 
                      method='Nelder-Mead', 
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
    
    if result.fun < best_value:
        best_value = result.fun
        best_coeffs = result.x
    
    # Try perturbed starting points
    np.random.seed(42)
    for _ in range(20):
        perturbation = np.random.normal(0, 0.01, 3)
        start = np.array([c0_seed, c1_seed, c2_seed]) + perturbation
        result = minimize(objective, start, 
                        method='Nelder-Mead', 
                        options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True})
        
        if result.fun < best_value:
            best_value = result.fun
            best_coeffs = result.x
    
    return best_coeffs
