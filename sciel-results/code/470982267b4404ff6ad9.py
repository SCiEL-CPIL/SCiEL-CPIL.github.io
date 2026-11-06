"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad


def construct():
    """
    Construct an 8-node positive quadrature rule for the luminosity distance integral.
    
    Uses a basis-function projection approach:
    1. Represent the integrand 1/E(a) as a linear combination of physical basis functions
    2. Fit coefficients on a diverse cosmology sample
    3. Find optimal nodes and weights that minimize worst-case error
    
    The integral is: dL/(c/H0) = (1+z) * z * sum_i w_i / E(a_i)
    where a_i = 1/(1+z*x_i), x_i are nodes in [0,1], and E(a) = H(a)/H0.
    """
    
    n = 8
    
    # Define representative cosmologies for fitting
    # w0, wa pairs covering a diverse range
    cosmologies = [
        (0.0, 0.0),    # LCDM-like
        (-1.0, 0.0),   # Cosmological constant
        (-0.5, 0.5),   # Phantom-ish
        (-1.5, 1.0),   # Strongly evolving
        (0.5, -0.5),   # Quintessence-like
        (-1.0, 0.5),   # Mildly evolving
        (-0.8, -0.2),  # Near LCDM
        (1.0, 0.0),    # Exotic
        (-2.0, 2.0),   # Very strong evolution
        (0.0, 1.0),    # Matter-dominated with evolution
        (-0.3, 0.3),   # Moderate
        (-1.2, -0.3),  # Another variant
    ]
    
    # Test redshifts
    z_values = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0])
    
    def E_of_a(a, w0, wa):
        """Compute E(a) = H(a)/H0 for CPL model."""
        Omega_m = 0.3
        Omega_r = 0.0001
        Omega_de = 1.0 - Omega_m - Omega_r
        
        de_part = Omega_de * a**(-3*(1+w0+wa)) * np.exp(3*wa*(1-a))
        matter_part = Omega_m * a**(-3)
        radiation_part = Omega_r * a**(-4)
        
        return np.sqrt(matter_part + radiation_part + de_part)
    
    def integrand_z(z, w0, wa):
        """The integrand for the luminosity distance integral."""
        a = 1.0/(1.0+z)
        return z / E_of_a(a, w0, wa)
    
    def exact_integral(z, w0, wa):
        """Compute the exact integral using adaptive quadrature."""
        if z < 1e-10:
            return 0.0
        result, _ = quad(integrand_z, 0, z, args=(w0, wa), limit=200, epsabs=1e-12, epsrel=1e-12)
        return result
    
    def quadrature_integral(z, w0, wa, nodes, weights):
        """Compute the quadrature approximation."""
        a_vals = 1.0/(1.0 + z*nodes)
        E_vals = E_of_a(a_vals, w0, wa)
        return z * np.sum(weights * nodes / E_vals)
    
    def relative_error(z, w0, wa, nodes, weights):
        """Compute relative error for a given cosmology and redshift."""
        exact = exact_integral(z, w0, wa)
        approx = quadrature_integral(z, w0, wa, nodes, weights)
        if abs(exact) < 1e-15:
            return 0.0
        return abs(approx - exact) / abs(exact)
    
    # Initial guess: use Gauss-Legendre nodes as starting point
    # Gauss-Legendre 8-point nodes on [-1,1]
    x_gl = np.array([
        -0.9602898564975362,
        -0.7966664774136267,
        -0.5255324099163290,
        -0.1834346425002680,
         0.1834346425002680,
         0.5255324099163290,
         0.7966664774136267,
         0.9602898564975362
    ])
    w_gl = np.array([
        0.1012285362903763,
        0.2223810344533745,
        0.3137066458778873,
        0.3626837833783620,
        0.3626837833783620,
        0.3137066458778873,
        0.2223810344533745,
        0.1012285362903763
    ])
    
    # Map to [0,1]
    x0_nodes = (x_gl + 1.0) / 2.0
    x0_weights = w_gl / 2.0
    
    # Sort
    idx = np.argsort(x0_nodes)
    x0_nodes = x0_nodes[idx]
    x0_weights = x0_weights[idx]
    
    x0 = np.concatenate([x0_nodes, x0_weights])
    
    def objective(params):
        """Objective: minimize worst-case relative error (in percent)."""
        nodes = params[:n]
        weights = params[n:]
        
        # Enforce constraints: nodes in (0,1), increasing, weights positive, sum to 1
        if np.any(nodes <= 0) or np.any(nodes >= 1):
            return 1e6
        if np.any(np.diff(nodes) <= 0):
            return 1e6
        if np.any(weights <= 0):
            return 1e6
        if abs(np.sum(weights) - 1.0) > 1e-10:
            return 1e6
        
        max_err = 0.0
        for w0, wa in cosmologies:
            for z in z_values:
                err = relative_error(z, w0, wa, nodes, weights)
                if err > max_err:
                    max_err = err
                if max_err > 1.0:  # Early exit if error is huge
                    return 100.0
        
        return max_err * 100.0  # Convert to percent
    
    # Use Nelder-Mead for optimization with multiple restarts
    best_params = None
    best_score = np.inf
    
    # Several initial guesses
    initial_guesses = [
        x0,  # Gauss-Legendre
        np.concatenate([(np.arange(n) + 0.5)/n, np.ones(n)/n]),  # Midpoint
        np.concatenate([np.linspace(0.05, 0.95, n), np.ones(n)/n]),  # Uniform
        np.concatenate([np.sqrt(np.linspace(0.01, 0.99, n)), np.ones(n)/n]),  # Sqrt-spaced
        np.concatenate([1-np.sqrt(np.linspace(0.01, 0.99, n)), np.ones(n)/n]),  # Inverse sqrt
        np.concatenate([np.array([0.02, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]), np.ones(n)/n]),
    ]
    
    for x_init in initial_guesses:
        try:
            result = minimize(objective, x_init, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception as e:
            pass
    
    if best_params is None:
        # Fallback to Gauss-Legendre
        nodes = (x_gl + 1.0) / 2.0
        weights = w_gl / 2.0
        idx = np.argsort(nodes)
        nodes = nodes[idx]
        weights = weights[idx]
    else:
        nodes = best_params[:n]
        weights = best_params[n:]
        
        # Ensure nodes are strictly in (0,1) and increasing
        nodes = np.clip(nodes, 1e-6, 1.0 - 1e-6)
        nodes = np.sort(nodes)
        
        # Ensure weights are positive and sum to 1
        weights = np.maximum(weights, 1e-10)
        weights = weights / np.sum(weights)
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
